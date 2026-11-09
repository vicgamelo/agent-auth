"""Resource server side: protect an MCP server or any ASGI API with tokens from the authorization server.

    app = ProtectedResource(
        mcp_app,
        resource="https://mcp.example.com/mcp",
        issuer="https://auth.example.com",
        scopes_supported=["notes:read", "notes:write"],
        required_scopes=["notes:read"],
        tool_scopes={"add_note": "notes:write"},
    )

What it does on every request:

- serves Protected Resource Metadata (RFC 9728) so MCP clients can find the authorization server;
- answers 401 with ``WWW-Authenticate: Bearer resource_metadata=..., scope=...`` when there is no
  valid token, and 403 ``insufficient_scope`` with every scope the operation needs;
- validates the JWT locally: ES256 only, ``typ=at+jwt``, exact issuer, audience = this resource,
  expiry, and the token id against the revocation feed (refreshed every few seconds, fail closed
  when it goes stale);
- for MCP ``tools/call`` requests, looks up the scope each tool needs;
- puts the verified ``Principal`` in ``scope["state"]`` and reports each call to an audit sink.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

import httpx
import jwt
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from .config import canonical_resource
from .crypto import ACCESS_TOKEN_TYP, JWT_BEARER, SIGNING_ALG, SigningKey, client_assertion

log = logging.getLogger("agent_auth.guard")

PRINCIPAL_KEY = "agent_auth.principal"
ACCESS_TOKEN_TYPE = "urn:ietf:params:oauth:token-type:access_token"


class InvalidToken(Exception):
    pass


class AuthUnavailable(Exception):
    pass


@dataclass(frozen=True)
class Principal:
    """Who is calling. ``subject`` is the user (or the client itself for machine-to-machine tokens)."""

    subject: str
    kind: str  # "user" | "client"
    client_id: str
    scopes: frozenset[str]
    actors: tuple[str, ...]  # most recent first: the services the request went through
    token_id: str
    expires_at: int
    token: str = field(repr=False)

    @property
    def is_user(self) -> bool:
        return self.kind == "user"


def get_principal(scope_or_request: Any) -> Principal | None:
    """Reads the principal from an ASGI scope, a Starlette Request or anything with ``.scope``."""
    scope = scope_or_request if isinstance(scope_or_request, dict) else getattr(scope_or_request, "scope", {})
    value = scope.get("state", {}).get(PRINCIPAL_KEY)
    return value if isinstance(value, Principal) else None


class AuthorizationServerMetadata:
    """Fetches and caches RFC 8414 metadata for one issuer."""

    def __init__(self, issuer: str, http: httpx.AsyncClient) -> None:
        self.issuer = issuer.rstrip("/")
        self.http = http
        self._data: dict[str, Any] | None = None
        self._lock = asyncio.Lock()

    async def get(self) -> dict[str, Any]:
        if self._data is not None:
            return self._data
        async with self._lock:
            if self._data is None:
                response = await self.http.get(f"{self.issuer}/.well-known/oauth-authorization-server")
                response.raise_for_status()
                data = response.json()
                if data.get("issuer") != self.issuer:
                    raise AuthUnavailable("authorization server metadata names a different issuer")
                self._data = data
        return self._data

    async def endpoint(self, name: str) -> str:
        value = (await self.get()).get(name)
        if not isinstance(value, str):
            raise AuthUnavailable(f"authorization server does not advertise {name}")
        return value


class TokenVerifier:
    def __init__(
        self,
        issuer: str,
        resource: str,
        *,
        http: httpx.AsyncClient | None = None,
        clock: Callable[[], float] = time.time,
        leeway: int = 30,
        revocation_interval: float = 5.0,
        max_staleness: float = 60.0,
    ) -> None:
        self.issuer = issuer.rstrip("/")
        self.resource = canonical_resource(resource)
        self.http = http or httpx.AsyncClient(timeout=5.0)
        self.metadata = AuthorizationServerMetadata(self.issuer, self.http)
        self.clock = clock
        self.leeway = leeway
        self.revocation_interval = revocation_interval
        self.max_staleness = max_staleness
        self._keys: dict[str, Any] = {}
        self._jwks_fetched = -math.inf
        self._revoked: frozenset[str] = frozenset()
        self._revoked_fetched = -math.inf
        self._lock = asyncio.Lock()

    async def _key(self, kid: str | None) -> Any:
        if kid in self._keys:
            return self._keys[kid]
        now = self.clock()
        if now - self._jwks_fetched < 30:  # an unknown kid must not make us hammer the server
            raise InvalidToken("unknown signing key")
        async with self._lock:
            self._jwks_fetched = now
            try:
                response = await self.http.get(await self.metadata.endpoint("jwks_uri"))
                response.raise_for_status()
                keys = response.json().get("keys", [])
            except (httpx.HTTPError, ValueError) as exc:
                raise AuthUnavailable("cannot fetch signing keys") from exc
            self._keys = {
                k["kid"]: jwt.PyJWK(k).key
                for k in keys
                if k.get("kty") == "EC" and k.get("alg", SIGNING_ALG) == SIGNING_ALG
            }
        if kid not in self._keys:
            raise InvalidToken("unknown signing key")
        return self._keys[kid]

    async def revoked(self) -> frozenset[str]:
        now = self.clock()
        if now - self._revoked_fetched < self.revocation_interval:
            return self._revoked
        async with self._lock:
            if self.clock() - self._revoked_fetched < self.revocation_interval:
                return self._revoked
            try:
                response = await self.http.get(await self.metadata.endpoint("revoked_tokens_endpoint"))
                response.raise_for_status()
                self._revoked = frozenset(response.json()["revoked"])
                self._revoked_fetched = self.clock()
            except (httpx.HTTPError, ValueError, KeyError, AuthUnavailable) as exc:
                if self.clock() - self._revoked_fetched > self.max_staleness:
                    raise AuthUnavailable("revocation list unavailable") from exc
                log.warning("revocation list refresh failed, using the cached copy: %s", exc)
        return self._revoked

    async def verify(self, token: str) -> Principal:
        try:
            header = jwt.get_unverified_header(token)
        except jwt.PyJWTError as exc:
            raise InvalidToken("malformed token") from exc
        if header.get("alg") != SIGNING_ALG:
            raise InvalidToken("unexpected token algorithm")
        if str(header.get("typ", "")).lower() not in (ACCESS_TOKEN_TYP, "application/at+jwt"):
            raise InvalidToken("not an access token")
        key = await self._key(header.get("kid"))
        try:
            claims = jwt.decode(
                token,
                key,
                algorithms=[SIGNING_ALG],
                issuer=self.issuer,
                audience=self.resource,
                options={
                    "verify_exp": False,
                    "verify_iat": False,
                    "verify_nbf": False,
                    "require": ["iss", "sub", "aud", "exp", "iat", "jti", "client_id"],
                },
            )
        except jwt.InvalidAudienceError as exc:
            raise InvalidToken("token was issued for a different resource") from exc
        except jwt.PyJWTError as exc:
            raise InvalidToken("invalid token") from exc
        now = self.clock()
        if int(claims["exp"]) + self.leeway <= now:
            raise InvalidToken("token expired")
        if int(claims["iat"]) - self.leeway > now:
            raise InvalidToken("token issued in the future")
        if claims["jti"] in await self.revoked():
            raise InvalidToken("token revoked")
        actors: list[str] = []
        act = claims.get("act")
        while isinstance(act, dict) and len(actors) < 10:
            actors.append(str(act.get("sub")))
            act = act.get("act")
        return Principal(
            subject=str(claims["sub"]),
            kind="user" if claims.get("sub_kind", "user") == "user" else "client",
            client_id=str(claims["client_id"]),
            scopes=frozenset(str(claims.get("scope", "")).split()),
            actors=tuple(actors),
            token_id=str(claims["jti"]),
            expires_at=int(claims["exp"]),
            token=token,
        )


AuditSink = Callable[[dict[str, Any]], Awaitable[None]]


async def log_audit_sink(event: dict[str, Any]) -> None:
    log.info("audit %s", json.dumps(event, sort_keys=True))


def _quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


class ProtectedResource:
    def __init__(
        self,
        app: ASGIApp,
        *,
        resource: str,
        issuer: str,
        scopes_supported: Iterable[str],
        required_scopes: Iterable[str] = (),
        tool_scopes: Mapping[str, str | Iterable[str]] | None = None,
        require_user: bool = False,
        resource_name: str | None = None,
        public_paths: Iterable[str] = ("/health",),
        verifier: TokenVerifier | None = None,
        http: httpx.AsyncClient | None = None,
        audit: AuditSink | None = log_audit_sink,
        max_body_bytes: int = 1024 * 1024,
    ) -> None:
        self.app = app
        self.resource = canonical_resource(resource)
        self.issuer = issuer.rstrip("/")
        self.scopes_supported = list(scopes_supported)
        self.required_scopes = frozenset(required_scopes)
        self.tool_scopes = {
            name: frozenset([s] if isinstance(s, str) else s) for name, s in (tool_scopes or {}).items()
        }
        self.require_user = require_user
        self.resource_name = resource_name
        self.public_paths = frozenset(public_paths)
        self.verifier = verifier or TokenVerifier(self.issuer, self.resource, http=http)
        self.audit = audit
        self.max_body_bytes = max_body_bytes
        parts = urlsplit(self.resource)
        self.origin = f"{parts.scheme}://{parts.netloc}"
        self.metadata_path = "/.well-known/oauth-protected-resource" + parts.path
        self.metadata_url = self.origin + self.metadata_path
        self._audit_tasks: set[asyncio.Task[None]] = set()

    def metadata(self) -> dict[str, Any]:
        body: dict[str, Any] = {
            "resource": self.resource,
            "authorization_servers": [self.issuer],
            "scopes_supported": self.scopes_supported,
            "bearer_methods_supported": ["header"],
        }
        if self.resource_name:
            body["resource_name"] = self.resource_name
        return body

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        path = scope.get("path", "")
        if scope.get("method") == "GET" and path in (
            self.metadata_path,
            "/.well-known/oauth-protected-resource",
        ):
            await self._json(send, 200, self.metadata(), cache=True)
            return
        if path in self.public_paths:
            await self.app(scope, receive, send)
            return

        token = self._bearer(scope)
        if token is None:
            await self._challenge(send, 401, self.required_scopes, None, "authentication required")
            return
        try:
            principal = await self.verifier.verify(token)
        except InvalidToken as exc:
            await self._challenge(send, 401, self.required_scopes, "invalid_token", str(exc))
            return
        except AuthUnavailable as exc:
            log.error("authorization unavailable: %s", exc)
            await self._json(
                send, 503, {"error": "temporarily_unavailable"}, headers=[(b"retry-after", b"5")]
            )
            return

        receive, tools, method, too_large = await self._inspect_body(scope, receive)
        if too_large:
            await self._json(send, 413, {"error": "request_too_large"})
            return
        required = set(self.required_scopes)
        for name in tools:  # a JSON-RPC batch needs the scopes of every tool in it
            required |= self.tool_scopes.get(name, frozenset())
        tool = ",".join(tools) or None

        status_box = [0]

        async def tracking_send(message: Message) -> None:
            if message["type"] == "http.response.start":
                status_box[0] = int(message["status"])
            await send(message)

        if self.require_user and not principal.is_user:
            await self._json(
                tracking_send,
                403,
                {
                    "error": "user_required",
                    "error_description": "this resource only accepts tokens issued for a user",
                },
            )
        elif not required <= principal.scopes:
            await self._challenge(
                tracking_send,
                403,
                frozenset(required),
                "insufficient_scope",
                f"needs {' '.join(sorted(required))}",
            )
        else:
            scope.setdefault("state", {})[PRINCIPAL_KEY] = principal
            await self.app(scope, receive, tracking_send)
        self._report(principal, tool, method, status_box[0], scope)

    def _bearer(self, scope: Scope) -> str | None:
        values = [v for k, v in scope.get("headers", []) if k == b"authorization"]
        if len(values) != 1:
            return None
        raw: str = bytes(values[0]).decode("latin-1")
        kind, _, token = raw.partition(" ")
        if kind.lower() != "bearer" or not token.strip() or " " in token.strip():
            return None
        return token.strip()

    async def _inspect_body(
        self, scope: Scope, receive: Receive
    ) -> tuple[Receive, list[str], str | None, bool]:
        """Reads a JSON-RPC body (when there is one) to find the MCP tool, then replays it downstream."""
        if scope.get("method") != "POST":
            return receive, [], None, False
        content_type = next((v for k, v in scope.get("headers", []) if k == b"content-type"), b"").decode(
            "latin-1"
        )
        if "json" not in content_type:
            return receive, [], None, False
        chunks: list[bytes] = []
        size = 0
        more = True
        while more:
            message = await receive()
            if message["type"] != "http.request":
                break
            body = message.get("body", b"")
            size += len(body)
            if size > self.max_body_bytes:
                return receive, [], None, True
            chunks.append(body)
            more = message.get("more_body", False)
        body = b"".join(chunks)
        method: str | None = None
        try:
            payload = json.loads(body) if body else None
        except ValueError:
            payload = None
        items = payload if isinstance(payload, list) else [payload]
        tools: list[str] = []
        for item in items:
            if isinstance(item, dict):
                method = method or (item.get("method") if isinstance(item.get("method"), str) else None)
                params = item.get("params")
                if (
                    item.get("method") == "tools/call"
                    and isinstance(params, dict)
                    and isinstance(params.get("name"), str)
                ):
                    tools.append(params["name"])
        sent = False

        async def replay() -> Message:
            nonlocal sent
            if not sent:
                sent = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        return replay, tools, method, False

    def _report(
        self, principal: Principal, tool: str | None, method: str | None, status: int, scope: Scope
    ) -> None:
        if self.audit is None:
            return
        event = {
            "subject": principal.subject,
            "sub_kind": principal.kind,
            "client_id": principal.client_id,
            "actors": list(principal.actors),
            "tool": tool,
            "method": method or f"{scope.get('method')} {scope.get('path')}",
            "status": status,
            "resource": self.resource,
        }
        task = asyncio.ensure_future(self.audit(event))
        self._audit_tasks.add(task)
        task.add_done_callback(self._audit_tasks.discard)

    async def _challenge(
        self, send: Send, status: int, scopes: frozenset[str], error: str | None, description: str
    ) -> None:
        parts = [f"resource_metadata={_quote(self.metadata_url)}"]
        if scopes:
            parts.append(f"scope={_quote(' '.join(sorted(scopes)))}")
        if error:
            parts.append(f"error={_quote(error)}")
            parts.append(f"error_description={_quote(description)}")
        header = "Bearer " + ", ".join(parts)
        body = {"error": error or "unauthorized", "error_description": description}
        await self._json(send, status, body, headers=[(b"www-authenticate", header.encode())])

    async def _json(
        self,
        send: Send,
        status: int,
        body: dict[str, Any],
        *,
        cache: bool = False,
        headers: list[tuple[bytes, bytes]] | None = None,
    ) -> None:
        payload = json.dumps(body).encode()
        all_headers = [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(payload)).encode()),
            (b"cache-control", b"max-age=300" if cache else b"no-store"),
            *(headers or []),
        ]
        await send({"type": "http.response.start", "status": status, "headers": all_headers})
        await send({"type": "http.response.body", "body": payload})


class ServiceClient:
    """A confidential client (private_key_jwt): an MCP server exchanging tokens, or an M2M job."""

    def __init__(
        self,
        issuer: str,
        client_id: str,
        key: SigningKey,
        *,
        http: httpx.AsyncClient | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.issuer = issuer.rstrip("/")
        self.client_id = client_id
        self.key = key
        self.http = http or httpx.AsyncClient(timeout=5.0)
        self.metadata = AuthorizationServerMetadata(self.issuer, self.http)
        self.clock = clock
        self._cache: dict[tuple[str, ...], tuple[float, str]] = {}

    def _assertion(self) -> dict[str, str]:
        return {
            "client_id": self.client_id,
            "client_assertion_type": JWT_BEARER,
            "client_assertion": client_assertion(self.client_id, self.key, self.issuer),
        }

    async def _token(self, form: dict[str, str], cache_key: tuple[str, ...]) -> str:
        cached = self._cache.get(cache_key)
        if cached and cached[0] > self.clock() + 15:
            return cached[1]
        response = await self.http.post(
            await self.metadata.endpoint("token_endpoint"), data={**form, **self._assertion()}
        )
        if response.status_code != 200:
            body = (
                response.json()
                if response.headers.get("content-type", "").startswith("application/json")
                else {}
            )
            raise PermissionError(
                f"{body.get('error', response.status_code)}: {body.get('error_description', '')}"
            )
        data = response.json()
        token = str(data["access_token"])
        self._cache[cache_key] = (self.clock() + int(data.get("expires_in", 60)), token)
        if len(self._cache) > 1000:
            self._cache.clear()
        return token

    async def client_credentials(self, resource: str, scope: str | None = None) -> str:
        form = {"grant_type": "client_credentials", "resource": resource}
        if scope:
            form["scope"] = scope
        return await self._token(form, ("cc", resource, scope or ""))

    async def exchange(self, principal: Principal, resource: str, scope: str | None = None) -> str:
        """Trades the token this service received for one scoped to a downstream resource (RFC 8693).

        Never forward the incoming token itself: it was issued for this service only.
        """
        form = {
            "grant_type": "urn:ietf:params:oauth:grant-type:token-exchange",
            "subject_token": principal.token,
            "subject_token_type": ACCESS_TOKEN_TYPE,
            "resource": resource,
        }
        if scope:
            form["scope"] = scope
        return await self._token(form, ("tx", principal.token_id, resource, scope or ""))

    async def report(self, events: list[dict[str, Any]]) -> None:
        if not events:
            return
        response = await self.http.post(
            await self.metadata.endpoint("audit_events_endpoint"),
            data={"events": json.dumps(events), **self._assertion()},
        )
        if response.status_code >= 300:
            log.warning("audit report failed: HTTP %s", response.status_code)


def reporting_audit_sink(service: ServiceClient) -> AuditSink:
    """An audit sink that sends each call to the authorization server's activity log."""

    async def sink(event: dict[str, Any]) -> None:
        try:
            await service.report([event])
        except httpx.HTTPError as exc:
            log.warning("audit report failed: %s", exc)

    return sink
