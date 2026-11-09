"""The authorization server (OAuth 2.1 profile used by MCP 2026-07-28).

Endpoints:
  GET  /.well-known/oauth-authorization-server   RFC 8414 metadata
  GET  /jwks.json                                signing keys
  GET  /authorize                                authorization code + PKCE (S256 only), RFC 9207 iss
  POST /authorize/decision                       consent form
  POST /token                                    authorization_code, refresh_token, client_credentials,
                                                 token-exchange (RFC 8693)
  POST /register                                 Dynamic Client Registration (RFC 7591, throttled)
  POST /revoke                                   RFC 7009
  GET  /revoked                                  revoked, unexpired token ids (for resource servers)
  POST /audit/events                             resource servers report what agents did
  GET  /login, POST /login, POST /logout         user session
  GET  /consents, POST /consents/{id}/revoke     consent management
  GET  /activity                                 the user's audit trail
"""

from __future__ import annotations

import hmac
import json
import logging
import re
import time
from collections.abc import Callable, Mapping
from typing import Any
from urllib.parse import parse_qsl, urlencode

import jwt
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from starlette.routing import Route
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from . import pages
from .cimd import CIMDFetcher, Resolver, looks_like_cimd, system_resolver
from .clients import Client, ClientRegistry, OAuthError, dcr_grant_types
from .config import (
    AUTHORIZATION_CODE,
    CLIENT_CREDENTIALS,
    REFRESH_TOKEN,
    TOKEN_EXCHANGE,
    ConfigError,
    ResourceConfig,
    ServerConfig,
    canonical_resource,
)
from .crypto import ACCESS_TOKEN_TYP, SIGNING_ALG, SigningKey, pkce_s256, random_token, sha256_hex
from .identity import IdentityProvider
from .ratelimit import RateLimiter, client_ip, parse_networks
from .redirects import redirect_uri_matches, redirect_uri_problem
from .store import Store, join_scopes

log = logging.getLogger("agent_auth.server")

ACCESS_TOKEN_TYPE = "urn:ietf:params:oauth:token-type:access_token"
PKCE_VERIFIER = re.compile(r"^[A-Za-z0-9\-._~]{43,128}$")
PKCE_CHALLENGE = re.compile(r"^[A-Za-z0-9\-_]{43}$")
SCOPE_TOKEN = re.compile(r"^[\x21\x23-\x5b\x5d-\x7e]+$")  # RFC 6749 scope-token
NO_STORE = {"Cache-Control": "no-store", "Pragma": "no-cache"}


class SecurityHeaders:
    """Adds security headers to every response."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        path: str = scope.get("path", "")
        cacheable = path.startswith(("/static/", "/.well-known/", "/jwks.json"))

        async def wrapped(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                extra = {
                    b"x-content-type-options": b"nosniff",
                    b"referrer-policy": b"no-referrer",
                    b"x-frame-options": b"DENY",
                    b"content-security-policy": (
                        b"default-src 'none'; style-src 'self'; img-src 'self'; "
                        b"frame-ancestors 'none'; base-uri 'none'"
                    ),
                }
                if not cacheable:
                    extra[b"cache-control"] = b"no-store"
                present = {k.lower() for k, _ in headers}
                headers += [(k, v) for k, v in extra.items() if k not in present]
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, wrapped)


def _split_scopes(value: str | None) -> set[str]:
    if not value:
        return set()
    scopes = set(value.split())
    if any(not SCOPE_TOKEN.match(s) for s in scopes):
        raise OAuthError("invalid_scope", "malformed scope")
    return scopes


def _append_query(uri: str, params: Mapping[str, str | None]) -> str:
    query = urlencode({k: v for k, v in params.items() if v is not None})
    return f"{uri}{'&' if '?' in uri else '?'}{query}"


def _safe_next(value: str | None, default: str = "/consents") -> str:
    if not value or not value.startswith("/") or value.startswith(("//", "/\\")) or "\\" in value:
        return default
    return value


def actor_chain(claims: Mapping[str, Any]) -> list[str]:
    """Actors from the most recent to the first, e.g. ['notes-mcp', 'dcr_abc']."""
    chain: list[str] = []
    act = claims.get("act")
    while isinstance(act, dict) and len(chain) < 10:
        chain.append(str(act.get("sub")))
        act = act.get("act")
    return chain


class AuthorizationServer:
    def __init__(
        self,
        config: ServerConfig,
        identity: IdentityProvider,
        *,
        store: Store | None = None,
        signing_key: SigningKey | None = None,
        clock: Callable[[], float] = time.time,
        cimd_transport: Any = None,
        cimd_resolver: Resolver = system_resolver,
    ) -> None:
        from pathlib import Path

        self.config = config
        self.identity = identity
        self.store = store or Store(config.database)
        if signing_key is None:
            signing_key = (
                SigningKey.load_or_create(Path(config.signing_key_path))
                if config.signing_key_path
                else SigningKey.generate()
            )
        self.key = signing_key
        self._public_key = jwt.PyJWK(signing_key.public_jwk).key
        self.clock = clock
        self.fetcher = CIMDFetcher(
            transport=cimd_transport,
            resolver=cimd_resolver,
            allow_loopback=config.allow_loopback_cimd,
            cache_ttl=config.limits.cimd_cache_ttl,
            clock=clock,
        )
        self.clients = ClientRegistry(config, self.store, self.fetcher, clock)
        self.trusted = parse_networks(config.trusted_proxies)
        lim = config.limits
        self.dcr_limiter = RateLimiter(lim.dcr_per_ip_per_hour / 60, lim.dcr_burst, clock=clock)
        self.login_limiter = RateLimiter(
            lim.login_per_ip_per_minute, max(1, int(lim.login_per_ip_per_minute)), clock=clock
        )
        self.token_limiter = RateLimiter(
            lim.token_per_ip_per_minute, max(1, int(lim.token_per_ip_per_minute)), clock=clock
        )
        self._login_failures: dict[str, tuple[int, float]] = {}
        self._last_prune = 0.0
        self.cookie = "__Host-agent_auth_session" if config.secure_cookies else "agent_auth_session"
        self.login_cookie = "__Host-agent_auth_login" if config.secure_cookies else "agent_auth_login"
        self.token_endpoint = f"{config.issuer}/token"
        routes = [
            Route("/.well-known/oauth-authorization-server", self.metadata, methods=["GET"]),
            Route("/jwks.json", self.jwks, methods=["GET"]),
            Route("/authorize", self.authorize, methods=["GET"]),
            Route("/authorize/decision", self.decision, methods=["POST"]),
            Route("/token", self.token, methods=["POST"]),
            Route("/register", self.register, methods=["POST"]),
            Route("/revoke", self.revoke, methods=["POST"]),
            Route("/revoked", self.revoked, methods=["GET"]),
            Route("/audit/events", self.audit_events, methods=["POST"]),
            Route("/login", self.login_form, methods=["GET"]),
            Route("/login", self.login, methods=["POST"]),
            Route("/logout", self.logout, methods=["POST"]),
            Route("/consents", self.consents, methods=["GET"]),
            Route("/consents/{grant_id}/revoke", self.revoke_consent, methods=["POST"]),
            Route("/activity", self.activity, methods=["GET"]),
            Route("/static/agent-auth.css", self.css, methods=["GET"]),
            Route("/health", self.health, methods=["GET"]),
        ]
        self.app = SecurityHeaders(Starlette(routes=routes))

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        await self.app(scope, receive, send)

    # ---- helpers ---------------------------------------------------------------------------

    def now(self) -> float:
        return self.clock()

    def ip(self, request: Request) -> str:
        return client_ip(request.scope, self.trusted)

    def audit(self, event: str, **fields: Any) -> None:
        self.store.audit(self.now(), event, **fields)

    async def _body(self, request: Request) -> bytes:
        limit = self.config.limits.max_body_bytes
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > limit:
            raise OAuthError("invalid_request", "request body too large", 413)
        body = b""
        async for chunk in request.stream():
            body += chunk
            if len(body) > limit:
                raise OAuthError("invalid_request", "request body too large", 413)
        return body

    async def _form(self, request: Request) -> dict[str, str]:
        ctype = request.headers.get("content-type", "").split(";")[0].strip().lower()
        if ctype != "application/x-www-form-urlencoded":
            raise OAuthError("invalid_request", "expected application/x-www-form-urlencoded")
        pairs = parse_qsl((await self._body(request)).decode("utf-8", "replace"), keep_blank_values=True)
        form: dict[str, str] = {}
        for key, value in pairs:
            if key in form:
                # RFC 6749 3.1/3.2: parameters must not be repeated. Several `resource`s are also refused.
                raise OAuthError("invalid_request", f"parameter {key} repeated")
            form[key] = value
        return form

    def _error(self, exc: OAuthError) -> JSONResponse:
        headers = dict(NO_STORE)
        if exc.status == 401:
            headers["WWW-Authenticate"] = 'Bearer error="invalid_client"'
        return JSONResponse(
            {"error": exc.error, "error_description": exc.description},
            status_code=exc.status,
            headers=headers,
        )

    def _same_origin(self, request: Request) -> bool:
        origin = request.headers.get("origin")
        if origin is not None and origin != self.config.issuer:
            return False
        return request.headers.get("sec-fetch-site") not in ("cross-site", "same-site")

    def _session(self, request: Request) -> tuple[str, str, str] | None:
        """(session hash, user id, csrf token) for a valid session cookie."""
        sid = request.cookies.get(self.cookie)
        if not sid:
            return None
        row = self.store.get_session(sha256_hex(sid), self.now())
        if row is None:
            return None
        return row["id_hash"], row["user_id"], row["csrf"]

    def _cookie(self, response: Response, name: str, value: str, max_age: int) -> None:
        response.set_cookie(
            name,
            value,
            max_age=max_age,
            path="/",
            httponly=True,
            secure=self.config.secure_cookies,
            samesite="lax",
        )

    def _maybe_prune(self) -> None:
        now = self.now()
        if now - self._last_prune > 60:
            self._last_prune = now
            self.store.prune(now)

    # ---- discovery -------------------------------------------------------------------------

    async def metadata(self, request: Request) -> JSONResponse:
        iss = self.config.issuer
        grant_types = [AUTHORIZATION_CODE, REFRESH_TOKEN, CLIENT_CREDENTIALS, TOKEN_EXCHANGE]
        body: dict[str, Any] = {
            "issuer": iss,
            "authorization_endpoint": f"{iss}/authorize",
            "token_endpoint": self.token_endpoint,
            "jwks_uri": f"{iss}/jwks.json",
            "revocation_endpoint": f"{iss}/revoke",
            "revocation_endpoint_auth_methods_supported": ["none", "private_key_jwt"],
            "response_types_supported": ["code"],
            "response_modes_supported": ["query"],
            "grant_types_supported": grant_types,
            "code_challenge_methods_supported": ["S256"],
            "token_endpoint_auth_methods_supported": ["none", "private_key_jwt"],
            "token_endpoint_auth_signing_alg_values_supported": ["ES256", "RS256", "EdDSA"],
            "scopes_supported": self.config.all_scopes(),
            "authorization_response_iss_parameter_supported": True,
            "client_id_metadata_document_supported": self.config.limits.cimd_enabled,
            # Not standard: where resource servers poll revoked token ids.
            "revoked_tokens_endpoint": f"{iss}/revoked",
            "audit_events_endpoint": f"{iss}/audit/events",
        }
        if self.config.limits.dcr_enabled:
            body["registration_endpoint"] = f"{iss}/register"
        return JSONResponse(body)

    async def jwks(self, request: Request) -> JSONResponse:
        return JSONResponse(self.key.jwks())

    async def css(self, request: Request) -> Response:
        return Response(pages.CSS, media_type="text/css", headers={"Cache-Control": "max-age=3600"})

    async def health(self, request: Request) -> JSONResponse:
        return JSONResponse({"status": "ok"})

    # ---- authorization endpoint ------------------------------------------------------------

    async def authorize(self, request: Request) -> Response:
        multi = request.query_params.multi_items()
        params: dict[str, str] = {}
        repeated = set()
        for k, v in multi:
            if k in params:
                repeated.add(k)
            params[k] = v

        client_id = params.get("client_id", "")
        client = await self.clients.get(client_id)
        if client is None:
            detail = "This application is not registered."
            if looks_like_cimd(client_id):
                problem = await self.clients.cimd_problem(client_id)
                detail = f"The client metadata document was refused: {problem}."
            self.audit("authorize.rejected", client_id=client_id[:200] or None, reason="unknown client")
            return HTMLResponse(pages.error_page("Unknown application", detail), status_code=400)
        redirect_uri = params.get("redirect_uri", "")
        if "redirect_uri" in repeated or not redirect_uri_matches(redirect_uri, client.redirect_uris):
            # Never redirect to an unregistered URI: that would be an open redirect.
            self.audit("authorize.rejected", client_id=client.client_id, reason="redirect_uri mismatch")
            return HTMLResponse(
                pages.error_page(
                    "Invalid redirect", "The redirect address is not registered for this application."
                ),
                status_code=400,
            )
        state = params.get("state")

        def fail(error: str, description: str) -> Response:
            self.audit("authorize.rejected", client_id=client.client_id, reason=f"{error}: {description}")
            location = _append_query(
                redirect_uri,
                {"error": error, "error_description": description, "state": state, "iss": self.config.issuer},
            )
            return RedirectResponse(location, status_code=302)

        if repeated:
            return fail("invalid_request", f"parameters repeated: {', '.join(sorted(repeated))}")
        if AUTHORIZATION_CODE not in client.grant_types:
            return fail("unauthorized_client", "client may not use the authorization code grant")
        if params.get("response_type") != "code":
            return fail("unsupported_response_type", "only response_type=code is supported")
        challenge = params.get("code_challenge")
        if not challenge:
            return fail("invalid_request", "code_challenge is required (PKCE)")
        if params.get("code_challenge_method") != "S256":
            return fail("invalid_request", "code_challenge_method must be S256")
        if not PKCE_CHALLENGE.match(challenge):
            return fail("invalid_request", "malformed code_challenge")
        resource_uri = params.get("resource")
        if not resource_uri:
            return fail("invalid_target", "resource is required (RFC 8707)")
        resource = self.config.resource(resource_uri)
        if resource is None:
            return fail("invalid_target", "unknown resource")
        try:
            scopes = _split_scopes(params.get("scope")) or set(resource.default_scopes)
        except OAuthError as exc:
            return fail(exc.error, exc.description)
        if not scopes:
            return fail("invalid_scope", "no scope requested and the resource has no default")
        unknown = scopes - set(resource.scopes)
        if unknown:
            return fail("invalid_scope", f"scopes not offered by this resource: {' '.join(sorted(unknown))}")
        prompt = set(params.get("prompt", "").split())

        session = self._session(request)
        if session is None:
            if "none" in prompt:
                return fail("login_required", "user is not signed in")
            target = "/authorize?" + request.url.query
            return RedirectResponse("/login?" + urlencode({"next": target}), status_code=303)
        session_hash, user_id, csrf = session

        request_params = {
            "client_id": client.client_id,
            "redirect_uri": redirect_uri,
            "state": state,
            "code_challenge": challenge,
            "resource": resource.uri,
            "scopes": sorted(scopes),
        }
        grant = self.store.active_grant(user_id, client.client_id, resource.uri)
        if grant is not None and scopes <= grant.scopes and "consent" not in prompt:
            return self._issue_code(user_id, grant.id, request_params, status=302)
        if "none" in prompt:
            return fail("consent_required", "user consent is required")

        pending_id = random_token()
        self.store.add_pending(
            pending_id, session_hash, request_params, self.now() + self.config.limits.pending_ttl
        )
        delegates = [r for r in (self.config.resource(d) for d in resource.delegates_to) if r is not None]
        html = pages.consent_page(
            pending_id=pending_id,
            csrf=csrf,
            user_id=user_id,
            client_name=client.name,
            client_id=client.client_id,
            verified=client.verified,
            redirect_uri=redirect_uri,
            resource=resource,
            scopes=scopes,
            delegates=delegates,
        )
        return HTMLResponse(html)

    def _issue_code(self, user_id: str, grant_id: str, p: dict[str, Any], status: int) -> Response:
        code = random_token()
        self.store.add_code(
            sha256_hex(code),
            grant_id=grant_id,
            client_id=p["client_id"],
            user_id=user_id,
            redirect_uri=p["redirect_uri"],
            code_challenge=p["code_challenge"],
            resource=p["resource"],
            scopes=" ".join(p["scopes"]),
            expires_at=self.now() + self.config.limits.code_ttl,
        )
        location = _append_query(
            p["redirect_uri"], {"code": code, "state": p["state"], "iss": self.config.issuer}
        )
        return RedirectResponse(location, status_code=status)

    async def decision(self, request: Request) -> Response:
        try:
            form = await self._form(request)
        except OAuthError as exc:
            return HTMLResponse(pages.error_page("Bad request", exc.description), status_code=400)
        session = self._session(request)
        if session is None or not self._same_origin(request):
            return HTMLResponse(pages.error_page("Session expired", "Please start again."), status_code=403)
        session_hash, user_id, csrf = session
        if not hmac.compare_digest(form.get("csrf", ""), csrf):
            return HTMLResponse(pages.error_page("Session expired", "Please start again."), status_code=403)
        p = self.store.take_pending(form.get("request", ""), session_hash, self.now())
        if p is None:
            return HTMLResponse(pages.error_page("Request expired", "Please start again."), status_code=400)
        if form.get("decision") != "approve":
            self.audit("consent.denied", user_id=user_id, client_id=p["client_id"], resource=p["resource"])
            location = _append_query(
                p["redirect_uri"],
                {
                    "error": "access_denied",
                    "error_description": "the user denied access",
                    "state": p["state"],
                    "iss": self.config.issuer,
                },
            )
            return RedirectResponse(location, status_code=303)
        grant = self.store.upsert_grant(
            random_token(16), user_id, p["client_id"], p["resource"], frozenset(p["scopes"]), self.now()
        )
        self.audit(
            "consent.granted",
            user_id=user_id,
            client_id=p["client_id"],
            resource=p["resource"],
            scope=" ".join(p["scopes"]),
            grant_id=grant.id,
        )
        return self._issue_code(user_id, grant.id, p, status=303)

    # ---- token endpoint --------------------------------------------------------------------

    async def token(self, request: Request) -> Response:
        try:
            retry = self.token_limiter.take(self.ip(request))
            if retry:
                raise OAuthError("slow_down", "too many token requests", 429)
            self._maybe_prune()
            form = await self._form(request)
            grant_type = form.get("grant_type", "")
            client = await self.clients.authenticate(form, self.token_endpoint)
            if grant_type not in client.grant_types:
                raise OAuthError(
                    "unauthorized_client", f"client may not use grant_type {grant_type or '(none)'}"
                )
            if grant_type == AUTHORIZATION_CODE:
                body = self._grant_authorization_code(client, form)
            elif grant_type == REFRESH_TOKEN:
                body = self._grant_refresh(client, form)
            elif grant_type == CLIENT_CREDENTIALS:
                body = self._grant_client_credentials(client, form)
            elif grant_type == TOKEN_EXCHANGE:
                body = self._grant_token_exchange(client, form)
            else:  # pragma: no cover - grant_types are validated in config
                raise OAuthError("unsupported_grant_type", "unsupported grant_type")
        except OAuthError as exc:
            return self._error(exc)
        if client.kind == "dcr":
            self.store.touch_client(client.client_id, self.now())
        return JSONResponse(body, headers=NO_STORE)

    def _issue_access(
        self,
        *,
        subject: str,
        sub_kind: str,
        client_id: str,
        resource: str,
        scopes: set[str] | frozenset[str],
        grant_id: str | None = None,
        family: str | None = None,
        parent_jti: str | None = None,
        act: dict[str, Any] | None = None,
        exp_cap: int | None = None,
    ) -> tuple[str, int]:
        now = int(self.now())
        exp = now + self.config.limits.access_token_ttl
        if exp_cap is not None:
            exp = min(exp, exp_cap)
        jti = random_token(16)
        claims: dict[str, Any] = {
            "iss": self.config.issuer,
            "sub": subject,
            "aud": resource,
            "client_id": client_id,
            "scope": join_scopes(set(scopes)),
            "iat": now,
            "exp": exp,
            "jti": jti,
            "sub_kind": sub_kind,
        }
        if act:
            claims["act"] = act
        self.store.add_access(
            jti,
            grant_id=grant_id,
            family=family,
            parent_jti=parent_jti,
            client_id=client_id,
            subject=subject,
            resource=resource,
            expires_at=exp,
        )
        return self.key.sign(claims, typ=ACCESS_TOKEN_TYP), exp - now

    def _issue_refresh(
        self,
        *,
        family: str,
        grant_id: str,
        client_id: str,
        user_id: str,
        resource: str,
        scopes: str,
        expires_at: float,
    ) -> str:
        token = random_token()
        self.store.add_refresh(
            sha256_hex(token),
            family=family,
            grant_id=grant_id,
            client_id=client_id,
            user_id=user_id,
            resource=resource,
            scopes=scopes,
            expires_at=expires_at,
        )
        return token

    def _resource_param(self, form: Mapping[str, str], expected: str) -> None:
        requested = form.get("resource")
        if requested is None:
            return
        try:
            same = canonical_resource(requested) == expected
        except ConfigError:
            same = False
        if not same:
            raise OAuthError("invalid_target", "resource does not match the authorization")

    def _grant_authorization_code(self, client: Client, form: Mapping[str, str]) -> dict[str, Any]:
        code = form.get("code")
        if not code:
            raise OAuthError("invalid_request", "code is required")
        code_hash = sha256_hex(code)
        row = self.store.get_code(code_hash)
        now = self.now()
        if row is None or row["client_id"] != client.client_id:
            raise OAuthError("invalid_grant", "invalid authorization code")
        if row["used_at"] is not None or not self.store.mark_code_used(code_hash, now):
            # Replay: someone has a copy of the code. Kill everything issued from it (OAuth 2.1 4.1.3).
            self.store.revoke_family(code_hash, now)
            self.audit(
                "code.replay",
                user_id=row["user_id"],
                client_id=client.client_id,
                resource=row["resource"],
                reason="authorization code used twice; tokens from it revoked",
            )
            raise OAuthError("invalid_grant", "authorization code already used")
        if row["expires_at"] < now:
            raise OAuthError("invalid_grant", "authorization code expired")
        verifier = form.get("code_verifier", "")
        if not PKCE_VERIFIER.match(verifier):
            raise OAuthError("invalid_grant", "code_verifier missing or malformed")
        if not hmac.compare_digest(pkce_s256(verifier), row["code_challenge"]):
            raise OAuthError("invalid_grant", "PKCE verification failed")
        if form.get("redirect_uri") != row["redirect_uri"]:
            raise OAuthError("invalid_grant", "redirect_uri does not match the authorization request")
        self._resource_param(form, row["resource"])
        grant = self.store.get_grant(row["grant_id"])
        if grant is None or grant.revoked_at is not None:
            raise OAuthError("invalid_grant", "consent was revoked")
        scopes = frozenset(row["scopes"].split())
        access, expires_in = self._issue_access(
            subject=row["user_id"],
            sub_kind="user",
            client_id=client.client_id,
            resource=row["resource"],
            scopes=scopes,
            grant_id=grant.id,
            family=code_hash,
        )
        body: dict[str, Any] = {
            "access_token": access,
            "token_type": "Bearer",
            "expires_in": expires_in,
            "scope": row["scopes"],
        }
        if REFRESH_TOKEN in client.grant_types:
            body["refresh_token"] = self._issue_refresh(
                family=code_hash,
                grant_id=grant.id,
                client_id=client.client_id,
                user_id=row["user_id"],
                resource=row["resource"],
                scopes=row["scopes"],
                expires_at=now + self.config.limits.refresh_token_ttl,
            )
        self.audit(
            "token.issued",
            user_id=row["user_id"],
            client_id=client.client_id,
            resource=row["resource"],
            grant_type=AUTHORIZATION_CODE,
            scope=row["scopes"],
        )
        return body

    def _grant_refresh(self, client: Client, form: Mapping[str, str]) -> dict[str, Any]:
        token = form.get("refresh_token")
        if not token:
            raise OAuthError("invalid_request", "refresh_token is required")
        token_hash = sha256_hex(token)
        row = self.store.get_refresh(token_hash)
        now = self.now()
        if row is None or row["client_id"] != client.client_id or row["revoked_at"] is not None:
            raise OAuthError("invalid_grant", "invalid refresh token")
        if row["rotated_at"] is not None or not self.store.rotate_refresh(token_hash, now):
            # A rotated token came back: either the client or an attacker holds a stolen copy.
            self.store.revoke_family(row["family"], now)
            self.audit(
                "refresh.reuse",
                user_id=row["user_id"],
                client_id=client.client_id,
                resource=row["resource"],
                reason="rotated refresh token reused; token family revoked",
            )
            raise OAuthError("invalid_grant", "refresh token already used")
        if row["expires_at"] < now:
            raise OAuthError("invalid_grant", "refresh token expired")
        grant = self.store.get_grant(row["grant_id"])
        if grant is None or grant.revoked_at is not None:
            raise OAuthError("invalid_grant", "consent was revoked")
        self._resource_param(form, row["resource"])
        granted = frozenset(row["scopes"].split())
        requested = _split_scopes(form.get("scope")) or set(granted)
        if not requested <= granted:
            raise OAuthError("invalid_scope", "refresh cannot widen scope")
        access, expires_in = self._issue_access(
            subject=row["user_id"],
            sub_kind="user",
            client_id=client.client_id,
            resource=row["resource"],
            scopes=requested,
            grant_id=grant.id,
            family=row["family"],
        )
        new_refresh = self._issue_refresh(
            family=row["family"],
            grant_id=grant.id,
            client_id=client.client_id,
            user_id=row["user_id"],
            resource=row["resource"],
            scopes=row["scopes"],
            expires_at=row["expires_at"],
        )
        self.audit(
            "token.refreshed",
            user_id=row["user_id"],
            client_id=client.client_id,
            resource=row["resource"],
            scope=join_scopes(requested),
        )
        return {
            "access_token": access,
            "token_type": "Bearer",
            "expires_in": expires_in,
            "scope": join_scopes(requested),
            "refresh_token": new_refresh,
        }

    def _grant_client_credentials(self, client: Client, form: Mapping[str, str]) -> dict[str, Any]:
        if client.auth_method != "private_key_jwt":
            raise OAuthError("unauthorized_client", "client_credentials needs a confidential client")
        target = form.get("resource")
        resource = self.config.resource(target) if target else None
        if resource is None or resource.uri not in client.allowed_resources:
            raise OAuthError("invalid_target", "resource missing or not allowed for this client")
        allowed = set(client.allowed_scopes) & set(resource.scopes)
        requested = _split_scopes(form.get("scope")) or allowed
        if not requested or not requested <= allowed:
            raise OAuthError("invalid_scope", "scope not allowed for this client")
        access, expires_in = self._issue_access(
            subject=client.client_id,
            sub_kind="client",
            client_id=client.client_id,
            resource=resource.uri,
            scopes=requested,
        )
        self.audit(
            "token.issued",
            client_id=client.client_id,
            resource=resource.uri,
            grant_type=CLIENT_CREDENTIALS,
            scope=join_scopes(requested),
        )
        return {
            "access_token": access,
            "token_type": "Bearer",
            "expires_in": expires_in,
            "scope": join_scopes(requested),
        }

    def verify_own_token(self, token: str) -> dict[str, Any]:
        """Validates an access token this server issued (signature, type, issuer, expiry)."""
        try:
            header = jwt.get_unverified_header(token)
            if header.get("typ", "").lower() not in (ACCESS_TOKEN_TYP, "application/at+jwt"):
                raise OAuthError("invalid_request", "subject_token is not an access token")
            claims: dict[str, Any] = jwt.decode(
                token,
                self._public_key,
                algorithms=[SIGNING_ALG],
                issuer=self.config.issuer,
                options={
                    "verify_aud": False,
                    "verify_exp": False,
                    "verify_iat": False,
                    "verify_nbf": False,
                    "require": ["exp", "jti", "sub", "aud", "client_id"],
                },
            )
        except jwt.PyJWTError as exc:
            raise OAuthError("invalid_request", "subject_token is invalid") from exc
        if int(claims["exp"]) <= self.now():
            raise OAuthError("invalid_request", "subject_token expired")
        return claims

    def _grant_token_exchange(self, client: Client, form: Mapping[str, str]) -> dict[str, Any]:
        if client.auth_method != "private_key_jwt" or not client.serves_resource:
            raise OAuthError(
                "unauthorized_client", "token exchange needs a confidential resource-server client"
            )
        if form.get("actor_token") or form.get("actor_token_type"):
            # The actor is whoever authenticated. Accepting a caller-supplied actor_token lets a
            # compromised intermediary splice someone else's identity into the chain.
            raise OAuthError(
                "invalid_request", "actor_token is not accepted; the authenticated client is the actor"
            )
        if form.get("subject_token_type") != ACCESS_TOKEN_TYPE:
            raise OAuthError("invalid_request", "subject_token_type must be an access token")
        if form.get("requested_token_type") not in (None, ACCESS_TOKEN_TYPE):
            raise OAuthError("invalid_request", "only access tokens can be requested")
        subject_token = form.get("subject_token", "")
        claims = self.verify_own_token(subject_token)
        row = self.store.get_access(claims["jti"])
        if row is None or row["revoked_at"] is not None:
            raise OAuthError("invalid_request", "subject_token was revoked")
        user_id = claims["sub"] if claims.get("sub_kind") == "user" else None
        if claims["aud"] != client.serves_resource:
            # Confused deputy: a token sent to some other service is being cashed in here.
            self.audit(
                "exchange.rejected",
                user_id=user_id,
                client_id=client.client_id,
                resource=claims["aud"],
                reason="subject_token audience is not the requesting service",
            )
            raise OAuthError("invalid_request", "subject_token was not issued to the requesting client")
        target_uri = form.get("resource") or form.get("audience")
        target = self.config.resource(target_uri) if target_uri else None
        source = self.config.resource(client.serves_resource)
        if target is None or source is None or target.uri not in source.delegates_to:
            self.audit(
                "exchange.rejected",
                user_id=user_id,
                client_id=client.client_id,
                resource=target_uri,
                reason="target resource not delegated",
            )
            raise OAuthError("invalid_target", "this service may not act on that resource")
        subject_scopes = set(claims.get("scope", "").split())
        allowed = subject_scopes & set(target.scopes)
        requested = _split_scopes(form.get("scope")) or allowed
        if not requested or not requested <= allowed:
            self.audit(
                "exchange.rejected",
                user_id=user_id,
                client_id=client.client_id,
                resource=target.uri,
                reason="scope wider than the subject token",
                scope=join_scopes(requested),
            )
            raise OAuthError("invalid_scope", "exchanged token cannot have more scope than the subject token")
        prior = claims.get("act") or {"sub": claims["client_id"]}
        act = {"sub": client.client_id, "act": prior}
        chain = actor_chain({"act": act})
        if len(chain) > self.config.limits.max_actor_depth:
            raise OAuthError("invalid_request", "delegation chain too long")
        access, expires_in = self._issue_access(
            subject=claims["sub"],
            sub_kind=claims.get("sub_kind", "user"),
            client_id=client.client_id,
            resource=target.uri,
            scopes=requested,
            grant_id=row["grant_id"],
            family=row["family"],
            parent_jti=claims["jti"],
            act=act,
            exp_cap=int(claims["exp"]),
        )
        self.audit(
            "token.exchanged",
            user_id=user_id,
            client_id=client.client_id,
            resource=target.uri,
            scope=join_scopes(requested),
            actors=chain,
        )
        return {
            "access_token": access,
            "issued_token_type": ACCESS_TOKEN_TYPE,
            "token_type": "Bearer",
            "expires_in": expires_in,
            "scope": join_scopes(requested),
        }

    # ---- dynamic client registration -------------------------------------------------------

    async def register(self, request: Request) -> Response:
        lim = self.config.limits
        if not lim.dcr_enabled:
            return JSONResponse({"error": "not_found"}, status_code=404)
        ip = self.ip(request)
        retry = self.dcr_limiter.take(ip)
        if retry:
            self.audit("registration.rejected", reason="rate limited")
            return JSONResponse(
                {
                    "error": "too_many_requests",
                    "error_description": "too many registrations from this address",
                },
                status_code=429,
                headers={"Retry-After": str(int(retry) + 1)},
            )
        now = self.now()
        self.store.prune_dcr_clients(now, lim.dcr_unused_ttl, lim.dcr_idle_ttl)
        if self.store.count_dcr_clients() >= lim.dcr_max_clients:
            self.audit("registration.rejected", reason="client cap reached")
            return JSONResponse(
                {"error": "temporarily_unavailable", "error_description": "registration is full, try later"},
                status_code=503,
                headers={"Retry-After": "3600"},
            )
        try:
            if request.headers.get("content-type", "").split(";")[0].strip().lower() != "application/json":
                raise OAuthError("invalid_client_metadata", "expected application/json")
            try:
                meta = json.loads(await self._body(request))
            except ValueError as exc:
                raise OAuthError("invalid_client_metadata", "body is not JSON") from exc
            if not isinstance(meta, dict):
                raise OAuthError("invalid_client_metadata", "body must be a JSON object")
            uris = meta.get("redirect_uris")
            if not isinstance(uris, list) or not 1 <= len(uris) <= 5:
                raise OAuthError("invalid_redirect_uri", "1 to 5 redirect_uris are required")
            for uri in uris:
                problem = redirect_uri_problem(uri)
                if problem:
                    raise OAuthError("invalid_redirect_uri", problem)
            if meta.get("token_endpoint_auth_method", "none") != "none":
                raise OAuthError(
                    "invalid_client_metadata",
                    "dynamic clients are public: token_endpoint_auth_method must be none",
                )
            if not set(meta.get("response_types", ["code"])) <= {"code"}:
                raise OAuthError("invalid_client_metadata", "only response_type code is supported")
            grant_types = dcr_grant_types(meta.get("grant_types"))
            name = meta.get("client_name") or "Unnamed agent"
            if not isinstance(name, str) or len(name) > 100 or not name.isprintable():
                raise OAuthError(
                    "invalid_client_metadata", "client_name must be printable, at most 100 characters"
                )
        except OAuthError as exc:
            self.audit("registration.rejected", reason=exc.description)
            return self._error(exc)
        client_id = "dcr_" + random_token(18)
        self.store.add_client(client_id, "dcr", name, uris, grant_types, now, ip)
        self.audit("client.registered", client_id=client_id, name=name)
        return JSONResponse(
            {
                "client_id": client_id,
                "client_id_issued_at": int(now),
                "client_name": name,
                "redirect_uris": uris,
                "grant_types": grant_types,
                "response_types": ["code"],
                "token_endpoint_auth_method": "none",
            },
            status_code=201,
            headers=NO_STORE,
        )

    # ---- revocation ------------------------------------------------------------------------

    async def revoke(self, request: Request) -> Response:
        try:
            form = await self._form(request)
            client = await self.clients.authenticate(form, self.token_endpoint)
        except OAuthError as exc:
            return self._error(exc)
        token = form.get("token", "")
        now = self.now()
        row = self.store.get_refresh(sha256_hex(token)) if token else None
        if row is not None:
            if row["client_id"] == client.client_id:
                self.store.revoke_family(row["family"], now)
                self.audit(
                    "token.revoked", user_id=row["user_id"], client_id=client.client_id, kind="refresh"
                )
            return Response(status_code=200, headers=NO_STORE)
        try:
            claims = jwt.decode(
                token,
                self._public_key,
                algorithms=[SIGNING_ALG],
                issuer=self.config.issuer,
                options={"verify_aud": False, "verify_exp": False, "require": ["jti"]},
            )
        except jwt.PyJWTError:
            return Response(status_code=200, headers=NO_STORE)  # RFC 7009: unknown tokens are not an error
        access = self.store.get_access(claims["jti"])
        if access is not None and access["client_id"] == client.client_id:
            self.store.revoke_access(claims["jti"], now)
            user_id = claims["sub"] if claims.get("sub_kind") == "user" else None
            self.audit("token.revoked", user_id=user_id, client_id=client.client_id, kind="access")
        return Response(status_code=200, headers=NO_STORE)

    async def revoked(self, request: Request) -> JSONResponse:
        now = self.now()
        return JSONResponse({"revoked": self.store.revoked_unexpired(now), "generated_at": int(now)})

    # ---- audit events from resource servers ------------------------------------------------

    async def audit_events(self, request: Request) -> Response:
        try:
            form = await self._form(request)
            client = await self.clients.authenticate(form, self.token_endpoint)
            if client.auth_method != "private_key_jwt" or not client.serves_resource:
                raise OAuthError("unauthorized_client", "only resource servers report events", 403)
            try:
                events = json.loads(form.get("events", "[]"))
            except ValueError as exc:
                raise OAuthError("invalid_request", "events must be JSON") from exc
            if not isinstance(events, list) or len(events) > 50:
                raise OAuthError("invalid_request", "events must be a list of at most 50 items")
        except OAuthError as exc:
            return self._error(exc)

        def text(value: Any, limit: int = 200) -> str | None:
            return value[:limit] if isinstance(value, str) and value else None

        for e in events:
            if not isinstance(e, dict):
                continue
            actors = e.get("actors")
            self.audit(
                "resource.call",
                user_id=text(e.get("subject")) if e.get("sub_kind") == "user" else None,
                client_id=text(e.get("client_id")),
                resource=client.serves_resource,
                tool=text(e.get("tool"), 100),
                method=text(e.get("method"), 100),
                status=e["status"] if isinstance(e.get("status"), int) else None,
                actors=[a[:200] for a in actors if isinstance(a, str)][:5]
                if isinstance(actors, list)
                else [],
                reported_by=client.client_id,
            )
        return Response(status_code=204)

    # ---- user session ----------------------------------------------------------------------

    async def login_form(self, request: Request) -> Response:
        if not self.identity.password_login():  # pragma: no cover - only the dev provider exists today
            return HTMLResponse(pages.error_page("Sign in", "Password login is disabled."), status_code=404)
        token = random_token(16)
        response = HTMLResponse(pages.login_page(_safe_next(request.query_params.get("next")), token))
        self._cookie(response, self.login_cookie, token, 600)
        return response

    async def login(self, request: Request) -> Response:
        try:
            form = await self._form(request)
        except OAuthError as exc:
            return HTMLResponse(pages.error_page("Bad request", exc.description), status_code=400)
        next_url = _safe_next(form.get("next"))
        cookie = request.cookies.get(self.login_cookie, "")
        if (
            not cookie
            or not hmac.compare_digest(cookie, form.get("csrf", ""))
            or not self._same_origin(request)
        ):
            return HTMLResponse(
                pages.error_page("Session expired", "Please reload the page."), status_code=403
            )
        if self.login_limiter.take(self.ip(request)):
            return HTMLResponse(
                pages.login_page(next_url, cookie, "Too many attempts. Wait a minute."), status_code=429
            )
        username = form.get("username", "")[:100]
        now = self.now()
        fails, locked_until = self._login_failures.get(username, (0, 0.0))
        if locked_until > now:
            return HTMLResponse(
                pages.login_page(next_url, cookie, "Account locked for a few minutes."), status_code=429
            )
        user_id = await self.identity.authenticate(form)
        if user_id is None:
            fails += 1
            lim = self.config.limits
            lock = now + lim.login_lock_seconds if fails >= lim.login_failures_before_lock else 0.0
            self._login_failures[username] = (0 if lock else fails, lock)
            if len(self._login_failures) > 10_000:
                self._login_failures.clear()
            return HTMLResponse(
                pages.login_page(next_url, cookie, "Wrong username or password."), status_code=401
            )
        self._login_failures.pop(username, None)
        sid = random_token()
        self.store.add_session(
            sha256_hex(sid), user_id, random_token(16), now + self.config.limits.session_ttl
        )
        self.audit("user.login", user_id=user_id, provider=self.identity.name)
        response = RedirectResponse(next_url, status_code=303)
        self._cookie(response, self.cookie, sid, self.config.limits.session_ttl)
        response.delete_cookie(self.login_cookie, path="/", secure=self.config.secure_cookies, httponly=True)
        return response

    async def logout(self, request: Request) -> Response:
        session = self._session(request)
        try:
            form = await self._form(request)
        except OAuthError:
            form = {}
        if session and self._same_origin(request) and hmac.compare_digest(form.get("csrf", ""), session[2]):
            self.store.delete_session(session[0])
        response = RedirectResponse("/login", status_code=303)
        response.delete_cookie(self.cookie, path="/", secure=self.config.secure_cookies, httponly=True)
        return response

    def _wants_json(self, request: Request) -> bool:
        return "application/json" in request.headers.get("accept", "")

    async def consents(self, request: Request) -> Response:
        session = self._session(request)
        if session is None:
            return RedirectResponse("/login?next=/consents", status_code=303)
        _, user_id, csrf = session
        grants = self.store.grants_for_user(user_id)
        if self._wants_json(request):
            return JSONResponse(
                {
                    "user": user_id,
                    "csrf": csrf,
                    "consents": [
                        {
                            "id": g.id,
                            "client_id": g.client_id,
                            "resource": g.resource,
                            "scopes": sorted(g.scopes),
                            "created_at": int(g.created_at),
                        }
                        for g in grants
                    ],
                }
            )
        rows = []
        for g in grants:
            client = await self.clients.get(g.client_id)
            res: ResourceConfig | None = self.config.resource(g.resource)
            rows.append((g, client.name if client else g.client_id, res.name if res else g.resource))
        return HTMLResponse(pages.consents_page(user_id, csrf, rows))

    async def revoke_consent(self, request: Request) -> Response:
        session = self._session(request)
        try:
            form = await self._form(request)
        except OAuthError as exc:
            return HTMLResponse(pages.error_page("Bad request", exc.description), status_code=400)
        if (
            session is None
            or not self._same_origin(request)
            or not hmac.compare_digest(form.get("csrf", ""), session[2])
        ):
            return HTMLResponse(
                pages.error_page("Session expired", "Please reload the page."), status_code=403
            )
        grant = self.store.get_grant(request.path_params["grant_id"])
        if grant is None or grant.user_id != session[1]:
            return HTMLResponse(pages.error_page("Not found", "No such consent."), status_code=404)
        if self.store.revoke_grant(grant.id, self.now()):
            self.audit(
                "consent.revoked", user_id=grant.user_id, client_id=grant.client_id, resource=grant.resource
            )
        return RedirectResponse("/consents", status_code=303)

    async def activity(self, request: Request) -> Response:
        session = self._session(request)
        if session is None:
            return RedirectResponse("/login?next=/activity", status_code=303)
        events = self.store.audit_events(session[1])
        if self._wants_json(request):
            return JSONResponse(
                {
                    "events": [
                        {
                            "seq": e.seq,
                            "ts": e.ts,
                            "event": e.event,
                            "client_id": e.client_id,
                            "resource": e.resource,
                            "detail": e.detail,
                            "hash": e.hash,
                        }
                        for e in events
                    ]
                }
            )
        return HTMLResponse(pages.activity_page(session[1], events))
