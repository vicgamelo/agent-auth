"""Client lookup (static, DCR, CIMD) and token endpoint client authentication."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

import jwt

from .cimd import CIMDError, CIMDFetcher, looks_like_cimd
from .config import AUTHORIZATION_CODE, REFRESH_TOKEN, ClientConfig, ServerConfig
from .crypto import JWT_BEARER
from .store import Store

ASSERTION_ALGS = ["ES256", "RS256", "EdDSA"]
MAX_ASSERTION_LIFETIME = 300


class OAuthError(Exception):
    """An RFC 6749 style error response."""

    def __init__(self, error: str, description: str, status: int = 400) -> None:
        super().__init__(description)
        self.error = error
        self.description = description
        self.status = status


@dataclass(frozen=True)
class Client:
    client_id: str
    kind: Literal["static", "dcr", "cimd"]
    name: str
    redirect_uris: tuple[str, ...]
    grant_types: frozenset[str]
    auth_method: Literal["none", "private_key_jwt"]
    jwks: tuple[Mapping[str, Any], ...] = ()
    allowed_resources: frozenset[str] = frozenset()
    allowed_scopes: frozenset[str] = frozenset()
    serves_resource: str | None = None

    @property
    def verified(self) -> bool:
        """Static clients were vetted by the operator; CIMD ones at least proved a domain."""
        return self.kind != "dcr"


def from_config(c: ClientConfig) -> Client:
    return Client(
        client_id=c.client_id,
        kind="static",
        name=c.name,
        redirect_uris=c.redirect_uris,
        grant_types=c.grant_types,
        auth_method=c.auth_method,
        jwks=c.jwks,
        allowed_resources=c.allowed_resources,
        allowed_scopes=c.allowed_scopes,
        serves_resource=c.serves_resource,
    )


class ClientRegistry:
    def __init__(
        self, config: ServerConfig, store: Store, fetcher: CIMDFetcher, clock: Callable[[], float]
    ) -> None:
        self._static = {c.client_id: from_config(c) for c in config.clients}
        self._config = config
        self._store = store
        self._fetcher = fetcher
        self._clock = clock

    async def get(self, client_id: str) -> Client | None:
        if not client_id or len(client_id) > 512:
            return None
        if client_id in self._static:
            return self._static[client_id]
        if looks_like_cimd(client_id):
            if not self._config.limits.cimd_enabled:
                return None
            try:
                meta = await self._fetcher.fetch(client_id)
            except CIMDError:
                return None
            return Client(
                client_id=client_id,
                kind="cimd",
                name=meta.client_name,
                redirect_uris=meta.redirect_uris,
                grant_types=frozenset(meta.grant_types) | {AUTHORIZATION_CODE},
                auth_method="none",
            )
        row = self._store.get_client(client_id)
        if row is None:
            return None
        return Client(
            client_id=row["client_id"],
            kind="dcr",
            name=row["name"],
            redirect_uris=tuple(json.loads(row["redirect_uris"])),
            grant_types=frozenset(json.loads(row["grant_types"])),
            auth_method="none",
        )

    async def cimd_problem(self, client_id: str) -> str | None:
        """Explains why a URL client_id was refused (for the error page)."""
        try:
            await self._fetcher.fetch(client_id)
        except CIMDError as exc:
            return str(exc)
        return None

    async def authenticate(self, form: Mapping[str, str], token_endpoint: str) -> Client:
        """Authenticates the caller of the token/revocation endpoints.

        Confidential clients must use private_key_jwt; shared secrets are not supported at all.
        Public clients identify themselves with client_id only, and PKCE/refresh rotation carry the
        security.
        """
        assertion_type = form.get("client_assertion_type")
        assertion = form.get("client_assertion")
        if form.get("client_secret"):
            raise OAuthError("invalid_client", "client secrets are not supported; use private_key_jwt", 401)
        if assertion_type or assertion:
            if assertion_type != JWT_BEARER or not assertion:
                raise OAuthError("invalid_client", "unsupported client assertion", 401)
            return await self._authenticate_assertion(assertion, form.get("client_id"), token_endpoint)
        client = await self.get(form.get("client_id", ""))
        if client is None:
            raise OAuthError("invalid_client", "unknown client", 401)
        if client.auth_method != "none":
            raise OAuthError("invalid_client", "this client must authenticate with private_key_jwt", 401)
        return client

    async def _authenticate_assertion(
        self, assertion: str, claimed: str | None, token_endpoint: str
    ) -> Client:
        try:
            unverified = jwt.decode(assertion, options={"verify_signature": False})
        except jwt.PyJWTError as exc:
            raise OAuthError("invalid_client", "malformed client assertion", 401) from exc
        client_id = unverified.get("sub")
        if not isinstance(client_id, str) or (claimed and claimed != client_id):
            raise OAuthError("invalid_client", "client assertion subject does not match client_id", 401)
        client = await self.get(client_id)
        if client is None or client.auth_method != "private_key_jwt" or not client.jwks:
            raise OAuthError("invalid_client", "unknown confidential client", 401)
        header = jwt.get_unverified_header(assertion)
        if header.get("alg") not in ASSERTION_ALGS:
            raise OAuthError("invalid_client", "client assertion algorithm not allowed", 401)
        claims: dict[str, Any] | None = None
        for jwk in client.jwks:
            if header.get("kid") and jwk.get("kid") and header["kid"] != jwk["kid"]:
                continue
            try:
                key = jwt.PyJWK(dict(jwk))
                claims = jwt.decode(
                    assertion,
                    key,
                    algorithms=ASSERTION_ALGS,
                    audience=[self._config.issuer, token_endpoint],
                    issuer=client_id,
                    leeway=5,
                    options={"require": ["exp", "iat", "jti", "sub", "iss", "aud"]},
                )
                break
            except jwt.PyJWTError:
                continue
        if claims is None:
            raise OAuthError("invalid_client", "client assertion signature or claims invalid", 401)
        now = self._clock()
        if int(claims["exp"]) - int(claims["iat"]) > MAX_ASSERTION_LIFETIME or int(claims["iat"]) > now + 5:
            raise OAuthError("invalid_client", "client assertion lifetime too long", 401)
        if not self._store.use_assertion_jti(f"{client_id}:{claims['jti']}", float(claims["exp"]) + 5):
            raise OAuthError("invalid_client", "client assertion was already used", 401)
        return client


def dcr_grant_types(requested: Any) -> list[str]:
    if requested is None:
        return [AUTHORIZATION_CODE, REFRESH_TOKEN]
    if not isinstance(requested, list) or not set(requested) <= {AUTHORIZATION_CODE, REFRESH_TOKEN}:
        raise OAuthError(
            "invalid_client_metadata", "grant_types may only be authorization_code and refresh_token"
        )
    return list(dict.fromkeys([AUTHORIZATION_CODE, *requested]))
