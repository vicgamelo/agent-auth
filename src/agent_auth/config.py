"""Server configuration: resources, pre-registered clients and limits.

Everything here is plain data, so it can be built in code (tests, the demo) or read from TOML
(``agent-auth serve --config``).
"""

from __future__ import annotations

import ipaddress
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

AuthMethod = Literal["none", "private_key_jwt"]

AUTHORIZATION_CODE = "authorization_code"
REFRESH_TOKEN = "refresh_token"
CLIENT_CREDENTIALS = "client_credentials"
TOKEN_EXCHANGE = "urn:ietf:params:oauth:grant-type:token-exchange"
GRANT_TYPES = frozenset({AUTHORIZATION_CODE, REFRESH_TOKEN, CLIENT_CREDENTIALS, TOKEN_EXCHANGE})


class ConfigError(ValueError):
    pass


def is_loopback_host(host: str | None) -> bool:
    if not host:
        return False
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


def canonical_resource(uri: str) -> str:
    """RFC 8707 resource identifier: absolute, no fragment, lowercase scheme/host, no trailing slash."""
    parts = urlsplit(uri)
    if parts.scheme.lower() not in ("https", "http") or not parts.netloc or parts.fragment:
        raise ConfigError(f"resource must be an absolute http(s) URI without fragment: {uri!r}")
    if parts.username or parts.password:
        raise ConfigError(f"resource must not carry credentials: {uri!r}")
    if parts.scheme.lower() == "http" and not is_loopback_host(parts.hostname):
        raise ConfigError(f"resource must use https (http only on loopback): {uri!r}")
    path = parts.path.rstrip("/")
    query = f"?{parts.query}" if parts.query else ""
    return f"{parts.scheme.lower()}://{parts.netloc.lower()}{path}{query}"


@dataclass(frozen=True)
class ResourceConfig:
    """A protected resource (an MCP server or an API) that tokens can be issued for."""

    uri: str
    name: str
    scopes: Mapping[str, str]  # scope -> description shown on the consent screen
    default_scopes: tuple[str, ...] = ()
    # Resources the server behind this one may reach on the user's behalf through token exchange.
    delegates_to: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "uri", canonical_resource(self.uri))
        object.__setattr__(self, "delegates_to", tuple(canonical_resource(d) for d in self.delegates_to))
        unknown = set(self.default_scopes) - set(self.scopes)
        if unknown:
            raise ConfigError(f"{self.uri}: default scopes not declared: {sorted(unknown)}")


@dataclass(frozen=True)
class ClientConfig:
    """A pre-registered client: an MCP server that exchanges tokens, an M2M job or a known agent."""

    client_id: str
    name: str
    auth_method: AuthMethod = "private_key_jwt"
    jwks: tuple[Mapping[str, Any], ...] = ()
    grant_types: frozenset[str] = frozenset({AUTHORIZATION_CODE, REFRESH_TOKEN})
    redirect_uris: tuple[str, ...] = ()
    # client_credentials: what the client may get for itself.
    allowed_resources: frozenset[str] = frozenset()
    allowed_scopes: frozenset[str] = frozenset()
    # Token exchange and audit reporting: the resource this client *is*.
    serves_resource: str | None = None

    def __post_init__(self) -> None:
        if not self.client_id or self.client_id.startswith(("dcr_", "https://", "http://")):
            raise ConfigError(f"static client_id {self.client_id!r} collides with dynamic client ids")
        unknown = set(self.grant_types) - GRANT_TYPES
        if unknown:
            raise ConfigError(f"{self.client_id}: unknown grant types {sorted(unknown)}")
        if self.auth_method == "private_key_jwt" and not self.jwks:
            raise ConfigError(f"{self.client_id}: private_key_jwt needs at least one public JWK")
        confidential_only = {CLIENT_CREDENTIALS, TOKEN_EXCHANGE}
        if self.auth_method == "none" and self.grant_types & confidential_only:
            raise ConfigError(
                f"{self.client_id}: public clients cannot use client_credentials/token exchange"
            )
        if TOKEN_EXCHANGE in self.grant_types and not self.serves_resource:
            raise ConfigError(f"{self.client_id}: token exchange needs serves_resource")
        object.__setattr__(
            self, "allowed_resources", frozenset(canonical_resource(r) for r in self.allowed_resources)
        )
        if self.serves_resource:
            object.__setattr__(self, "serves_resource", canonical_resource(self.serves_resource))


@dataclass(frozen=True)
class Limits:
    access_token_ttl: int = 300
    refresh_token_ttl: int = 30 * 24 * 3600
    code_ttl: int = 60
    pending_ttl: int = 600
    session_ttl: int = 8 * 3600
    max_actor_depth: int = 3
    # Dynamic Client Registration (deprecated in MCP 2026-07-28, kept for older clients).
    dcr_enabled: bool = True
    dcr_per_ip_per_hour: float = 10
    dcr_burst: int = 3
    dcr_max_clients: int = 500
    dcr_unused_ttl: int = 24 * 3600
    dcr_idle_ttl: int = 30 * 24 * 3600
    # Client ID Metadata Documents.
    cimd_enabled: bool = True
    cimd_cache_ttl: int = 300
    # Login and token endpoint throttling.
    login_per_ip_per_minute: float = 10
    login_failures_before_lock: int = 5
    login_lock_seconds: int = 300
    token_per_ip_per_minute: float = 120
    max_body_bytes: int = 16 * 1024


@dataclass(frozen=True)
class ServerConfig:
    issuer: str
    resources: tuple[ResourceConfig, ...]
    clients: tuple[ClientConfig, ...] = ()
    limits: Limits = field(default_factory=Limits)
    trusted_proxies: tuple[str, ...] = ()  # CIDRs whose X-Forwarded-For is believed
    database: str = "agent-auth.db"
    signing_key_path: str | None = "agent-auth-key.pem"
    # Dev only: lets CIMD documents and redirect URIs live on http://localhost.
    allow_loopback_cimd: bool = False

    def __post_init__(self) -> None:
        parts = urlsplit(self.issuer)
        if parts.path not in ("", "/") or parts.query or parts.fragment:
            raise ConfigError("issuer must be an origin (scheme://host[:port]) without path")
        if parts.scheme != "https" and not (parts.scheme == "http" and is_loopback_host(parts.hostname)):
            raise ConfigError("issuer must use https (http is accepted only on loopback, for development)")
        object.__setattr__(self, "issuer", f"{parts.scheme}://{parts.netloc}")
        uris = [r.uri for r in self.resources]
        if len(set(uris)) != len(uris):
            raise ConfigError("duplicate resource URIs")
        known = set(uris)
        for r in self.resources:
            missing = set(r.delegates_to) - known
            if missing:
                raise ConfigError(f"{r.uri}: delegates_to unknown resources {sorted(missing)}")
        ids = [c.client_id for c in self.clients]
        if len(set(ids)) != len(ids):
            raise ConfigError("duplicate client ids")
        for c in self.clients:
            for res in c.allowed_resources | ({c.serves_resource} if c.serves_resource else set()):
                if res not in known:
                    raise ConfigError(f"{c.client_id}: unknown resource {res}")

    @property
    def secure_cookies(self) -> bool:
        return self.issuer.startswith("https://")

    def resource(self, uri: str) -> ResourceConfig | None:
        try:
            canonical = canonical_resource(uri)
        except ConfigError:
            return None
        return next((r for r in self.resources if r.uri == canonical), None)

    def all_scopes(self) -> list[str]:
        return sorted({s for r in self.resources for s in r.scopes})


def load_config(path: Path) -> ServerConfig:
    raw = tomllib.loads(path.read_text())
    limits = Limits(**raw.get("limits", {}))
    resources = tuple(
        ResourceConfig(
            uri=r["uri"],
            name=r["name"],
            scopes=r["scopes"],
            default_scopes=tuple(r.get("default_scopes", ())),
            delegates_to=tuple(r.get("delegates_to", ())),
        )
        for r in raw.get("resources", [])
    )
    clients = tuple(
        ClientConfig(
            client_id=c["client_id"],
            name=c.get("name", c["client_id"]),
            auth_method=c.get("auth_method", "private_key_jwt"),
            jwks=tuple(c.get("jwks", ())),
            grant_types=frozenset(c.get("grant_types", (AUTHORIZATION_CODE, REFRESH_TOKEN))),
            redirect_uris=tuple(c.get("redirect_uris", ())),
            allowed_resources=frozenset(c.get("allowed_resources", ())),
            allowed_scopes=frozenset(c.get("allowed_scopes", ())),
            serves_resource=c.get("serves_resource"),
        )
        for c in raw.get("clients", [])
    )
    return ServerConfig(
        issuer=raw["issuer"],
        resources=resources,
        clients=clients,
        limits=limits,
        trusted_proxies=tuple(raw.get("trusted_proxies", ())),
        database=raw.get("database", "agent-auth.db"),
        signing_key_path=raw.get("signing_key_path", "agent-auth-key.pem"),
        allow_loopback_cimd=raw.get("allow_loopback_cimd", False),
    )
