from __future__ import annotations

import re
import secrets
import time
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import parse_qs, urlencode, urlsplit

import httpx
import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.types import ASGIApp

from agent_auth.config import (
    AUTHORIZATION_CODE,
    CLIENT_CREDENTIALS,
    REFRESH_TOKEN,
    TOKEN_EXCHANGE,
    ClientConfig,
    Limits,
    ResourceConfig,
    ServerConfig,
)
from agent_auth.crypto import JWT_BEARER, SigningKey, client_assertion, pkce_s256
from agent_auth.guard import ProtectedResource, ServiceClient, TokenVerifier, get_principal
from agent_auth.identity import DevIdentityProvider
from agent_auth.server import AuthorizationServer
from agent_auth.store import Store

ISSUER = "https://as.test"
MCP = "https://mcp.test/mcp"
API = "https://api.test"
OTHER = "https://other.test/mcp"
BRASIL = "https://brasil.test/mcp"
CLI_REDIRECT = "http://127.0.0.1:8765/callback"


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class Clock:
    def __init__(self) -> None:
        self.now = time.time()

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class Router(httpx.AsyncBaseTransport):
    """Sends each request to the ASGI app registered for its host."""

    def __init__(self, apps: Mapping[str, ASGIApp]) -> None:
        self.transports = {host: httpx.ASGITransport(app=app) for host, app in apps.items()}

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        transport = self.transports.get(request.url.host)
        if transport is None:
            return httpx.Response(502, text=f"no route to {request.url.host}")
        return await transport.handle_async_request(request)


@dataclass
class Env:
    clock: Clock
    server: AuthorizationServer
    keys: dict[str, SigningKey]
    router: Router
    apps: dict[str, ASGIApp] = field(default_factory=dict)

    @property
    def store(self) -> Store:
        return self.server.store

    def http(self, **kw: Any) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=self.router, follow_redirects=False, **kw)

    def browser(self) -> httpx.AsyncClient:
        return self.http(base_url=ISSUER)

    def service(self, client_id: str) -> ServiceClient:
        return ServiceClient(ISSUER, client_id, self.keys[client_id], http=self.http(), clock=self.clock)

    def assertion(self, client_id: str, audience: str = ISSUER) -> dict[str, str]:
        return {
            "client_id": client_id,
            "client_assertion_type": JWT_BEARER,
            "client_assertion": client_assertion(client_id, self.keys[client_id], audience),
        }


def resources() -> tuple[ResourceConfig, ...]:
    return (
        ResourceConfig(
            MCP,
            "Notes MCP",
            {"notes:read": "read your notes", "notes:write": "add notes"},
            default_scopes=("notes:read",),
            delegates_to=(API,),
        ),
        ResourceConfig(API, "Notes API", {"notes:read": "read notes", "notes:write": "write notes"}),
        ResourceConfig(OTHER, "Other MCP", {"notes:read": "read notes"}, default_scopes=("notes:read",)),
        ResourceConfig(BRASIL, "Brasil MCP", {"brasil:read": "public data"}, default_scopes=("brasil:read",)),
    )


def make_env(limits: Limits | None = None, **server_kw: Any) -> Env:
    keys = {name: SigningKey.generate() for name in ("notes-mcp", "other-mcp", "report-job")}
    clients = (
        ClientConfig(
            "notes-mcp",
            "Notes MCP",
            jwks=(keys["notes-mcp"].public_jwk,),
            grant_types=frozenset({TOKEN_EXCHANGE}),
            serves_resource=MCP,
        ),
        ClientConfig(
            "other-mcp",
            "Other MCP",
            jwks=(keys["other-mcp"].public_jwk,),
            grant_types=frozenset({TOKEN_EXCHANGE}),
            serves_resource=OTHER,
        ),
        ClientConfig(
            "report-job",
            "Nightly report",
            jwks=(keys["report-job"].public_jwk,),
            grant_types=frozenset({CLIENT_CREDENTIALS}),
            allowed_resources=frozenset({BRASIL, API}),
            allowed_scopes=frozenset({"brasil:read", "notes:read"}),
        ),
        ClientConfig(
            "cli-agent",
            "CLI agent",
            auth_method="none",
            redirect_uris=(CLI_REDIRECT,),
            grant_types=frozenset({AUTHORIZATION_CODE, REFRESH_TOKEN}),
        ),
    )
    config = ServerConfig(
        issuer=ISSUER,
        resources=resources(),
        clients=clients,
        limits=limits or Limits(),
        database=":memory:",
        signing_key_path=None,
    )
    clock = Clock()
    server = AuthorizationServer(
        config, DevIdentityProvider({"alice": "pw-alice", "bob": "pw-bob"}), clock=clock, **server_kw
    )
    router = Router({"as.test": server})
    return Env(clock=clock, server=server, keys=keys, router=router)


@pytest.fixture
def env() -> Env:
    return make_env()


# ---- browser helpers -------------------------------------------------------------------------

CSRF = re.compile(r'name=csrf value="([^"]+)"')
REQUEST = re.compile(r'name=request value="([^"]+)"')


async def login(browser: httpx.AsyncClient, user: str = "alice", password: str | None = None) -> None:
    page = await browser.get("/login")
    csrf = CSRF.search(page.text)
    assert csrf, page.text
    r = await browser.post(
        "/login",
        data={
            "csrf": csrf.group(1),
            "username": user,
            "password": password or f"pw-{user}",
            "next": "/consents",
        },
    )
    assert r.status_code == 303, r.text


@dataclass
class Pkce:
    verifier: str = field(default_factory=lambda: secrets.token_urlsafe(48))

    @property
    def challenge(self) -> str:
        return pkce_s256(self.verifier)


def authorize_params(
    client_id: str,
    redirect_uri: str = CLI_REDIRECT,
    *,
    pkce: Pkce | None = None,
    resource: str | None = MCP,
    scope: str | None = "notes:read",
    **extra: str,
) -> dict[str, str]:
    pkce = pkce or Pkce()
    params = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "state": "st-123",
        "code_challenge": pkce.challenge,
        "code_challenge_method": "S256",
    }
    if resource is not None:
        params["resource"] = resource
    if scope is not None:
        params["scope"] = scope
    params.update(extra)
    return {k: v for k, v in params.items() if v != "__omit__"}


async def approve(
    browser: httpx.AsyncClient, response: httpx.Response, decision: str = "approve"
) -> httpx.Response:
    assert response.status_code == 200, (response.status_code, response.text[:300])
    csrf = CSRF.search(response.text)
    req = REQUEST.search(response.text)
    assert csrf and req
    return await browser.post(
        "/authorize/decision", data={"csrf": csrf.group(1), "request": req.group(1), "decision": decision}
    )


def query(location: str) -> dict[str, str]:
    return {k: v[0] for k, v in parse_qs(urlsplit(location).query).items()}


async def get_code(browser: httpx.AsyncClient, params: dict[str, str]) -> dict[str, str]:
    r = await browser.get("/authorize?" + urlencode(params))
    if r.status_code == 200:
        r = await approve(browser, r)
    assert r.status_code in (302, 303), (r.status_code, r.text[:300])
    return query(r.headers["location"])


async def token(http: httpx.AsyncClient, form: dict[str, str]) -> httpx.Response:
    return await http.post(f"{ISSUER}/token", data=form)


async def user_tokens(
    env: Env,
    client_id: str = "cli-agent",
    scope: str = "notes:read",
    user: str = "alice",
    resource: str = MCP,
    redirect_uri: str = CLI_REDIRECT,
) -> dict[str, Any]:
    async with env.browser() as browser:
        await login(browser, user)
        pkce = Pkce()
        result = await get_code(
            browser, authorize_params(client_id, redirect_uri, pkce=pkce, resource=resource, scope=scope)
        )
    async with env.http() as http:
        r = await token(
            http,
            {
                "grant_type": "authorization_code",
                "client_id": client_id,
                "code": result["code"],
                "code_verifier": pkce.verifier,
                "redirect_uri": redirect_uri,
                "resource": resource,
            },
        )
    assert r.status_code == 200, r.text
    data: dict[str, Any] = r.json()
    return data


async def register(
    http: httpx.AsyncClient, meta: dict[str, Any] | None = None, **headers: str
) -> httpx.Response:
    body = meta if meta is not None else {"client_name": "Test agent", "redirect_uris": [CLI_REDIRECT]}
    return await http.post(f"{ISSUER}/register", json=body, headers=headers)


# ---- protected apps --------------------------------------------------------------------------


def echo_app() -> Starlette:
    async def whoami(request: Request) -> JSONResponse:
        p = get_principal(request)
        assert p is not None
        return JSONResponse(
            {
                "sub": p.subject,
                "kind": p.kind,
                "client_id": p.client_id,
                "actors": list(p.actors),
                "scopes": sorted(p.scopes),
            }
        )

    return Starlette(routes=[Route("/{path:path}", whoami, methods=["GET", "POST"])])


def protect(
    env: Env, resource: str, scopes: list[str], *, events: list[dict[str, Any]] | None = None, **kw: Any
) -> ProtectedResource:
    sink_events = events if events is not None else []

    async def sink(event: dict[str, Any]) -> None:
        sink_events.append(event)

    verifier = TokenVerifier(ISSUER, resource, http=env.http(), clock=env.clock, revocation_interval=0)
    guard = ProtectedResource(
        kw.pop("app", None) or echo_app(),
        resource=resource,
        issuer=ISSUER,
        scopes_supported=scopes,
        verifier=verifier,
        audit=sink,
        **kw,
    )
    env.router.transports[urlsplit(resource).hostname or ""] = httpx.ASGITransport(app=guard)
    return guard


@pytest.fixture
async def http(env: Env) -> AsyncIterator[httpx.AsyncClient]:
    async with env.http() as client:
        yield client
