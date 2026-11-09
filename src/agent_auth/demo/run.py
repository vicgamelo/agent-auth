"""agent-auth-demo: runs everything locally and walks through the flow.

    uv run agent-auth-demo                 # scripted run, prints each step, exits
    uv run agent-auth-demo --keep-running  # then leaves the servers up to explore in a browser

Services (all on 127.0.0.1):
    authorization server   :9400   sign in as alice / alice-demo-password
    Notes MCP              :9401/mcp
    Notes API              :9402
    Brasil MCP             :9403/mcp   (only if mcp-server-brasil is installed)

The agent is the official MCP Python SDK client with its OAuth provider: it discovers the
authorization server from the 401, registers itself, sends the user through consent with PKCE,
steps up its scope when a tool needs more, and loses access when the user revokes it.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import socket
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import parse_qs, urljoin, urlsplit

import httpx
import uvicorn
from starlette.types import ASGIApp

from ..config import (
    CLIENT_CREDENTIALS,
    TOKEN_EXCHANGE,
    ClientConfig,
    Limits,
    ResourceConfig,
    ServerConfig,
)
from ..crypto import SigningKey
from ..guard import ServiceClient, reporting_audit_sink
from ..identity import DevIdentityProvider
from ..server import AuthorizationServer
from .apps import build_brasil_mcp, build_notes_api, build_notes_mcp

USER = "alice"
PASSWORD = "alice-demo-password"
SERVICE_IDS = ("notes-mcp", "notes-api", "brasil-mcp", "report-job")


@dataclass
class Urls:
    issuer: str
    mcp: str
    api: str
    brasil: str

    @classmethod
    def local(cls, base: int) -> Urls:
        host = "http://127.0.0.1"
        return cls(f"{host}:{base}", f"{host}:{base + 1}/mcp", f"{host}:{base + 2}", f"{host}:{base + 3}/mcp")


def demo_config(urls: Urls, keys: dict[str, SigningKey], database: str = ":memory:") -> ServerConfig:
    notes = {"notes:read": "read your notes", "notes:write": "add notes"}
    return ServerConfig(
        issuer=urls.issuer,
        resources=(
            ResourceConfig(
                urls.mcp, "Notes MCP", notes, default_scopes=("notes:read",), delegates_to=(urls.api,)
            ),
            ResourceConfig(urls.api, "Notes API", notes),
            ResourceConfig(
                urls.brasil,
                "Brasil MCP",
                {"brasil:read": "Brazilian public data"},
                default_scopes=("brasil:read",),
            ),
        ),
        clients=(
            ClientConfig(
                "notes-mcp",
                "Notes MCP",
                jwks=(keys["notes-mcp"].public_jwk,),
                grant_types=frozenset({TOKEN_EXCHANGE}),
                serves_resource=urls.mcp,
            ),
            ClientConfig(
                "notes-api",
                "Notes API",
                jwks=(keys["notes-api"].public_jwk,),
                grant_types=frozenset(),
                serves_resource=urls.api,
            ),
            ClientConfig(
                "brasil-mcp",
                "Brasil MCP",
                jwks=(keys["brasil-mcp"].public_jwk,),
                grant_types=frozenset(),
                serves_resource=urls.brasil,
            ),
            ClientConfig(
                "report-job",
                "Nightly report job",
                jwks=(keys["report-job"].public_jwk,),
                grant_types=frozenset({CLIENT_CREDENTIALS}),
                allowed_resources=frozenset({urls.brasil, urls.api}),
                allowed_scopes=frozenset({"brasil:read", "notes:read"}),
            ),
        ),
        limits=Limits(),
        database=database,
        signing_key_path=None,
    )


@dataclass
class Stack:
    urls: Urls
    keys: dict[str, SigningKey]
    server: AuthorizationServer
    has_brasil: bool
    revocation_interval: float = 1.0
    servers: list[uvicorn.Server] = field(default_factory=list)
    tasks: list[asyncio.Task[None]] = field(default_factory=list)

    async def stop(self) -> None:
        for s in self.servers:
            s.should_exit = True
        await asyncio.gather(*self.tasks, return_exceptions=True)


def free_port_base() -> int:
    """Four consecutive free ports (for tests)."""
    for _ in range(50):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            base = int(s.getsockname()[1])
        if base + 3 > 65535:
            continue
        try:
            for offset in range(4):
                with socket.socket() as probe:
                    probe.bind(("127.0.0.1", base + offset))
        except OSError:
            continue
        return base
    raise RuntimeError("no free ports")


async def _serve(app: ASGIApp, port: int, stack: Stack) -> None:
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", lifespan="auto")
    server = uvicorn.Server(config)
    stack.servers.append(server)
    stack.tasks.append(asyncio.create_task(server.serve()))
    for _ in range(200):
        if server.started:
            return
        await asyncio.sleep(0.02)
    raise RuntimeError(f"port {port} did not start")


async def start_stack(port_base: int = 9400, revocation_interval: float = 1.0) -> Stack:
    urls = Urls.local(port_base)
    keys = {name: SigningKey.generate() for name in SERVICE_IDS}
    server = AuthorizationServer(
        demo_config(urls, keys), DevIdentityProvider({USER: PASSWORD, "bob": "bob-demo-password"})
    )
    http = httpx.AsyncClient(timeout=10.0)

    def service(name: str) -> ServiceClient:
        return ServiceClient(urls.issuer, name, keys[name], http=http)

    def tuned(guard: Any) -> Any:
        guard.verifier.revocation_interval = revocation_interval
        return guard

    notes_api = tuned(
        build_notes_api(urls.issuer, urls.api, http=http, audit=reporting_audit_sink(service("notes-api")))
    )
    notes_mcp = tuned(
        build_notes_mcp(
            urls.issuer,
            urls.mcp,
            urls.api,
            service("notes-mcp"),
            http=http,
            audit=reporting_audit_sink(service("notes-mcp")),
        )
    )
    brasil = build_brasil_mcp(
        urls.issuer, urls.brasil, http=http, audit=reporting_audit_sink(service("brasil-mcp"))
    )
    stack = Stack(
        urls=urls,
        keys=keys,
        server=server,
        has_brasil=brasil is not None,
        revocation_interval=revocation_interval,
    )
    await _serve(server, port_base, stack)
    await _serve(notes_mcp, port_base + 1, stack)
    await _serve(notes_api, port_base + 2, stack)
    if brasil is not None:
        await _serve(tuned(brasil), port_base + 3, stack)
    return stack


class SimulatedUser:
    """Stands in for Alice at her browser: signs in and clicks Allow (or Deny) on the consent page."""

    def __init__(
        self, issuer: str, username: str = USER, password: str = PASSWORD, say: Callable[[str], None] = print
    ) -> None:
        self.issuer = issuer
        self.username = username
        self.password = password
        self.browser = httpx.AsyncClient(base_url=issuer, follow_redirects=False, timeout=10.0)
        self.decision = "approve"
        self.consent_screens: list[str] = []
        self._callback: str | None = None
        self.say = say

    async def open(self, url: str) -> None:
        response = await self.browser.get(url)
        for _ in range(10):
            location = response.headers.get("location")
            if response.status_code in (302, 303) and location:
                target = urljoin(str(response.url), location)
                if not target.startswith(self.issuer):
                    self._callback = target  # back to the agent's redirect_uri
                    return
                if urlsplit(target).path == "/login":
                    response = await self._login(target)
                else:
                    response = await self.browser.get(target)
                continue
            if response.status_code == 200 and "/authorize/decision" in response.text:
                self.consent_screens.append(response.text)
                scopes = sorted(
                    set(_between_all(response.text, "<code>", "</code>")) & {"notes:read", "notes:write"}
                )
                self.say(f"   [browser] consent screen for {', '.join(scopes)} -> {self.decision}")
                response = await self.browser.post(
                    "/authorize/decision",
                    data={
                        "csrf": _field(response.text, "csrf"),
                        "request": _field(response.text, "request"),
                        "decision": self.decision,
                    },
                )
                continue
            raise RuntimeError(f"unexpected page {response.status_code}: {response.text[:200]}")
        raise RuntimeError("too many redirects")

    async def _login(self, url: str) -> httpx.Response:
        page = await self.browser.get(url)
        self.say(f"   [browser] signs in as {self.username}")
        return await self.browser.post(
            "/login",
            data={
                "csrf": _field(page.text, "csrf"),
                "next": _field(page.text, "next"),
                "username": self.username,
                "password": self.password,
            },
        )

    async def callback(self) -> Any:
        from mcp.shared.auth import AuthorizationCodeResult

        assert self._callback is not None
        q = {k: v[0] for k, v in parse_qs(urlsplit(self._callback).query).items()}
        self._callback = None
        if "error" in q:
            raise PermissionError(f"authorization refused: {q['error']}")
        return AuthorizationCodeResult(code=q["code"], state=q.get("state"), iss=q.get("iss"))

    async def revoke_everything(self) -> int:
        data = (await self.browser.get("/consents", headers={"Accept": "application/json"})).json()
        for consent in data["consents"]:
            await self.browser.post(f"/consents/{consent['id']}/revoke", data={"csrf": data["csrf"]})
        return len(data["consents"])

    async def activity(self) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = (
            await self.browser.get("/activity", headers={"Accept": "application/json"})
        ).json()["events"]
        return events


def _root_cause(exc: BaseException) -> str:
    while isinstance(exc, BaseExceptionGroup) and exc.exceptions:
        exc = exc.exceptions[0]
    return f"{type(exc).__name__}: {exc}"


def _field(html: str, name: str) -> str:
    marker = f'name={name} value="'
    return html.split(marker, 1)[1].split('"', 1)[0].replace("&amp;", "&")


def _between_all(text: str, start: str, end: str) -> list[str]:
    out = []
    for chunk in text.split(start)[1:]:
        out.append(chunk.split(end, 1)[0])
    return out


class MemoryStorage:
    def __init__(self) -> None:
        self.tokens: Any = None
        self.client_info: Any = None

    async def get_tokens(self) -> Any:
        return self.tokens

    async def set_tokens(self, tokens: Any) -> None:
        self.tokens = tokens

    async def get_client_info(self) -> Any:
        return self.client_info

    async def set_client_info(self, client_info: Any) -> None:
        self.client_info = client_info


@dataclass
class Result:
    notes: list[dict[str, Any]] = field(default_factory=list)
    stepped_up: bool = False
    agent_client_id: str = ""
    brasil: str | None = None
    machine_on_user_api: int = 0
    after_revocation: str = ""
    old_token_status: int = 0
    audit_ok: bool = False
    events: list[str] = field(default_factory=list)


def _tool_json(result: Any) -> Any:
    if getattr(result, "structured_content", None):
        return result.structured_content
    import json

    return json.loads(result.content[0].text)


async def run_scenario(stack: Stack, say: Callable[[str], None] = print) -> Result:
    import httpx2
    from mcp.client import Client
    from mcp.client.auth import OAuthClientProvider
    from mcp.client.streamable_http import streamable_http_client
    from mcp.shared.auth import OAuthClientMetadata

    urls = stack.urls
    out = Result()
    user = SimulatedUser(urls.issuer, say=say)
    storage = MemoryStorage()
    provider = OAuthClientProvider(
        server_url=urls.mcp,
        client_metadata=OAuthClientMetadata(
            client_name="Demo assistant",
            redirect_uris=["http://127.0.0.1:8765/callback"],  # type: ignore[list-item]
            grant_types=["authorization_code", "refresh_token"],
            response_types=["code"],
        ),
        storage=storage,
        redirect_handler=user.open,
        callback_handler=user.callback,
    )

    say("1. The agent connects to the Notes MCP with no token")
    phase = "setup"
    try:
        async with (
            httpx2.AsyncClient(auth=provider, timeout=30.0) as http,
            Client(streamable_http_client(urls.mcp, http_client=http)) as agent,
        ):
            out.agent_client_id = storage.client_info.client_id
            say(f"   401 -> {urls.issuer}, registered {out.agent_client_id}, PKCE + consent: notes:read")
            tools = await agent.list_tools()
            say(f"   tools: {', '.join(t.name for t in tools.tools)}")

            say(
                "2. add_note needs notes:write: the MCP answers 403 insufficient_scope and the agent steps up"
            )
            before = len(user.consent_screens)
            added = _tool_json(
                await agent.call_tool("add_note", {"text": "Call Ana about the CNPJ migration"})
            )
            out.stepped_up = len(user.consent_screens) > before
            say(
                f"   note written by {' <- '.join(added['written_by'])} (exchanged token, API sees the chain)"
            )
            listed = _tool_json(await agent.call_tool("list_notes", {}))
            out.notes = listed["notes"]
            say(f"   list_notes -> {len(out.notes)} note(s)")

            say("3. A machine client (client_credentials + private_key_jwt) acts as itself")
            job = ServiceClient(urls.issuer, "report-job", stack.keys["report-job"])
            if stack.has_brasil:
                brasil_token = await job.client_credentials(urls.brasil)
                async with (
                    httpx2.AsyncClient(
                        headers={"Authorization": f"Bearer {brasil_token}"}, timeout=30.0
                    ) as bh,
                    Client(streamable_http_client(urls.brasil, http_client=bh)) as brasil,
                ):
                    checked = await brasil.call_tool("validar_documento", {"documento": "11.222.333/0001-81"})
                    first = checked.content[0] if checked.content else None
                    out.brasil = " ".join(getattr(first, "text", "").split())
                say(f"   Brasil MCP validar_documento -> {out.brasil[:80]}")
            else:
                say("   (mcp-server-brasil not installed: skipping the Brasil MCP call)")
            machine = await job.client_credentials(urls.api, "notes:read")
            async with httpx.AsyncClient() as plain:
                r = await plain.get(f"{urls.api}/notes", headers={"Authorization": f"Bearer {machine}"})
            out.machine_on_user_api = r.status_code
            say(f"   the same job on the user-only Notes API -> {r.status_code} {r.json().get('error')}")

            phase = "revoked"
            say("4. Alice revokes the agent on the consents page")
            revoked = await user.revoke_everything()
            user.decision = "deny"
            say(f"   {revoked} consent(s) revoked; waiting for the revocation list to refresh")
            await asyncio.sleep(stack.revocation_interval + 0.3)
            old_token = storage.tokens.access_token
            async with httpx.AsyncClient() as plain:
                r = await plain.post(
                    urls.mcp,
                    headers={"Authorization": f"Bearer {old_token}"},
                    json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
                )
            out.old_token_status = r.status_code
            say(
                f"   the agent's last token at the MCP -> {r.status_code} {r.json().get('error_description')}"
            )
            try:
                await agent.call_tool("list_notes", {})
                out.after_revocation = "still worked"
            except Exception as exc:  # the SDK tries to re-authorize and Alice says no
                out.after_revocation = _root_cause(exc)
    except BaseException as exc:
        if phase != "revoked":
            raise
        out.after_revocation = out.after_revocation or _root_cause(exc)
    say(f"   the agent tries to sign in again, Alice denies -> {out.after_revocation}")

    say("5. Alice's activity log (hash-chained)")
    names = {urls.mcp: "notes-mcp", urls.api: "notes-api", urls.brasil: "brasil-mcp"}
    for event in await user.activity():
        d = event["detail"]
        what = d.get("tool") or d.get("method") or d.get("scope") or ""
        if event["event"] == "resource.call":
            what = f"{names.get(event['resource'], event['resource'])} {what} -> {d.get('status')}"
        chain = " via " + " <- ".join(d["actors"]) if d.get("actors") else ""
        out.events.append(event["event"])
        say(f"   {event['event']:<16} {(event['client_id'] or '')[:24]:<24} {what}{chain}")
    ok, count, head = stack.server.store.verify_audit()
    out.audit_ok = ok
    say(f"   chain verified: {ok} ({count} events, head {str(head)[:12]})")
    await user.browser.aclose()
    return out


async def amain(port_base: int, keep_running: bool) -> None:
    logging.basicConfig(level=logging.WARNING)
    # The SDK logs the refused re-authorization at the end with a traceback; it is expected here.
    logging.getLogger("mcp.client.auth.oauth2").setLevel(logging.CRITICAL)
    stack = await start_stack(port_base)
    try:
        await run_scenario(stack)
        if keep_running:
            print(f"\nServers are up: {stack.urls.issuer}/consents, user {USER} / {PASSWORD}. Ctrl+C stops.")
            await asyncio.Event().wait()
    finally:
        await stack.stop()


def main() -> None:
    parser = argparse.ArgumentParser(prog="agent-auth-demo")
    parser.add_argument("--port-base", type=int, default=9400)
    parser.add_argument("--keep-running", action="store_true")
    args = parser.parse_args()
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(amain(args.port_base, args.keep_running))


if __name__ == "__main__":
    main()
