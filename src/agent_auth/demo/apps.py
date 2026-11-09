"""The protected services used by the demo.

- Notes API: a plain HTTP API that only accepts tokens issued for a user.
- Notes MCP: an MCP server whose tools call the Notes API. It never forwards the token it received;
  it exchanges it for a token scoped to the API (RFC 8693), so the API sees the whole chain.
- Brasil MCP: the real ``mcp-server-brasil`` wrapped by the guard, when it is installed.
"""

from __future__ import annotations

from typing import Any

import httpx
from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.types import ASGIApp

from ..guard import AuditSink, Principal, ProtectedResource, ServiceClient, get_principal, log_audit_sink


def build_notes_api(
    issuer: str,
    resource: str,
    *,
    http: httpx.AsyncClient | None = None,
    audit: AuditSink | None = log_audit_sink,
) -> ProtectedResource:
    notes: dict[str, list[dict[str, Any]]] = {}

    def who(request: Request) -> Principal:
        principal = get_principal(request)
        assert principal is not None  # the guard put it there
        return principal

    async def list_notes(request: Request) -> JSONResponse:
        p = who(request)
        if "notes:read" not in p.scopes:
            return JSONResponse({"error": "insufficient_scope"}, status_code=403)
        return JSONResponse({"notes": notes.get(p.subject, [])})

    async def add_note(request: Request) -> JSONResponse:
        p = who(request)
        if "notes:write" not in p.scopes:
            return JSONResponse({"error": "insufficient_scope"}, status_code=403)
        body = await request.json()
        text = str(body.get("text", ""))[:500]
        note = {"text": text, "written_by": list(p.actors) or [p.client_id]}
        notes.setdefault(p.subject, []).append(note)
        return JSONResponse(note, status_code=201)

    app = Starlette(
        routes=[Route("/notes", list_notes, methods=["GET"]), Route("/notes", add_note, methods=["POST"])]
    )
    return ProtectedResource(
        app,
        resource=resource,
        issuer=issuer,
        scopes_supported=["notes:read", "notes:write"],
        require_user=True,
        resource_name="Notes API",
        http=http,
        audit=audit,
    )


def build_notes_mcp(
    issuer: str,
    resource: str,
    api: str,
    service: ServiceClient,
    *,
    http: httpx.AsyncClient | None = None,
    audit: AuditSink | None = log_audit_sink,
) -> ProtectedResource:
    client = http or httpx.AsyncClient(timeout=5.0)
    mcp = MCPServer(
        name="notes-mcp", title="Notes", instructions="Read and write the signed-in user's notes."
    )

    def principal(ctx: Context) -> Principal:
        p = get_principal(ctx.request_context.request)
        if p is None:  # pragma: no cover - the guard runs before every request
            raise ToolError("not authenticated")
        return p

    async def call_api(p: Principal, method: str, scope: str, json: Any = None) -> Any:
        # Token exchange, not passthrough: the API gets its own short token naming this server as actor.
        token = await service.exchange(p, api, scope=scope)
        response = await client.request(
            method, f"{api}/notes", json=json, headers={"Authorization": f"Bearer {token}"}
        )
        if response.status_code >= 400:
            raise ToolError(f"notes API answered {response.status_code}")
        return response.json()

    @mcp.tool(title="List notes")
    async def list_notes(ctx: Context) -> dict[str, Any]:
        """List the user's notes."""
        result: dict[str, Any] = await call_api(principal(ctx), "GET", "notes:read")
        return result

    @mcp.tool(title="Add note")
    async def add_note(text: str, ctx: Context) -> dict[str, Any]:
        """Add a note for the user."""
        result: dict[str, Any] = await call_api(principal(ctx), "POST", "notes:write", {"text": text})
        return result

    app = mcp.streamable_http_app(stateless_http=True, json_response=True)
    return ProtectedResource(
        app,
        resource=resource,
        issuer=issuer,
        scopes_supported=["notes:read", "notes:write"],
        required_scopes=["notes:read"],
        tool_scopes={"add_note": "notes:write"},
        resource_name="Notes MCP",
        http=http,
        audit=audit,
    )


def build_brasil_mcp(
    issuer: str,
    resource: str,
    *,
    http: httpx.AsyncClient | None = None,
    audit: AuditSink | None = log_audit_sink,
) -> ASGIApp | None:
    """mcp-server-brasil behind the guard, unchanged. None when it isn't installed."""
    try:
        from mcp_server_brasil.server import build_server
    except ImportError:
        return None
    app = build_server().streamable_http_app(stateless_http=True, json_response=True)
    return ProtectedResource(
        app,
        resource=resource,
        issuer=issuer,
        scopes_supported=["brasil:read"],
        required_scopes=["brasil:read"],
        resource_name="Brasil MCP",
        http=http,
        audit=audit,
    )
