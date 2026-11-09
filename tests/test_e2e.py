"""The demo over real sockets: uvicorn servers, the official MCP SDK client and its OAuth provider."""

from __future__ import annotations

import importlib.util

import pytest

from agent_auth.demo.run import free_port_base, run_scenario, start_stack

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


async def test_agent_gets_token_calls_mcp_and_loses_access_on_revocation() -> None:
    stack = await start_stack(free_port_base(), revocation_interval=0.5)
    lines: list[str] = []
    try:
        result = await run_scenario(stack, say=lines.append)
    finally:
        await stack.stop()

    assert result.agent_client_id.startswith("dcr_")
    assert result.stepped_up  # 403 insufficient_scope -> second consent with notes:write
    assert result.notes == [
        {"text": "Call Ana about the CNPJ migration", "written_by": ["notes-mcp", result.agent_client_id]}
    ]
    assert result.machine_on_user_api == 403
    assert result.old_token_status == 401
    assert "access_denied" in result.after_revocation
    assert result.audit_ok
    for event in (
        "consent.granted",
        "token.issued",
        "token.exchanged",
        "resource.call",
        "consent.revoked",
        "consent.denied",
    ):
        assert event in result.events
    if importlib.util.find_spec("mcp_server_brasil") is not None:
        assert result.brasil and '"valido": true' in result.brasil
