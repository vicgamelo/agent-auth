"""The normal paths: discovery, authorization code + PKCE, refresh, client credentials, token exchange,
consent management and the audit trail."""

from __future__ import annotations

import httpx
import jwt
import pytest
from conftest import (
    API,
    BRASIL,
    CLI_REDIRECT,
    ISSUER,
    MCP,
    Env,
    Pkce,
    authorize_params,
    get_code,
    login,
    protect,
    register,
    token,
    user_tokens,
)

pytestmark = pytest.mark.anyio


async def test_authorization_server_metadata(http: httpx.AsyncClient) -> None:
    meta = (await http.get(f"{ISSUER}/.well-known/oauth-authorization-server")).json()
    assert meta["issuer"] == ISSUER
    assert meta["code_challenge_methods_supported"] == ["S256"]
    assert meta["authorization_response_iss_parameter_supported"] is True
    assert meta["client_id_metadata_document_supported"] is True
    assert "urn:ietf:params:oauth:grant-type:token-exchange" in meta["grant_types_supported"]
    assert "client_secret_basic" not in meta["token_endpoint_auth_methods_supported"]
    jwks = (await http.get(meta["jwks_uri"])).json()
    assert jwks["keys"][0]["alg"] == "ES256"
    assert "d" not in jwks["keys"][0]


async def test_protected_resource_metadata_and_challenge(env: Env, http: httpx.AsyncClient) -> None:
    protect(env, MCP, ["notes:read", "notes:write"], required_scopes=["notes:read"], resource_name="Notes")
    prm = (await http.get("https://mcp.test/.well-known/oauth-protected-resource/mcp")).json()
    assert prm == {
        "resource": MCP,
        "authorization_servers": [ISSUER],
        "scopes_supported": ["notes:read", "notes:write"],
        "bearer_methods_supported": ["header"],
        "resource_name": "Notes",
    }
    r = await http.post(MCP, json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    assert r.status_code == 401
    challenge = r.headers["www-authenticate"]
    assert challenge.startswith("Bearer ")
    assert 'resource_metadata="https://mcp.test/.well-known/oauth-protected-resource/mcp"' in challenge
    assert 'scope="notes:read"' in challenge


async def test_authorization_code_flow_end_to_end(env: Env, http: httpx.AsyncClient) -> None:
    protect(env, MCP, ["notes:read"], required_scopes=["notes:read"])
    tokens = await user_tokens(env)
    assert tokens["token_type"] == "Bearer"
    assert tokens["expires_in"] == 300
    assert tokens["scope"] == "notes:read"
    claims = jwt.decode(tokens["access_token"], options={"verify_signature": False})
    assert jwt.get_unverified_header(tokens["access_token"])["typ"] == "at+jwt"
    assert claims["aud"] == MCP and claims["sub"] == "alice" and claims["client_id"] == "cli-agent"
    r = await http.get(MCP, headers={"Authorization": f"Bearer {tokens['access_token']}"})
    assert r.status_code == 200
    assert r.json()["sub"] == "alice"


async def test_authorization_response_carries_iss_and_state(env: Env) -> None:
    async with env.browser() as browser:
        await login(browser)
        result = await get_code(browser, authorize_params("cli-agent"))
    assert result["iss"] == ISSUER
    assert result["state"] == "st-123"


async def test_existing_consent_is_reused_for_same_client(env: Env) -> None:
    await user_tokens(env)
    async with env.browser() as browser:
        await login(browser)
        r = await browser.get("/authorize", params=authorize_params("cli-agent"))
    assert r.status_code == 302  # no consent screen the second time
    # but asking for more than was granted shows it again
    async with env.browser() as browser:
        await login(browser)
        r = await browser.get(
            "/authorize", params=authorize_params("cli-agent", scope="notes:read notes:write")
        )
    assert r.status_code == 200 and "notes:write" in r.text


async def test_loopback_redirect_may_change_port(env: Env) -> None:
    async with env.browser() as browser:
        await login(browser)
        result = await get_code(browser, authorize_params("cli-agent", "http://127.0.0.1:51234/callback"))
    assert "code" in result


async def test_refresh_rotates_and_can_narrow(env: Env, http: httpx.AsyncClient) -> None:
    tokens = await user_tokens(env, scope="notes:read notes:write")
    r = await token(
        http,
        {
            "grant_type": "refresh_token",
            "client_id": "cli-agent",
            "refresh_token": tokens["refresh_token"],
            "scope": "notes:read",
        },
    )
    assert r.status_code == 200, r.text
    new = r.json()
    assert new["scope"] == "notes:read"
    assert new["refresh_token"] != tokens["refresh_token"]
    # the rotated token still carries the full grant
    r = await token(
        http, {"grant_type": "refresh_token", "client_id": "cli-agent", "refresh_token": new["refresh_token"]}
    )
    assert r.json()["scope"] == "notes:read notes:write"


async def test_client_credentials_with_private_key_jwt(env: Env, http: httpx.AsyncClient) -> None:
    protect(env, BRASIL, ["brasil:read"], required_scopes=["brasil:read"])
    r = await token(
        http, {"grant_type": "client_credentials", "resource": BRASIL, **env.assertion("report-job")}
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert "refresh_token" not in body
    claims = jwt.decode(body["access_token"], options={"verify_signature": False})
    assert claims["sub"] == "report-job" and claims["sub_kind"] == "client"
    assert claims["scope"] == "brasil:read"  # only the scopes valid at that resource
    who = (await http.get(BRASIL, headers={"Authorization": f"Bearer {body['access_token']}"})).json()
    assert who["kind"] == "client"


async def test_token_exchange_builds_actor_chain(env: Env, http: httpx.AsyncClient) -> None:
    protect(env, API, ["notes:read", "notes:write"], required_scopes=["notes:read"], require_user=True)
    tokens = await user_tokens(env, scope="notes:read notes:write")
    service = env.service("notes-mcp")
    from agent_auth.guard import Principal

    subject = Principal(
        "alice", "user", "cli-agent", frozenset({"notes:read"}), (), "jti", 0, token=tokens["access_token"]
    )
    downstream = await service.exchange(subject, API, scope="notes:read")
    claims = jwt.decode(downstream, options={"verify_signature": False})
    original = jwt.decode(tokens["access_token"], options={"verify_signature": False})
    assert claims["aud"] == API
    assert claims["sub"] == "alice"
    assert claims["client_id"] == "notes-mcp"
    assert claims["act"] == {"sub": "notes-mcp", "act": {"sub": "cli-agent"}}
    assert claims["exp"] <= original["exp"]
    who = (await http.get(f"{API}/notes", headers={"Authorization": f"Bearer {downstream}"})).json()
    assert who["actors"] == ["notes-mcp", "cli-agent"]
    assert who["scopes"] == ["notes:read"]


async def test_mcp_tool_scopes(env: Env, http: httpx.AsyncClient) -> None:
    protect(
        env,
        MCP,
        ["notes:read", "notes:write"],
        required_scopes=["notes:read"],
        tool_scopes={"add_note": "notes:write"},
    )
    tokens = await user_tokens(env, scope="notes:read")
    auth = {"Authorization": f"Bearer {tokens['access_token']}"}
    listing = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": "list_notes", "arguments": {}},
    }
    assert (await http.post(MCP, json=listing, headers=auth)).status_code == 200
    adding = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {"name": "add_note", "arguments": {}},
    }
    r = await http.post(MCP, json=adding, headers=auth)
    assert r.status_code == 403
    challenge = r.headers["www-authenticate"]
    assert 'error="insufficient_scope"' in challenge
    # every scope the operation needs, so the client can step up in one go
    assert 'scope="notes:read notes:write"' in challenge
    # a batch that hides a write behind a read still needs the write scope
    r = await http.post(MCP, json=[listing, adding], headers=auth)
    assert r.status_code == 403


async def test_body_is_replayed_to_the_app(env: Env, http: httpx.AsyncClient) -> None:
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import JSONResponse
    from starlette.routing import Route

    async def echo(request: Request) -> JSONResponse:
        return JSONResponse(await request.json())

    protect(
        env,
        MCP,
        ["notes:read"],
        required_scopes=["notes:read"],
        app=Starlette(routes=[Route("/mcp", echo, methods=["POST"])]),
    )
    tokens = await user_tokens(env)
    payload = {
        "jsonrpc": "2.0",
        "id": 7,
        "method": "tools/call",
        "params": {"name": "x", "arguments": {"a": 1}},
    }
    r = await http.post(MCP, json=payload, headers={"Authorization": f"Bearer {tokens['access_token']}"})
    assert r.json() == payload


async def test_consent_page_lists_and_revokes(env: Env) -> None:
    await user_tokens(env)
    async with env.browser() as browser:
        await login(browser)
        data = (await browser.get("/consents", headers={"Accept": "application/json"})).json()
        assert [c["client_id"] for c in data["consents"]] == ["cli-agent"]
        html = (await browser.get("/consents")).text
        assert "CLI agent" in html and "Notes MCP" in html
        grant_id = data["consents"][0]["id"]
        r = await browser.post(f"/consents/{grant_id}/revoke", data={"csrf": data["csrf"]})
        assert r.status_code == 303
        data = (await browser.get("/consents", headers={"Accept": "application/json"})).json()
        assert data["consents"] == []


async def test_dcr_client_can_authorize(env: Env, http: httpx.AsyncClient) -> None:
    r = await register(http)
    assert r.status_code == 201, r.text
    client_id = r.json()["client_id"]
    assert client_id.startswith("dcr_")
    tokens = await user_tokens(env, client_id=client_id)
    assert "access_token" in tokens


async def test_consent_screen_shows_redirect_host_and_unverified_badge(
    env: Env, http: httpx.AsyncClient
) -> None:
    client_id = (
        await register(http, {"client_name": "Totally Claude", "redirect_uris": ["https://evil.example/cb"]})
    ).json()["client_id"]
    async with env.browser() as browser:
        await login(browser)
        r = await browser.get("/authorize", params=authorize_params(client_id, "https://evil.example/cb"))
    assert r.status_code == 200
    assert "evil.example" in r.text
    assert "not verified" in r.text
    assert "Notes API" in r.text  # downstream delegation is disclosed


async def test_audit_trail_records_who_did_what(env: Env, http: httpx.AsyncClient) -> None:
    events: list[dict[str, object]] = []
    protect(env, MCP, ["notes:read"], required_scopes=["notes:read"], events=events)
    tokens = await user_tokens(env)
    call = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "list_notes"}}
    await http.post(MCP, json=call, headers={"Authorization": f"Bearer {tokens['access_token']}"})
    import asyncio

    await asyncio.sleep(0)
    assert events and events[0]["tool"] == "list_notes" and events[0]["subject"] == "alice"
    assert events[0]["client_id"] == "cli-agent" and events[0]["status"] == 200
    # the resource server reports it to the authorization server
    await env.service("notes-mcp").report([e for e in events])
    kinds = [e.event for e in env.store.audit_events("alice")]
    assert kinds[:4] == ["user.login", "consent.granted", "token.issued", "resource.call"]
    ok, count, _ = env.store.verify_audit()
    assert ok and count >= 4
    async with env.browser() as browser:
        await login(browser)
        activity = (await browser.get("/activity", headers={"Accept": "application/json"})).json()
    call_event = next(e for e in activity["events"] if e["event"] == "resource.call")
    assert call_event["detail"]["tool"] == "list_notes" and call_event["resource"] == MCP


async def test_audit_chain_detects_tampering(env: Env) -> None:
    await user_tokens(env)
    assert env.store.verify_audit()[0]
    env.store._exec("UPDATE audit SET client_id='someone-else' WHERE seq=2")
    ok, _, where = env.store.verify_audit()
    assert not ok and where == "seq 2"


async def test_pkce_s256_reference_vector() -> None:
    # RFC 7636 appendix B
    from agent_auth.crypto import pkce_s256

    assert (
        pkce_s256("dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk")
        == "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM"
    )


async def test_redirect_uri_must_match_token_request(env: Env, http: httpx.AsyncClient) -> None:
    async with env.browser() as browser:
        await login(browser)
        pkce = Pkce()
        result = await get_code(browser, authorize_params("cli-agent", pkce=pkce))
    r = await token(
        http,
        {
            "grant_type": "authorization_code",
            "client_id": "cli-agent",
            "code": result["code"],
            "code_verifier": pkce.verifier,
            "redirect_uri": CLI_REDIRECT,
        },
    )
    assert r.status_code == 200  # `resource` is optional on the token request; the code is bound to it
