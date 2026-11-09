"""Each test is an attack. They all must fail for the attacker."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import time

import httpx
import jwt
import pytest
from conftest import (
    API,
    BRASIL,
    CLI_REDIRECT,
    ISSUER,
    MCP,
    OTHER,
    Env,
    Pkce,
    approve,
    authorize_params,
    get_code,
    login,
    make_env,
    protect,
    query,
    register,
    token,
    user_tokens,
)

from agent_auth.config import Limits
from agent_auth.crypto import JWT_BEARER, SigningKey, client_assertion
from agent_auth.guard import Principal

pytestmark = pytest.mark.anyio

TX = "urn:ietf:params:oauth:grant-type:token-exchange"
AT = "urn:ietf:params:oauth:token-type:access_token"


def principal_for(token_value: str) -> Principal:
    claims = jwt.decode(token_value, options={"verify_signature": False})
    return Principal(
        claims["sub"],
        claims.get("sub_kind", "user"),
        claims["client_id"],
        frozenset(claims["scope"].split()),
        (),
        claims["jti"],
        claims["exp"],
        token=token_value,
    )


async def code_and_verifier(env: Env, client_id: str = "cli-agent", **kw: str) -> tuple[str, Pkce]:
    pkce = Pkce()
    async with env.browser() as browser:
        await login(browser)
        result = await get_code(browser, authorize_params(client_id, pkce=pkce, **kw))
    return result["code"], pkce


def code_form(code: str, pkce: Pkce, **over: str) -> dict[str, str]:
    form = {
        "grant_type": "authorization_code",
        "client_id": "cli-agent",
        "code": code,
        "code_verifier": pkce.verifier,
        "redirect_uri": CLI_REDIRECT,
    }
    form.update(over)
    return {k: v for k, v in form.items() if v != "__omit__"}


# ---- 1. redirect_uri swapped -----------------------------------------------------------------


@pytest.mark.parametrize(
    "evil",
    [
        "https://attacker.example/callback",
        "http://127.0.0.1:8765/callback/../steal",
        "http://127.0.0.1:8765/other",
        "http://127.0.0.1.attacker.example:8765/callback",
        "http://localhost:8765/callback",  # registered host is 127.0.0.1, not localhost
        "http://127.0.0.1:8765/callback?x=1",
        "http://127.0.0.1:8765/callback#frag",
        "http://user@127.0.0.1:8765/callback",
    ],
)
async def test_unregistered_redirect_uri_is_not_followed(env: Env, evil: str) -> None:
    async with env.browser() as browser:
        await login(browser)
        r = await browser.get("/authorize", params=authorize_params("cli-agent", evil))
    assert r.status_code == 400
    assert "location" not in r.headers  # no open redirect, not even with an error


async def test_redirect_uri_repeated_is_rejected(env: Env) -> None:
    params = [*authorize_params("cli-agent").items(), ("redirect_uri", "https://attacker.example/cb")]
    async with env.browser() as browser:
        await login(browser)
        r = await browser.get("/authorize", params=params)
    assert r.status_code == 400 and "location" not in r.headers


async def test_token_request_with_other_redirect_uri_fails(env: Env, http: httpx.AsyncClient) -> None:
    code, pkce = await code_and_verifier(env)
    r = await token(http, code_form(code, pkce, redirect_uri="http://127.0.0.1:9999/callback"))
    assert r.status_code == 400 and r.json()["error"] == "invalid_grant"


async def test_login_next_cannot_redirect_off_site(env: Env) -> None:
    async with env.browser() as browser:
        page = await browser.get("/login", params={"next": "//attacker.example/x"})
        csrf = page.text.split('name=csrf value="')[1].split('"')[0]
        for evil in ("//attacker.example/x", "https://attacker.example", "/\\attacker.example"):
            r = await browser.post(
                "/login", data={"csrf": csrf, "username": "alice", "password": "pw-alice", "next": evil}
            )
            if r.status_code == 403:  # the login cookie is single use; fetch a new one
                page = await browser.get("/login")
                csrf = page.text.split('name=csrf value="')[1].split('"')[0]
                r = await browser.post(
                    "/login", data={"csrf": csrf, "username": "alice", "password": "pw-alice", "next": evil}
                )
            assert r.status_code == 303 and r.headers["location"] == "/consents"


# ---- 2. PKCE missing or wrong ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("override", "description"),
    [
        ({"code_challenge": "__omit__"}, "code_challenge is required"),
        ({"code_challenge_method": "plain"}, "S256"),
        ({"code_challenge_method": "__omit__"}, "S256"),
        ({"code_challenge": "short"}, "malformed"),
    ],
)
async def test_authorization_without_proper_pkce_is_refused(
    env: Env, override: dict[str, str], description: str
) -> None:
    async with env.browser() as browser:
        await login(browser)
        r = await browser.get("/authorize", params=authorize_params("cli-agent", **override))
    assert r.status_code == 302
    q = query(r.headers["location"])
    assert q["error"] == "invalid_request" and description in q["error_description"]
    assert q["iss"] == ISSUER and q["state"] == "st-123"
    assert "code" not in q


@pytest.mark.parametrize("verifier", ["__omit__", "x" * 43, "too-short"])
async def test_token_with_wrong_or_missing_verifier_fails(
    env: Env, http: httpx.AsyncClient, verifier: str
) -> None:
    code, pkce = await code_and_verifier(env)
    r = await token(http, code_form(code, pkce, code_verifier=verifier))
    assert r.status_code == 400 and r.json()["error"] == "invalid_grant"


async def test_failed_pkce_burns_the_code(env: Env, http: httpx.AsyncClient) -> None:
    """Guessing verifiers one after another doesn't work: the first wrong try uses the code up."""
    code, pkce = await code_and_verifier(env)
    assert (await token(http, code_form(code, pkce, code_verifier="y" * 43))).status_code == 400
    assert (await token(http, code_form(code, pkce))).status_code == 400


# ---- 3. token for another audience ------------------------------------------------------------


async def test_token_for_one_resource_is_rejected_by_another(env: Env, http: httpx.AsyncClient) -> None:
    protect(env, API, ["notes:read"], required_scopes=["notes:read"])
    protect(env, MCP, ["notes:read"], required_scopes=["notes:read"])
    mcp_token = (await user_tokens(env))["access_token"]
    assert (await http.get(MCP, headers={"Authorization": f"Bearer {mcp_token}"})).status_code == 200
    r = await http.get(API, headers={"Authorization": f"Bearer {mcp_token}"})
    assert r.status_code == 401
    assert 'error="invalid_token"' in r.headers["www-authenticate"]
    assert "different resource" in r.json()["error_description"]


async def test_authorize_needs_a_known_resource(env: Env) -> None:
    async with env.browser() as browser:
        await login(browser)
        for resource, error in ((None, "invalid_target"), ("https://unknown.example", "invalid_target")):
            r = await browser.get("/authorize", params=authorize_params("cli-agent", resource=resource))
            assert query(r.headers["location"])["error"] == error


async def test_token_request_cannot_retarget_the_code(env: Env, http: httpx.AsyncClient) -> None:
    code, pkce = await code_and_verifier(env)
    r = await token(http, code_form(code, pkce, resource=API))
    assert r.status_code == 400 and r.json()["error"] == "invalid_target"


async def test_forged_tokens_are_rejected(env: Env, http: httpx.AsyncClient) -> None:
    protect(env, MCP, ["notes:read"], required_scopes=["notes:read"])
    real = (await user_tokens(env))["access_token"]
    claims = jwt.decode(real, options={"verify_signature": False})
    header = jwt.get_unverified_header(real)

    def b64(obj: dict[str, object]) -> str:
        return base64.urlsafe_b64encode(json.dumps(obj).encode()).rstrip(b"=").decode()

    claims_admin = {**claims, "scope": "notes:read notes:write", "sub": "bob"}

    def hs256(secret: bytes) -> str:  # PyJWT refuses to build this one, so it is done by hand
        signing_input = f"{b64({'alg': 'HS256', 'typ': 'at+jwt', 'kid': header['kid']})}.{b64(claims_admin)}"
        mac = hmac.new(secret, signing_input.encode(), hashlib.sha256).digest()
        return signing_input + "." + base64.urlsafe_b64encode(mac).rstrip(b"=").decode()

    forgeries = {
        "alg none": f"{b64({'alg': 'none', 'typ': 'at+jwt'})}.{b64(claims_admin)}.",
        "other key": SigningKey.generate().sign(claims_admin, typ="at+jwt"),
        "hs256 with public key": hs256(json.dumps(env.server.key.public_jwk).encode()),
        "edited payload": real.split(".")[0] + "." + b64(claims_admin) + "." + real.split(".")[2],
        "client assertion as access token": client_assertion("notes-mcp", env.keys["notes-mcp"], MCP),
    }
    for name, forged in forgeries.items():
        r = await http.get(MCP, headers={"Authorization": f"Bearer {forged}"})
        assert r.status_code == 401, name


async def test_expired_token_is_rejected(env: Env, http: httpx.AsyncClient) -> None:
    protect(env, MCP, ["notes:read"], required_scopes=["notes:read"])
    access = (await user_tokens(env))["access_token"]
    env.clock.advance(300 + 31)
    r = await http.get(MCP, headers={"Authorization": f"Bearer {access}"})
    assert r.status_code == 401 and "expired" in r.json()["error_description"]


async def test_token_in_query_string_is_ignored(env: Env, http: httpx.AsyncClient) -> None:
    protect(env, MCP, ["notes:read"], required_scopes=["notes:read"])
    access = (await user_tokens(env))["access_token"]
    assert (await http.get(MCP, params={"access_token": access})).status_code == 401


# ---- 4. scope widened ------------------------------------------------------------------------


async def test_exchange_cannot_widen_scope(env: Env) -> None:
    access = (await user_tokens(env, scope="notes:read"))["access_token"]
    with pytest.raises(PermissionError, match="invalid_scope"):
        await env.service("notes-mcp").exchange(principal_for(access), API, scope="notes:read notes:write")
    assert any(e.event == "exchange.rejected" for e in env.store.audit_events("alice"))


async def test_exchange_cannot_outlive_the_subject_token(env: Env) -> None:
    access = (await user_tokens(env))["access_token"]
    env.clock.advance(200)
    downstream = await env.service("notes-mcp").exchange(principal_for(access), API)
    assert (
        jwt.decode(downstream, options={"verify_signature": False})["exp"]
        == jwt.decode(access, options={"verify_signature": False})["exp"]
    )


async def test_refresh_cannot_widen_scope(env: Env, http: httpx.AsyncClient) -> None:
    tokens = await user_tokens(env, scope="notes:read")
    r = await token(
        http,
        {
            "grant_type": "refresh_token",
            "client_id": "cli-agent",
            "refresh_token": tokens["refresh_token"],
            "scope": "notes:read notes:write",
        },
    )
    assert r.status_code == 400 and r.json()["error"] == "invalid_scope"


async def test_scope_outside_the_resource_is_refused(env: Env) -> None:
    async with env.browser() as browser:
        await login(browser)
        r = await browser.get(
            "/authorize", params=authorize_params("cli-agent", scope="notes:read brasil:read")
        )
    assert query(r.headers["location"])["error"] == "invalid_scope"


async def test_client_credentials_scope_is_capped(env: Env, http: httpx.AsyncClient) -> None:
    r = await token(
        http,
        {
            "grant_type": "client_credentials",
            "resource": API,
            "scope": "notes:write",
            **env.assertion("report-job"),
        },
    )
    assert r.status_code == 400 and r.json()["error"] == "invalid_scope"
    r = await token(
        http, {"grant_type": "client_credentials", "resource": MCP, **env.assertion("report-job")}
    )
    assert r.status_code == 400 and r.json()["error"] == "invalid_target"


# ---- 5. code replay --------------------------------------------------------------------------


async def test_code_replay_fails_and_revokes_what_it_issued(env: Env, http: httpx.AsyncClient) -> None:
    protect(env, MCP, ["notes:read"], required_scopes=["notes:read"])
    code, pkce = await code_and_verifier(env)
    first = await token(http, code_form(code, pkce))
    assert first.status_code == 200
    tokens = first.json()
    second = await token(http, code_form(code, pkce))
    assert second.status_code == 400 and second.json()["error"] == "invalid_grant"
    # the tokens from the first redemption die with it
    r = await http.get(MCP, headers={"Authorization": f"Bearer {tokens['access_token']}"})
    assert r.status_code == 401 and "revoked" in r.json()["error_description"]
    r = await token(
        http,
        {"grant_type": "refresh_token", "client_id": "cli-agent", "refresh_token": tokens["refresh_token"]},
    )
    assert r.status_code == 400
    assert any(e.event == "code.replay" for e in env.store.audit_events("alice"))


async def test_concurrent_redemption_issues_one_token(env: Env, http: httpx.AsyncClient) -> None:
    code, pkce = await code_and_verifier(env)
    results = await asyncio.gather(*(token(http, code_form(code, pkce)) for _ in range(5)))
    assert sorted(r.status_code for r in results).count(200) <= 1


async def test_expired_code_fails(env: Env, http: httpx.AsyncClient) -> None:
    code, pkce = await code_and_verifier(env)
    env.clock.advance(61)
    assert (await token(http, code_form(code, pkce))).json()["error"] == "invalid_grant"


async def test_code_from_another_client_fails(env: Env, http: httpx.AsyncClient) -> None:
    code, pkce = await code_and_verifier(env)
    other = (await register(http)).json()["client_id"]
    r = await token(http, code_form(code, pkce, client_id=other))
    assert r.json()["error"] == "invalid_grant"
    # and the legitimate client can still redeem it: the attacker can't burn codes
    assert (await token(http, code_form(code, pkce))).status_code == 200


async def test_refresh_token_reuse_revokes_the_family(env: Env, http: httpx.AsyncClient) -> None:
    tokens = await user_tokens(env)
    rotated = (
        await token(
            http,
            {
                "grant_type": "refresh_token",
                "client_id": "cli-agent",
                "refresh_token": tokens["refresh_token"],
            },
        )
    ).json()
    stolen = await token(
        http,
        {"grant_type": "refresh_token", "client_id": "cli-agent", "refresh_token": tokens["refresh_token"]},
    )
    assert stolen.status_code == 400
    after = await token(
        http,
        {"grant_type": "refresh_token", "client_id": "cli-agent", "refresh_token": rotated["refresh_token"]},
    )
    assert after.status_code == 400  # the whole family is gone
    assert any(e.event == "refresh.reuse" for e in env.store.audit_events("alice"))


async def test_client_assertion_replay_fails(env: Env, http: httpx.AsyncClient) -> None:
    form = {"grant_type": "client_credentials", "resource": BRASIL, **env.assertion("report-job")}
    assert (await token(http, form)).status_code == 200
    r = await token(http, form)
    assert r.status_code == 401 and "already used" in r.json()["error_description"]


async def test_client_assertion_must_be_short_lived_and_for_this_server(
    env: Env, http: httpx.AsyncClient
) -> None:
    key = env.keys["report-job"]
    now = int(time.time())
    long_lived = key.sign(
        {"iss": "report-job", "sub": "report-job", "aud": ISSUER, "iat": now, "exp": now + 3600, "jti": "a"},
        typ="JWT",
    )
    wrong_aud = client_assertion("report-job", key, "https://other-as.example")
    for assertion in (long_lived, wrong_aud):
        r = await token(
            http,
            {
                "grant_type": "client_credentials",
                "resource": BRASIL,
                "client_assertion_type": JWT_BEARER,
                "client_assertion": assertion,
            },
        )
        assert r.status_code == 401


async def test_client_secret_is_not_accepted(env: Env, http: httpx.AsyncClient) -> None:
    r = await token(
        http,
        {
            "grant_type": "client_credentials",
            "resource": BRASIL,
            "client_id": "report-job",
            "client_secret": "guess",
        },
    )
    assert r.status_code == 401


# ---- 6. revoked token ------------------------------------------------------------------------


async def test_revoked_access_token_stops_working(env: Env, http: httpx.AsyncClient) -> None:
    protect(env, MCP, ["notes:read"], required_scopes=["notes:read"])
    access = (await user_tokens(env))["access_token"]
    auth = {"Authorization": f"Bearer {access}"}
    assert (await http.get(MCP, headers=auth)).status_code == 200
    r = await http.post(f"{ISSUER}/revoke", data={"token": access, "client_id": "cli-agent"})
    assert r.status_code == 200
    r = await http.get(MCP, headers=auth)
    assert r.status_code == 401 and "revoked" in r.json()["error_description"]


async def test_revoking_consent_cuts_agent_and_downstream(env: Env, http: httpx.AsyncClient) -> None:
    protect(env, MCP, ["notes:read"], required_scopes=["notes:read"])
    protect(env, API, ["notes:read"], required_scopes=["notes:read"], require_user=True)
    tokens = await user_tokens(env)
    downstream = await env.service("notes-mcp").exchange(principal_for(tokens["access_token"]), API)
    assert (await http.get(API, headers={"Authorization": f"Bearer {downstream}"})).status_code == 200
    async with env.browser() as browser:
        await login(browser)
        data = (await browser.get("/consents", headers={"Accept": "application/json"})).json()
        await browser.post(f"/consents/{data['consents'][0]['id']}/revoke", data={"csrf": data["csrf"]})
    assert (
        await http.get(MCP, headers={"Authorization": f"Bearer {tokens['access_token']}"})
    ).status_code == 401
    assert (await http.get(API, headers={"Authorization": f"Bearer {downstream}"})).status_code == 401
    r = await token(
        http,
        {"grant_type": "refresh_token", "client_id": "cli-agent", "refresh_token": tokens["refresh_token"]},
    )
    assert r.status_code == 400
    # and the revoked token can't be exchanged again
    with pytest.raises(PermissionError, match="revoked"):
        await env.service("notes-mcp").exchange(
            principal_for(tokens["access_token"]), API, scope="notes:read"
        )


async def test_revoking_one_token_revokes_tokens_exchanged_from_it(env: Env, http: httpx.AsyncClient) -> None:
    protect(env, API, ["notes:read"], required_scopes=["notes:read"])
    access = (await user_tokens(env))["access_token"]
    downstream = await env.service("notes-mcp").exchange(principal_for(access), API)
    await http.post(f"{ISSUER}/revoke", data={"token": access, "client_id": "cli-agent"})
    assert (await http.get(API, headers={"Authorization": f"Bearer {downstream}"})).status_code == 401


async def test_client_cannot_revoke_someone_elses_token(env: Env, http: httpx.AsyncClient) -> None:
    protect(env, MCP, ["notes:read"], required_scopes=["notes:read"])
    access = (await user_tokens(env))["access_token"]
    other = (await register(http)).json()["client_id"]
    await http.post(f"{ISSUER}/revoke", data={"token": access, "client_id": other})
    assert (await http.get(MCP, headers={"Authorization": f"Bearer {access}"})).status_code == 200


async def test_stale_revocation_list_fails_closed(env: Env, http: httpx.AsyncClient) -> None:
    from agent_auth.guard import ProtectedResource, TokenVerifier

    access = (await user_tokens(env))["access_token"]
    verifier = TokenVerifier(
        ISSUER, MCP, http=env.http(), clock=env.clock, revocation_interval=5, max_staleness=60
    )
    guard = ProtectedResource(
        lambda s, r, se: None,
        resource=MCP,
        issuer=ISSUER,  # type: ignore[arg-type,return-value]
        scopes_supported=["notes:read"],
        verifier=verifier,
        audit=None,
    )
    await verifier.verify(access)  # primes keys and revocation list
    env.router.transports.pop("as.test")  # the authorization server goes away
    env.clock.advance(30)
    await verifier.verify(access)  # within max_staleness: still served from cache
    env.clock.advance(60)
    env.router.transports["mcp.test"] = httpx.ASGITransport(app=guard)
    r = await http.get(MCP, headers={"Authorization": f"Bearer {access}"})
    assert r.status_code == 503


# ---- 7. client registration abuse ------------------------------------------------------------


async def test_registration_flood_is_throttled(env: Env, http: httpx.AsyncClient) -> None:
    statuses = [(await register(http)).status_code for _ in range(6)]
    assert statuses[:3] == [201, 201, 201]
    assert set(statuses[3:]) == {429}
    rejected = [e for e in env.store.audit_events() if e.event == "registration.rejected"]
    assert len(rejected) == 3


async def test_forwarded_for_does_not_bypass_the_limit(env: Env, http: httpx.AsyncClient) -> None:
    statuses = [(await register(http, **{"X-Forwarded-For": f"203.0.113.{i}"})).status_code for i in range(6)]
    assert statuses.count(429) == 3  # no trusted proxy: the header is ignored


async def test_registration_cap_and_expiry_of_unused_clients() -> None:
    env = make_env(Limits(dcr_max_clients=2, dcr_burst=10, dcr_per_ip_per_hour=1000, dcr_unused_ttl=3600))
    async with env.http() as http:
        assert [(await register(http)).status_code for _ in range(3)] == [201, 201, 503]
        env.clock.advance(3601)  # never used: pruned
        assert (await register(http)).status_code == 201


@pytest.mark.parametrize(
    "meta",
    [
        {"redirect_uris": ["http://attacker.example/cb"]},
        {"redirect_uris": ["javascript:alert(1)"]},
        {"redirect_uris": ["https://ok.example/cb#frag"]},
        {"redirect_uris": [f"https://ok.example/{i}" for i in range(6)]},
        {"redirect_uris": []},
        {"redirect_uris": [CLI_REDIRECT], "token_endpoint_auth_method": "client_secret_basic"},
        {"redirect_uris": [CLI_REDIRECT], "grant_types": ["client_credentials"]},
        {"redirect_uris": [CLI_REDIRECT], "grant_types": [TX]},
        {"redirect_uris": [CLI_REDIRECT], "response_types": ["token"]},
        {"redirect_uris": [CLI_REDIRECT], "client_name": "x" * 101},
        {"redirect_uris": [CLI_REDIRECT], "client_name": "Claude‮evil"},
    ],
)
async def test_bad_registration_metadata_is_refused(meta: dict[str, object]) -> None:
    env = make_env(Limits(dcr_burst=100, dcr_per_ip_per_hour=1000))
    async with env.http() as http:
        r = await register(http, meta)
    assert r.status_code == 400, r.text
    assert r.json()["error"] in ("invalid_redirect_uri", "invalid_client_metadata")


async def test_oversized_registration_is_refused(http: httpx.AsyncClient) -> None:
    r = await http.post(
        f"{ISSUER}/register", content=b"{" + b" " * 20000 + b"}", headers={"content-type": "application/json"}
    )
    assert r.status_code == 413


async def test_dcr_clients_cannot_use_confidential_grants(env: Env, http: httpx.AsyncClient) -> None:
    client_id = (await register(http)).json()["client_id"]
    r = await token(http, {"grant_type": "client_credentials", "client_id": client_id, "resource": BRASIL})
    assert r.status_code == 400 and r.json()["error"] == "unauthorized_client"


# ---- 8. confused deputy ----------------------------------------------------------------------


async def test_consent_for_one_agent_does_not_cover_another(env: Env, http: httpx.AsyncClient) -> None:
    """Alice trusted the CLI agent. A look-alike client must get its own consent screen."""
    await user_tokens(env)
    impostor = (await register(http, {"client_name": "CLI agent", "redirect_uris": [CLI_REDIRECT]})).json()[
        "client_id"
    ]
    async with env.browser() as browser:
        await login(browser)
        r = await browser.get("/authorize", params=authorize_params(impostor))
        assert r.status_code == 200 and "not verified" in r.text
        denied = await approve(browser, r, decision="deny")
    q = query(denied.headers["location"])
    assert q["error"] == "access_denied" and "code" not in q


async def test_service_cannot_cash_in_a_token_sent_to_another_service(env: Env) -> None:
    """other-mcp got hold of a token meant for notes-mcp and tries to turn it into API access."""
    access = (await user_tokens(env))["access_token"]  # aud = notes MCP
    with pytest.raises(PermissionError, match="not issued to the requesting client"):
        await env.service("other-mcp").exchange(principal_for(access), API)
    rejected = [e for e in env.store.audit_events("alice") if e.event == "exchange.rejected"]
    assert rejected and rejected[0].client_id == "other-mcp"


async def test_service_cannot_exchange_into_an_undelegated_resource(env: Env) -> None:
    access = (await user_tokens(env))["access_token"]
    with pytest.raises(PermissionError, match="invalid_target"):
        await env.service("notes-mcp").exchange(principal_for(access), BRASIL)
    with pytest.raises(PermissionError, match="invalid_target"):
        await env.service("notes-mcp").exchange(principal_for(access), OTHER)


async def test_actor_token_splicing_is_refused(env: Env, http: httpx.AsyncClient) -> None:
    """A compromised intermediary tries to name someone else as the actor."""
    access = (await user_tokens(env))["access_token"]
    someone_else = (
        await token(
            http, {"grant_type": "client_credentials", "resource": API, **env.assertion("report-job")}
        )
    ).json()["access_token"]
    r = await token(
        http,
        {
            "grant_type": TX,
            "subject_token": access,
            "subject_token_type": AT,
            "resource": API,
            "actor_token": someone_else,
            "actor_token_type": AT,
            **env.assertion("notes-mcp"),
        },
    )
    assert r.status_code == 400 and "actor_token is not accepted" in r.json()["error_description"]


async def test_agent_cannot_use_its_own_identity_for_user_data(env: Env, http: httpx.AsyncClient) -> None:
    """A machine token (no user behind it) can't read a user's notes, even with the right scope."""
    protect(env, API, ["notes:read"], required_scopes=["notes:read"], require_user=True)
    machine = (
        await token(
            http, {"grant_type": "client_credentials", "resource": API, **env.assertion("report-job")}
        )
    ).json()["access_token"]
    r = await http.get(f"{API}/notes", headers={"Authorization": f"Bearer {machine}"})
    assert r.status_code == 403 and r.json()["error"] == "user_required"


async def test_public_agent_cannot_exchange_tokens(env: Env, http: httpx.AsyncClient) -> None:
    access = (await user_tokens(env))["access_token"]
    r = await token(
        http,
        {
            "grant_type": TX,
            "client_id": "cli-agent",
            "subject_token": access,
            "subject_token_type": AT,
            "resource": API,
        },
    )
    assert r.status_code == 400 and r.json()["error"] == "unauthorized_client"


async def test_exchanged_token_cannot_be_exchanged_by_its_target(env: Env) -> None:
    """The API (if it were a client) would see aud=API; notes-mcp cannot re-exchange the API token."""
    access = (await user_tokens(env))["access_token"]
    downstream = await env.service("notes-mcp").exchange(principal_for(access), API)
    with pytest.raises(PermissionError, match="not issued to the requesting client"):
        await env.service("notes-mcp").exchange(principal_for(downstream), API)


async def test_consent_decision_needs_csrf_and_same_origin(env: Env) -> None:
    async with env.browser() as browser:
        await login(browser)
        page = await browser.get("/authorize", params=authorize_params("cli-agent"))
        request_id = page.text.split('name=request value="')[1].split('"')[0]
        csrf = page.text.split('name=csrf value="')[1].split('"')[0]
        r = await browser.post(
            "/authorize/decision", data={"request": request_id, "decision": "approve", "csrf": "forged"}
        )
        assert r.status_code == 403
        r = await browser.post(
            "/authorize/decision",
            data={"request": request_id, "decision": "approve", "csrf": csrf},
            headers={"Origin": "https://attacker.example"},
        )
        assert r.status_code == 403
        r = await browser.post(
            "/authorize/decision",
            data={"request": request_id, "decision": "approve", "csrf": csrf},
            headers={"Sec-Fetch-Site": "cross-site"},
        )
        assert r.status_code == 403


async def test_pending_request_belongs_to_the_session(env: Env) -> None:
    """Bob can't approve a request that was shown to Alice."""
    async with env.browser() as alice:
        await login(alice)
        page = await alice.get("/authorize", params=authorize_params("cli-agent"))
        request_id = page.text.split('name=request value="')[1].split('"')[0]
    async with env.browser() as bob:
        await login(bob, "bob")
        consents = (await bob.get("/consents", headers={"Accept": "application/json"})).json()
        r = await bob.post(
            "/authorize/decision",
            data={"request": request_id, "decision": "approve", "csrf": consents["csrf"]},
        )
    assert r.status_code == 400


async def test_user_cannot_revoke_another_users_consent(env: Env) -> None:
    await user_tokens(env)
    grant_id = env.store.grants_for_user("alice")[0].id
    async with env.browser() as bob:
        await login(bob, "bob")
        csrf = (await bob.get("/consents", headers={"Accept": "application/json"})).json()["csrf"]
        r = await bob.post(f"/consents/{grant_id}/revoke", data={"csrf": csrf})
    assert r.status_code == 404
    assert env.store.grants_for_user("alice")


async def test_consent_page_cannot_be_framed(env: Env) -> None:
    async with env.browser() as browser:
        await login(browser)
        r = await browser.get("/authorize", params=authorize_params("cli-agent"))
    assert r.headers["x-frame-options"] == "DENY"
    assert "frame-ancestors 'none'" in r.headers["content-security-policy"]
    assert r.headers["referrer-policy"] == "no-referrer"


async def test_login_is_throttled_and_locks(env: Env) -> None:
    async with env.browser() as browser:
        statuses = []
        for _ in range(7):
            page = await browser.get("/login")
            csrf = page.text.split('name=csrf value="')[1].split('"')[0]
            r = await browser.post("/login", data={"csrf": csrf, "username": "alice", "password": "wrong"})
            statuses.append(r.status_code)
        page = await browser.get("/login")
        csrf = page.text.split('name=csrf value="')[1].split('"')[0]
        r = await browser.post("/login", data={"csrf": csrf, "username": "alice", "password": "pw-alice"})
    assert statuses[:5] == [401] * 5
    assert 429 in statuses[5:]
    assert r.status_code == 429  # locked even with the right password


async def test_xss_in_client_name_is_escaped(env: Env, http: httpx.AsyncClient) -> None:
    client_id = (
        await register(http, {"client_name": "<script>alert(1)</script>", "redirect_uris": [CLI_REDIRECT]})
    ).json()["client_id"]
    async with env.browser() as browser:
        await login(browser)
        r = await browser.get("/authorize", params=authorize_params(client_id))
    assert "<script>" not in r.text and "&lt;script&gt;" in r.text


async def test_repeated_parameters_are_refused(env: Env, http: httpx.AsyncClient) -> None:
    code, pkce = await code_and_verifier(env)
    body = "&".join(f"{k}={v}" for k, v in code_form(code, pkce).items()) + "&code_verifier=" + "z" * 43
    r = await http.post(
        f"{ISSUER}/token", content=body, headers={"content-type": "application/x-www-form-urlencoded"}
    )
    assert r.status_code == 400 and r.json()["error"] == "invalid_request"


async def test_other_jwt_types_from_the_same_issuer_are_rejected(env: Env, http: httpx.AsyncClient) -> None:
    """Same key, same issuer, right audience, but not an access token (e.g. a future ID token)."""
    protect(env, MCP, ["notes:read"], required_scopes=["notes:read"])
    real = (await user_tokens(env))["access_token"]
    claims = jwt.decode(real, options={"verify_signature": False})
    id_token_like = env.server.key.sign(claims, typ="JWT")
    r = await http.get(MCP, headers={"Authorization": f"Bearer {id_token_like}"})
    assert r.status_code == 401 and "not an access token" in r.json()["error_description"]
