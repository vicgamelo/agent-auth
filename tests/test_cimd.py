"""Client ID Metadata Documents: the happy path and SSRF attempts against the fetcher."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from conftest import CLI_REDIRECT, ISSUER, Pkce, authorize_params, get_code, login, make_env, token

from agent_auth.cimd import CIMDError, CIMDFetcher

pytestmark = pytest.mark.anyio

CLIENT_URL = "https://agent.example/oauth/client.json"


def doc(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "client_id": CLIENT_URL,
        "client_name": "Example Agent",
        "redirect_uris": [CLI_REDIRECT],
        "grant_types": ["authorization_code", "refresh_token"],
    }
    base.update(over)
    return base


def resolver_for(mapping: dict[str, list[str]]) -> Any:
    async def resolve(host: str, port: int) -> list[str]:
        return mapping.get(host, [])

    return resolve


class Recorder(httpx.AsyncBaseTransport):
    def __init__(self, response: httpx.Response | None = None, body: Any = None) -> None:
        self.requests: list[httpx.Request] = []
        self.response = response
        self.body = body if body is not None else doc()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.response is not None:
            return self.response
        return httpx.Response(200, json=self.body)


PUBLIC = {"agent.example": ["93.184.215.14"]}


async def test_cimd_client_completes_the_flow() -> None:
    transport = Recorder()
    env = make_env(cimd_transport=transport, cimd_resolver=resolver_for(PUBLIC))
    pkce = Pkce()
    async with env.browser() as browser:
        await login(browser)
        r = await browser.get("/authorize", params=authorize_params(CLIENT_URL, pkce=pkce))
        assert r.status_code == 200 and "Example Agent" in r.text and "not verified" not in r.text
        result = await get_code(browser, authorize_params(CLIENT_URL, pkce=pkce))
    async with env.http() as http:
        r = await token(
            http,
            {
                "grant_type": "authorization_code",
                "client_id": CLIENT_URL,
                "code": result["code"],
                "code_verifier": pkce.verifier,
                "redirect_uri": CLI_REDIRECT,
            },
        )
    assert r.status_code == 200, r.text
    # connected to the address that was checked, with the real name in Host and SNI
    req = transport.requests[0]
    assert req.url.host == "93.184.215.14"
    assert req.headers["host"] == "agent.example"
    assert req.extensions["sni_hostname"] == "agent.example"
    assert len(transport.requests) == 1  # cached


@pytest.mark.parametrize(
    ("client_id", "resolved", "message"),
    [
        ("http://agent.example/oauth/client.json", PUBLIC, "https"),
        ("https://agent.example/", PUBLIC, "path"),
        ("https://agent.example/a/../client.json", PUBLIC, "dot segments"),
        ("https://user:pw@agent.example/client.json", PUBLIC, "credentials"),
        ("https://127.0.0.1/client.json", {}, "non-public"),
        ("https://[::1]/client.json", {}, "non-public"),
        ("https://[::ffff:10.0.0.1]/client.json", {}, "non-public"),
        ("https://169.254.169.254/latest/meta-data/x", {}, "non-public"),
        ("https://metadata.internal/client.json", {"metadata.internal": ["10.0.0.5"]}, "non-public"),
        (
            "https://rebind.example/client.json",
            {"rebind.example": ["93.184.215.14", "192.168.1.1"]},
            "non-public",
        ),
        ("https://nowhere.example/client.json", {}, "resolve"),
        ("https://agent.example:99999/client.json", PUBLIC, "port"),
    ],
)
async def test_fetcher_refuses_dangerous_urls(
    client_id: str, resolved: dict[str, list[str]], message: str
) -> None:
    transport = Recorder()
    fetcher = CIMDFetcher(transport=transport, resolver=resolver_for(resolved))
    with pytest.raises(CIMDError, match=message):
        await fetcher.fetch(client_id)
    assert transport.requests == []  # nothing left the building


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (httpx.Response(302, headers={"location": "http://169.254.169.254/"}), "HTTP 302"),
        (httpx.Response(404), "HTTP 404"),
        (httpx.Response(200, content=b"x" * 6000), "too large"),
        (httpx.Response(200, content=b"not json"), "not JSON"),
        (httpx.Response(200, json=doc(client_id="https://someone-else.example/c.json")), "does not match"),
        (httpx.Response(200, json=doc(client_secret="s")), "client_secret"),
        (httpx.Response(200, json=doc(token_endpoint_auth_method="private_key_jwt")), "public"),
        (httpx.Response(200, json=doc(redirect_uris=["http://attacker.example/cb"])), "loopback"),
        (httpx.Response(200, json=doc(grant_types=["client_credentials"])), "authorization_code"),
        (httpx.Response(200, json=[1, 2]), "object"),
    ],
)
async def test_fetcher_refuses_bad_documents(response: httpx.Response, message: str) -> None:
    fetcher = CIMDFetcher(transport=Recorder(response), resolver=resolver_for(PUBLIC))
    with pytest.raises(CIMDError, match=message):
        await fetcher.fetch(CLIENT_URL)


async def test_refused_cimd_client_gets_an_error_page_not_a_redirect() -> None:
    env = make_env(cimd_transport=Recorder(), cimd_resolver=resolver_for({"agent.example": ["10.1.2.3"]}))
    async with env.browser() as browser:
        await login(browser)
        r = await browser.get("/authorize", params=authorize_params(CLIENT_URL))
    assert r.status_code == 400 and "location" not in r.headers
    assert "non-public" in r.text


async def test_slow_server_times_out() -> None:
    class Slow(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("slow", request=request)

    fetcher = CIMDFetcher(transport=Slow(), resolver=resolver_for(PUBLIC))
    with pytest.raises(CIMDError, match="could not be fetched"):
        await fetcher.fetch(CLIENT_URL)


async def test_metadata_advertises_cimd() -> None:
    env = make_env()
    async with env.http() as http:
        meta = (await http.get(f"{ISSUER}/.well-known/oauth-authorization-server")).json()
    assert meta["client_id_metadata_document_supported"] is True
    assert json.dumps(meta)  # serialisable
