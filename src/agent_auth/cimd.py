"""OAuth Client ID Metadata Documents: the client_id is an HTTPS URL that serves the client's metadata.

Fetching a URL chosen by an anonymous caller is an SSRF risk, so the fetcher:

- accepts only https URLs with a non-root path, no credentials, no fragment, no dot segments;
- resolves the host once, refuses private/loopback/link-local/reserved addresses, and connects to the
  address it checked (the hostname still goes in Host and SNI, so TLS is verified against it), which
  closes the DNS rebinding window between check and connect;
- does not follow redirects, caps the body size and the time spent, and caches results.
"""

from __future__ import annotations

import asyncio
import ipaddress
import json
import socket
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from urllib.parse import urlsplit

import httpx

from .config import is_loopback_host
from .redirects import redirect_uri_problem

Resolver = Callable[[str, int], Awaitable[list[str]]]

MAX_BYTES = 5 * 1024
TIMEOUT = 3.0


class CIMDError(ValueError):
    pass


@dataclass(frozen=True)
class ClientMetadata:
    client_id: str
    client_name: str
    redirect_uris: tuple[str, ...]
    grant_types: tuple[str, ...]


async def system_resolver(host: str, port: int) -> list[str]:
    infos = await asyncio.get_running_loop().getaddrinfo(host, port, type=socket.SOCK_STREAM)
    return list(dict.fromkeys(str(info[4][0]) for info in infos))


def looks_like_cimd(client_id: str) -> bool:
    return client_id.startswith("https://") or client_id.startswith("http://")


def _public(addr: str) -> bool:
    ip = ipaddress.ip_address(addr)
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
        ip = ip.ipv4_mapped
    return ip.is_global and not ip.is_multicast


class CIMDFetcher:
    def __init__(
        self,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        resolver: Resolver = system_resolver,
        allow_loopback: bool = False,
        cache_ttl: int = 300,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._transport = transport
        self._resolver = resolver
        self._allow_loopback = allow_loopback
        self._cache_ttl = cache_ttl
        self._clock = clock
        self._cache: dict[str, tuple[float, ClientMetadata]] = {}

    def check_url(self, client_id: str) -> tuple[str, int]:
        if len(client_id) > 512:
            raise CIMDError("client_id URL too long")
        parts = urlsplit(client_id)
        loopback_dev = self._allow_loopback and parts.scheme == "http" and is_loopback_host(parts.hostname)
        if parts.scheme != "https" and not loopback_dev:
            raise CIMDError("client_id URL must use https")
        if not parts.hostname or parts.username or parts.password:
            raise CIMDError("client_id URL must have a host and no credentials")
        if parts.fragment or "#" in client_id:
            raise CIMDError("client_id URL must not have a fragment")
        if parts.path in ("", "/"):
            raise CIMDError("client_id URL must have a path")
        if any(seg in (".", "..") for seg in parts.path.split("/")):
            raise CIMDError("client_id URL must not have dot segments")
        try:
            port = parts.port or (443 if parts.scheme == "https" else 80)
        except ValueError as exc:
            raise CIMDError("client_id URL has an invalid port") from exc
        return parts.hostname, port

    async def _pinned_address(self, host: str, port: int) -> str:
        try:
            literal = ipaddress.ip_address(host)
        except ValueError:
            literal = None
        if literal is not None:
            addresses = [str(literal)]
        else:
            try:
                addresses = await asyncio.wait_for(self._resolver(host, port), timeout=TIMEOUT)
            except (OSError, TimeoutError) as exc:
                raise CIMDError("client_id host does not resolve") from exc
        if not addresses:
            raise CIMDError("client_id host does not resolve")
        for addr in addresses:
            if not _public(addr) and not (self._allow_loopback and ipaddress.ip_address(addr).is_loopback):
                raise CIMDError("client_id host resolves to a non-public address")
        return addresses[0]

    async def fetch(self, client_id: str) -> ClientMetadata:
        cached = self._cache.get(client_id)
        now = self._clock()
        if cached and cached[0] > now:
            return cached[1]
        host, port = self.check_url(client_id)
        address = await self._pinned_address(host, port)
        parts = urlsplit(client_id)
        ip_host = f"[{address}]" if ":" in address else address
        pinned_url = f"{parts.scheme}://{ip_host}:{port}{parts.path}" + (
            f"?{parts.query}" if parts.query else ""
        )
        host_header = parts.netloc
        try:
            async with httpx.AsyncClient(
                transport=self._transport, follow_redirects=False, timeout=TIMEOUT, trust_env=False
            ) as http:
                request = http.build_request(
                    "GET",
                    pinned_url,
                    headers={"Host": host_header, "Accept": "application/json"},
                    extensions={"sni_hostname": host},
                )
                response = await http.send(request, stream=True)
                try:
                    if response.status_code != 200:
                        raise CIMDError(f"client metadata fetch returned HTTP {response.status_code}")
                    body = b""
                    async for chunk in response.aiter_bytes():
                        body += chunk
                        if len(body) > MAX_BYTES:
                            raise CIMDError("client metadata document too large")
                finally:
                    await response.aclose()
        except httpx.HTTPError as exc:
            raise CIMDError("client metadata could not be fetched") from exc
        metadata = self.parse(client_id, body)
        self._cache[client_id] = (now + self._cache_ttl, metadata)
        return metadata

    def parse(self, client_id: str, body: bytes) -> ClientMetadata:
        try:
            doc = json.loads(body)
        except ValueError as exc:
            raise CIMDError("client metadata is not JSON") from exc
        if not isinstance(doc, dict):
            raise CIMDError("client metadata must be a JSON object")
        if doc.get("client_id") != client_id:
            raise CIMDError("client metadata client_id does not match its URL")
        for forbidden in ("client_secret", "client_secret_expires_at"):
            if forbidden in doc:
                raise CIMDError(f"client metadata must not contain {forbidden}")
        method = doc.get("token_endpoint_auth_method", "none")
        if method != "none":
            raise CIMDError("only public CIMD clients (token_endpoint_auth_method=none) are supported")
        uris = doc.get("redirect_uris")
        if not isinstance(uris, list) or not uris or len(uris) > 5:
            raise CIMDError("client metadata needs 1 to 5 redirect_uris")
        for uri in uris:
            problem = redirect_uri_problem(uri)
            if problem:
                raise CIMDError(problem)
        grant_types = doc.get("grant_types", ["authorization_code"])
        if not isinstance(grant_types, list) or not set(grant_types) <= {
            "authorization_code",
            "refresh_token",
        }:
            raise CIMDError("CIMD clients may only use authorization_code and refresh_token")
        name = doc.get("client_name", "")
        if not isinstance(name, str) or len(name) > 100 or not name.isprintable():
            raise CIMDError("client_name must be printable text of at most 100 characters")
        return ClientMetadata(
            client_id=client_id,
            client_name=name or urlsplit(client_id).netloc,
            redirect_uris=tuple(uris),
            grant_types=tuple(grant_types),
        )
