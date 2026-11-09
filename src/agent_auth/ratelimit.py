"""Token buckets and client IP resolution (X-Forwarded-For only from trusted proxies)."""

from __future__ import annotations

import ipaddress
import time
from collections import OrderedDict
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from starlette.types import Scope

Network = ipaddress.IPv4Network | ipaddress.IPv6Network


def parse_networks(cidrs: Iterable[str]) -> tuple[Network, ...]:
    return tuple(ipaddress.ip_network(c.strip(), strict=False) for c in cidrs if c.strip())


def client_ip(scope: Scope, trusted: tuple[Network, ...]) -> str:
    peer = scope.get("client")
    ip = peer[0] if peer else "unknown"
    if not trusted:
        return ip

    def is_trusted(value: str) -> bool:
        try:
            addr = ipaddress.ip_address(value)
        except ValueError:
            return False
        return any(addr in net for net in trusted)

    if not is_trusted(ip):
        return ip
    forwarded = ""
    for name, value in scope.get("headers", []):
        if name == b"x-forwarded-for":
            forwarded = value.decode("latin-1")
    # Walk right to left, skipping our own proxies; the first untrusted hop is the client.
    for hop in reversed([h.strip() for h in forwarded.split(",") if h.strip()]):
        if not is_trusted(hop):
            return hop
    return ip


@dataclass
class _Bucket:
    tokens: float
    updated: float


class RateLimiter:
    """Per-key token bucket with a bounded number of keys (oldest evicted first)."""

    def __init__(
        self,
        per_minute: float,
        burst: int,
        max_keys: int = 10_000,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.rate = per_minute / 60.0
        self.burst = burst
        self.max_keys = max_keys
        self.clock = clock
        self._buckets: OrderedDict[str, _Bucket] = OrderedDict()

    def take(self, key: str) -> float:
        """Consumes one token. Returns 0 when allowed, else seconds until the next token."""
        now = self.clock()
        bucket = self._buckets.pop(key, None) or _Bucket(tokens=float(self.burst), updated=now)
        bucket.tokens = min(self.burst, bucket.tokens + (now - bucket.updated) * self.rate)
        bucket.updated = now
        self._buckets[key] = bucket
        while len(self._buckets) > self.max_keys:
            self._buckets.popitem(last=False)
        if bucket.tokens >= 1:
            bucket.tokens -= 1
            return 0.0
        return (1 - bucket.tokens) / self.rate if self.rate else 3600.0
