"""Redirect URI rules shared by DCR, CIMD and static clients."""

from __future__ import annotations

from urllib.parse import urlsplit

from .config import is_loopback_host

LOOPBACK_IPS = ("127.0.0.1", "[::1]", "::1")


def redirect_uri_problem(uri: str) -> str | None:
    """Why a redirect URI can't be registered, or None if it can.

    https anywhere, or http on loopback (native apps and CLIs, RFC 8252). No fragments, no
    credentials, no custom schemes: agents here are CLIs or web apps.
    """
    if not isinstance(uri, str) or len(uri) > 512:
        return "redirect_uri must be a string of at most 512 characters"
    parts = urlsplit(uri)
    if parts.scheme not in ("https", "http"):
        return "redirect_uri must use https (or http on loopback)"
    if parts.scheme == "http" and not is_loopback_host(parts.hostname):
        return "http redirect_uri is only allowed on loopback"
    if not parts.hostname:
        return "redirect_uri must be absolute"
    if parts.fragment or "#" in uri:
        return "redirect_uri must not contain a fragment"
    if parts.username or parts.password:
        return "redirect_uri must not contain credentials"
    return None


def redirect_uri_matches(requested: str, registered: tuple[str, ...] | list[str]) -> bool:
    """Exact string match, except that loopback IP redirects may use any port (RFC 8252 7.3)."""
    if requested in registered:
        return True
    req = urlsplit(requested)
    if req.scheme != "http" or req.hostname not in ("127.0.0.1", "::1", "localhost"):
        return False
    for candidate in registered:
        reg = urlsplit(candidate)
        if (
            reg.scheme == "http"
            and reg.hostname == req.hostname
            and reg.path == req.path
            and reg.query == req.query
            and not req.fragment
            and not req.username
            and not req.password
        ):
            return True
    return False
