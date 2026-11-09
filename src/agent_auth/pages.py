"""Server-rendered pages. Every value is escaped; no inline script or style (CSP default-src 'none')."""

from __future__ import annotations

import time
from collections.abc import Iterable
from html import escape
from urllib.parse import urlsplit

from .config import ResourceConfig, is_loopback_host
from .store import AuditEvent, Grant

CSS = """
:root{color-scheme:light dark;--fg:#1a1a1a;--bg:#fafafa;--muted:#666;--line:#ddd;--accent:#0b5fff;--warn:#b45309}
@media (prefers-color-scheme:dark){:root{--fg:#eee;--bg:#141414;--muted:#aaa;--line:#333;--accent:#6ea0ff;--warn:#f59e0b}}
body{font:16px/1.5 system-ui,sans-serif;color:var(--fg);background:var(--bg);margin:0;padding:24px 16px}
main{max-width:560px;margin:0 auto}
h1{font-size:1.35rem;margin:0 0 16px}
.card{border:1px solid var(--line);border-radius:10px;padding:16px;margin:12px 0}
.muted{color:var(--muted);font-size:.9rem}.warn{color:var(--warn)}
code{font-size:.85rem;word-break:break-all}
ul{padding-left:20px}
label{display:block;margin:8px 0 4px}
input[type=text],input[type=password]{width:100%;box-sizing:border-box;padding:8px;border:1px solid var(--line);border-radius:6px;background:transparent;color:inherit}
button{padding:8px 16px;border-radius:6px;border:1px solid var(--line);background:transparent;color:inherit;cursor:pointer;margin:8px 8px 0 0}
button.primary{background:var(--accent);border-color:var(--accent);color:#fff}
table{width:100%;border-collapse:collapse;font-size:.85rem}td,th{border-bottom:1px solid var(--line);padding:6px 4px;text-align:left;vertical-align:top}
"""


def page(title: str, body: str) -> str:
    return (
        "<!doctype html><html lang=en><head><meta charset=utf-8>"
        '<meta name=viewport content="width=device-width,initial-scale=1">'
        f"<title>{escape(title)}</title><link rel=stylesheet href=/static/agent-auth.css></head>"
        f"<body><main>{body}</main></body></html>"
    )


def error_page(title: str, message: str) -> str:
    return page(title, f"<h1>{escape(title)}</h1><div class=card><p>{escape(message)}</p></div>")


def login_page(next_url: str, csrf: str, error: str | None = None) -> str:
    err = f"<p class=warn>{escape(error)}</p>" if error else ""
    return page(
        "Sign in",
        f"""<h1>Sign in</h1>{err}<form class=card method=post action=/login>
<input type=hidden name=csrf value="{escape(csrf)}"><input type=hidden name=next value="{escape(next_url)}">
<label for=u>Username</label><input id=u type=text name=username autocomplete=username required>
<label for=p>Password</label><input id=p type=password name=password autocomplete=current-password required>
<button class=primary type=submit>Sign in</button></form>""",
    )


def consent_page(
    *,
    pending_id: str,
    csrf: str,
    user_id: str,
    client_name: str,
    client_id: str,
    verified: bool,
    redirect_uri: str,
    resource: ResourceConfig,
    scopes: Iterable[str],
    delegates: list[ResourceConfig],
) -> str:
    host = urlsplit(redirect_uri).hostname or "?"
    badge = "" if verified else "<p class=warn>This client registered itself and is not verified.</p>"
    local = (
        "<p class=warn>The code will be sent to an app on this computer. "
        "Only continue if you just started it.</p>"
        if is_loopback_host(host)
        else ""
    )
    scope_items = "".join(
        f"<li><code>{escape(s)}</code> — {escape(resource.scopes.get(s, ''))}</li>" for s in sorted(scopes)
    )
    delegation = ""
    if delegates:
        names = "".join(
            f"<li>{escape(d.name)} <span class=muted>{escape(d.uri)}</span></li>" for d in delegates
        )
        delegation = (
            f"<p>{escape(resource.name)} may call these services for you, with the same or fewer "
            f"permissions:</p><ul>{names}</ul>"
        )
    return page(
        "Authorize access",
        f"""<h1>{escape(client_name)} wants to act for you</h1>
<div class=card><p>Signed in as <strong>{escape(user_id)}</strong></p>{badge}
<p class=muted>Client: <code>{escape(client_id)}</code><br>Sends you back to: <strong>{escape(host)}</strong></p>{local}</div>
<div class=card><p>Access to <strong>{escape(resource.name)}</strong> <span class=muted>{escape(resource.uri)}</span>:</p>
<ul>{scope_items}</ul>{delegation}
<p class=muted>Tokens last a few minutes. You can revoke this at any time on the consents page.</p></div>
<form method=post action=/authorize/decision>
<input type=hidden name=csrf value="{escape(csrf)}"><input type=hidden name=request value="{escape(pending_id)}">
<button class=primary type=submit name=decision value=approve>Allow</button>
<button type=submit name=decision value=deny>Deny</button></form>""",
    )


def _when(ts: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(ts)) + "Z"


def consents_page(user_id: str, csrf: str, grants: list[tuple[Grant, str, str]]) -> str:
    """grants: (grant, client name, resource name)."""
    if not grants:
        items = "<div class=card><p>No agent has access right now.</p></div>"
    else:
        items = "".join(
            f"""<div class=card><p><strong>{escape(client)}</strong> → {escape(res)}</p>
<p class=muted><code>{escape(g.client_id)}</code><br>Scopes: {escape(" ".join(sorted(g.scopes)))}<br>
Since {_when(g.created_at)}</p>
<form method=post action="/consents/{escape(g.id)}/revoke"><input type=hidden name=csrf value="{escape(csrf)}">
<button type=submit>Revoke</button></form></div>"""
            for g, client, res in grants
        )
    return page(
        "Your consents",
        f"""<h1>Agents with access, {escape(user_id)}</h1>{items}
<p><a href=/activity>Activity log</a></p>
<form method=post action=/logout><input type=hidden name=csrf value="{escape(csrf)}">
<button type=submit>Sign out</button></form>""",
    )


def activity_page(user_id: str, events: list[AuditEvent]) -> str:
    rows = "".join(
        f"<tr><td>{_when(e.ts)}</td><td>{escape(e.event)}</td><td><code>{escape(e.client_id or '')}</code></td>"
        f"<td>{escape(_summary(e))}</td></tr>"
        for e in reversed(events)
    )
    return page(
        "Activity",
        f"""<h1>What agents did for {escape(user_id)}</h1>
<table><thead><tr><th>When</th><th>Event</th><th>Client</th><th>Details</th></tr></thead><tbody>{rows}</tbody></table>
<p><a href=/consents>Consents</a></p>""",
    )


def _summary(e: AuditEvent) -> str:
    d = e.detail
    parts = []
    if e.resource:
        parts.append(e.resource)
    for key in ("tool", "scope", "actors", "status", "reason", "grant_type"):
        if d.get(key):
            value = d[key]
            parts.append(f"{key}={' > '.join(value) if isinstance(value, list) else value}")
    return " · ".join(parts)
