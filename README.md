# agent-auth

![Python](https://img.shields.io/badge/python-3.12%2B-blue)
![MCP](https://img.shields.io/badge/MCP-2026--07--28-black)
![License](https://img.shields.io/badge/license-MIT-green)

OAuth 2.1 for AI agents and the systems they call. It has three parts:

- a small authorization server that follows the [MCP 2026-07-28 authorization spec](https://modelcontextprotocol.io/specification/2026-07-28/basic/authorization);
- an ASGI guard that protects MCP servers and APIs;
- token exchange, so every downstream call says which agent acted for which user.

> 🇧🇷 Autorização para agentes de IA e para conexão entre sistemas: um servidor OAuth 2.1 leve, um
> middleware que protege servidores MCP e APIs, troca de token com cadeia de atores, consentimento
> revogável e log de auditoria encadeado por hash.

## Why

An agent that calls tools for a user raises questions an API key can't answer:

- Who is the user?
- Which agent is acting for them, and through which services?
- What did the user allow it to do?
- Can the user take that back right now?

The MCP spec answers part of this with OAuth 2.1, but leaves the authorization server to you. This
project is that server plus the resource side, kept small enough to read in an afternoon and run
next to other services on a 2 vCPU box: one process, SQLite, about 70 MB of RAM.

- **Spec-shaped.** The server implements:
  - PKCE (S256 only), exact redirect URIs and `iss` in the response (RFC 9207);
  - `resource` → `aud` binding (RFC 8707), JWT access tokens (RFC 9068) with a 5-minute lifetime;
  - Client ID Metadata Documents, and Dynamic Client Registration (deprecated, throttled);
  - Protected Resource Metadata (RFC 9728) and `insufficient_scope` step-up.
- **Agents acting for users, on the record.** An MCP server never forwards the token it received. It
  exchanges it (RFC 8693) for a token that:
  - targets the downstream API;
  - carries the same or fewer scopes;
  - lives no longer than the original;
  - names the chain in `act`: `notes-mcp <- the agent <- alice`.
- **Machines as themselves.** M2M jobs use `client_credentials` with `private_key_jwt` (RFC 7523).
  There are no shared secrets anywhere. A user-only API rejects machine tokens even when the scope
  matches.
- **Revocable.**
  - Users see and revoke consents. Revoking one cascades to refresh tokens and to every token
    exchanged from them.
  - Resource servers poll a small revocation list every few seconds and fail closed when it goes
    stale.
- **Auditable.** Every grant, token, exchange, rejection and tool call goes into an append-only log.
  Each entry hashes the previous one, so `agent-auth audit verify` detects any edit.

## Protect an MCP server

```python
from agent_auth import ProtectedResource, ServiceClient, get_principal

app = ProtectedResource(
    mcp.streamable_http_app(stateless_http=True, json_response=True),
    resource="https://notes.example.com/mcp",
    issuer="https://auth.example.com",
    scopes_supported=["notes:read", "notes:write"],
    required_scopes=["notes:read"],
    tool_scopes={"add_note": "notes:write"},  # per-tool scopes from the JSON-RPC body
)

# inside a tool: who is calling, and a token for the next hop (never the incoming one)
principal = get_principal(ctx.request_context.request)
api_token = await service.exchange(principal, "https://api.example.com", scope="notes:write")
```

Requests without a token get `401` with `WWW-Authenticate: Bearer resource_metadata=..., scope=...`.
The official MCP SDK client then takes care of discovery, registration, PKCE and step-up by itself.

## Demo

```bash
uv sync
uv pip install git+https://github.com/vicgamelo/mcp-server-brasil   # optional: protect a real MCP server too
uv run --no-sync agent-auth-demo
```

It starts the authorization server, a Notes MCP server, a Notes API and, if `mcp-server-brasil` is
installed, the Brasil MCP behind the guard. Then it runs this scenario with the official MCP Python
SDK client:

1. The agent connects with no token. It gets a 401, discovers the authorization server, registers,
   and sends Alice through login and consent for `notes:read`.
2. `add_note` answers `403 insufficient_scope`. The agent steps up to `notes:write`, and the MCP
   server exchanges the token to call the Notes API.
3. A nightly job with `client_credentials` calls the Brasil MCP. Trying the user-only Notes API, it
   gets `403 user_required`.
4. Alice revokes the agent. Its token gets `401 token revoked` within a second, and re-authorization
   is denied.
5. Alice's activity log prints with the hash chain verified.

Add `--keep-running` to look around in a browser: `http://127.0.0.1:9400/consents`, user `alice`,
password `alice-demo-password`.

## Attacks covered by the tests

`tests/test_attacks.py` and `tests/test_cimd.py` try to break it, and every attempt must fail.
Every protection has been mutated, one at a time, to check that some test notices.

- **Redirects:** a swapped `redirect_uri`, at authorize and at token, including look-alike hosts,
  paths, queries and fragments. An unknown URI gets an error page, never a redirect.
- **PKCE:** missing, `plain`, malformed or wrong. A wrong verifier burns the code.
- **Audience:** a token for another audience. Also forged tokens: `alg: none`, another key, HS256
  with the public key, an edited payload, other JWT types from the same issuer.
- **Scope:** widened through token exchange, through refresh or through `client_credentials`.
- **Replay:**
  - a replayed authorization code (tokens from the first use are revoked) and concurrent
    redemption;
  - a reused refresh token (the family is revoked);
  - a replayed client assertion.
- **Revocation:** revoked tokens, consents and cascades. A stale revocation list fails closed.
- **Registration abuse:** a DCR flood, a spoofed `X-Forwarded-For`, the client cap, bad metadata.
- **Confused deputy:**
  - a look-alike client riding someone else's consent;
  - a service cashing in a token sent to another service;
  - exchange into a resource that was not delegated;
  - `actor_token` splicing;
  - a machine token on user data;
  - a public client trying token exchange.
- **CSRF and clickjacking:** cross-site consent POSTs, framing, and Bob approving Alice's request.
- **CIMD SSRF:**
  - loopback, private, link-local and IPv4-mapped addresses;
  - DNS answers that mix in a private IP;
  - redirects to metadata endpoints, oversized or slow responses, documents that lie about their
    `client_id`.

## Run it

```bash
agent-auth keygen --out notes-mcp.pem        # prints the public JWK for a confidential client
AGENT_AUTH_DEV_USERS=alice:change-me agent-auth serve --config examples/agent-auth.toml
agent-auth audit verify --database agent-auth.db
```

The only identity provider shipped today is a development one with a fixed user list.
`IdentityProvider` is a two-method protocol: plug in your real login (an upstream OIDC provider, or
an existing session) before exposing it.

## Development

```bash
uv sync
uv run pytest          # ASGI in-process tests + one end-to-end test over real sockets
uv run ruff check . && uv run ruff format --check . && uv run mypy
```

```
src/agent_auth/
├── server.py      # authorization server endpoints
├── guard.py       # ProtectedResource, TokenVerifier, ServiceClient (exchange, M2M, audit reports)
├── clients.py     # static / DCR / CIMD clients, private_key_jwt authentication
├── cimd.py        # Client ID Metadata Document fetcher with SSRF guards
├── store.py       # SQLite: hashed codes and refresh tokens, grants, hash-chained audit log
├── config.py      # resources, clients, limits (code or TOML)
└── demo/          # Notes MCP, Notes API, Brasil MCP and the scripted agent
```

## Limits

- Bearer tokens, not sender-constrained (DPoP is on the roadmap).
- A single signing key with no rotation yet.
- One process, because rate limits and the login lockout live in memory.
- The revocation list is a non-standard endpoint (`revoked_tokens_endpoint` in the metadata). It
  is small because tokens last 5 minutes.
- Resource servers that report audit events are trusted to report them honestly.
- Not independently audited.

## Roadmap

- [ ] Real identity providers (upstream OIDC, signed assertion from an existing portal)
- [ ] DPoP (RFC 9449) for sender-constrained tokens
- [ ] Key rotation with overlapping JWKS
- [ ] Enterprise-Managed Authorization (ID-JAG) extension
- [ ] Rich Authorization Requests (RFC 9396) for per-tool consent

## License

MIT © Victor Gabriel Andrade de Melo
