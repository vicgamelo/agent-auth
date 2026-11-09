"""Command line.

agent-auth serve --config agent-auth.toml [--host 127.0.0.1] [--port 9400]
agent-auth keygen --out service-key.pem      # prints the public JWK to put in the config
agent-auth audit verify --database agent-auth.db
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .config import load_config
from .crypto import SigningKey
from .store import Store


def _dev_users() -> dict[str, str]:
    raw = os.environ.get("AGENT_AUTH_DEV_USERS", "")
    users = dict(pair.split(":", 1) for pair in raw.split(",") if ":" in pair)
    if not users:
        sys.exit("set AGENT_AUTH_DEV_USERS=user:password[,user:password] (development identity provider)")
    return users


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="agent-auth")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run the authorization server")
    serve.add_argument("--config", type=Path, required=True)
    serve.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    serve.add_argument("--port", type=int, default=int(os.environ.get("PORT", "9400")))

    keygen = sub.add_parser("keygen", help="create an ES256 key for a confidential client")
    keygen.add_argument("--out", type=Path, required=True)

    audit = sub.add_parser("audit", help="audit log tools")
    audit_sub = audit.add_subparsers(dest="audit_command", required=True)
    verify = audit_sub.add_parser("verify", help="recompute the hash chain")
    verify.add_argument("--database", default="agent-auth.db")

    args = parser.parse_args(argv)
    if args.command == "serve":
        import uvicorn

        from .identity import DevIdentityProvider
        from .server import AuthorizationServer

        config = load_config(args.config)
        app = AuthorizationServer(config, DevIdentityProvider(_dev_users()))
        print("agent-auth: using the development identity provider (fixed user list)", file=sys.stderr)
        uvicorn.run(
            app, host=args.host, port=args.port, proxy_headers=False, server_header=False, access_log=False
        )
    elif args.command == "keygen":
        if args.out.exists():
            sys.exit(f"{args.out} already exists")
        key = SigningKey.load_or_create(args.out)
        print(json.dumps(key.public_jwk, indent=2))
    elif args.command == "audit":
        ok, count, where = Store(args.database).verify_audit()
        if ok:
            print(f"ok: {count} events, head {where}")
        else:
            print(f"TAMPERED at {where} (after {count} good events)")
            sys.exit(1)


if __name__ == "__main__":
    main()
