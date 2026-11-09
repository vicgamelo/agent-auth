"""Who the user is. The server only needs a user id after a successful login.

``DevIdentityProvider`` checks a fixed user list and exists for local runs and tests. A real
deployment plugs in its own provider (an upstream OIDC login, or an existing portal session) by
implementing the same two methods.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from collections.abc import Mapping
from typing import Protocol


class IdentityProvider(Protocol):
    name: str

    def password_login(self) -> bool:
        """True when the login page should show a username/password form."""
        ...

    async def authenticate(self, form: Mapping[str, str]) -> str | None:
        """Returns the user id for valid credentials, else None."""
        ...


class DevIdentityProvider:
    name = "dev"

    def __init__(self, users: Mapping[str, str]) -> None:
        self._salt = secrets.token_bytes(16)
        self._users = {u: self._digest(p) for u, p in users.items()}

    def _digest(self, password: str) -> bytes:
        return hashlib.scrypt(password.encode(), salt=self._salt, n=2**14, r=8, p=1)

    def password_login(self) -> bool:
        return True

    async def authenticate(self, form: Mapping[str, str]) -> str | None:
        username = form.get("username", "")
        expected = self._users.get(username)
        # Hash even for unknown users so timing does not reveal who exists.
        supplied = self._digest(form.get("password", ""))
        if expected is None:
            return None
        return username if hmac.compare_digest(expected, supplied) else None
