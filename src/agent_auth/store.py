"""SQLite persistence. Secrets (codes, refresh tokens, sessions) are stored only as SHA-256 hashes.

One connection behind a lock: the server runs as a single small process and every query here is a
point lookup, so this is simpler and cheaper than a pool.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS clients (
    client_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,              -- dcr | cimd
    name TEXT NOT NULL,
    redirect_uris TEXT NOT NULL,     -- json list
    grant_types TEXT NOT NULL,       -- json list
    created_at REAL NOT NULL,
    last_used_at REAL,
    registered_ip TEXT
);
CREATE TABLE IF NOT EXISTS sessions (
    id_hash TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    csrf TEXT NOT NULL,
    expires_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS pending (
    id TEXT PRIMARY KEY,
    session_hash TEXT NOT NULL,
    params TEXT NOT NULL,
    expires_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS grants (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    client_id TEXT NOT NULL,
    resource TEXT NOT NULL,
    scopes TEXT NOT NULL,            -- space separated
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    revoked_at REAL
);
CREATE INDEX IF NOT EXISTS grants_user ON grants(user_id);
CREATE TABLE IF NOT EXISTS codes (
    code_hash TEXT PRIMARY KEY,
    grant_id TEXT NOT NULL,
    client_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    redirect_uri TEXT NOT NULL,
    code_challenge TEXT NOT NULL,
    resource TEXT NOT NULL,
    scopes TEXT NOT NULL,
    expires_at REAL NOT NULL,
    used_at REAL
);
CREATE TABLE IF NOT EXISTS refresh_tokens (
    token_hash TEXT PRIMARY KEY,
    family TEXT NOT NULL,            -- code hash that started the chain
    grant_id TEXT NOT NULL,
    client_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    resource TEXT NOT NULL,
    scopes TEXT NOT NULL,
    expires_at REAL NOT NULL,
    rotated_at REAL,
    revoked_at REAL
);
CREATE TABLE IF NOT EXISTS access_tokens (
    jti TEXT PRIMARY KEY,
    grant_id TEXT,
    family TEXT,
    parent_jti TEXT,
    client_id TEXT NOT NULL,
    subject TEXT NOT NULL,
    resource TEXT NOT NULL,
    expires_at REAL NOT NULL,
    revoked_at REAL
);
CREATE INDEX IF NOT EXISTS at_grant ON access_tokens(grant_id);
CREATE INDEX IF NOT EXISTS at_family ON access_tokens(family);
CREATE INDEX IF NOT EXISTS at_parent ON access_tokens(parent_jti);
CREATE TABLE IF NOT EXISTS assertion_jtis (
    key TEXT PRIMARY KEY,
    expires_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS audit (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    event TEXT NOT NULL,
    user_id TEXT,
    client_id TEXT,
    resource TEXT,
    detail TEXT NOT NULL,
    prev_hash TEXT NOT NULL,
    hash TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS audit_user ON audit(user_id);
"""

GENESIS = "0" * 64


@dataclass(frozen=True)
class Grant:
    id: str
    user_id: str
    client_id: str
    resource: str
    scopes: frozenset[str]
    created_at: float
    updated_at: float
    revoked_at: float | None


@dataclass(frozen=True)
class AuditEvent:
    seq: int
    ts: float
    event: str
    user_id: str | None
    client_id: str | None
    resource: str | None
    detail: dict[str, Any]
    prev_hash: str
    hash: str


def _audit_hash(prev_hash: str, record: dict[str, Any]) -> str:
    body = json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256((prev_hash + body).encode()).hexdigest()


def _scopes(value: str) -> frozenset[str]:
    return frozenset(value.split())


def join_scopes(scopes: frozenset[str] | set[str]) -> str:
    return " ".join(sorted(scopes))


class Store:
    def __init__(self, path: str = ":memory:") -> None:
        self._db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock:
            if path != ":memory:":
                self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("PRAGMA foreign_keys=ON")
            self._db.executescript(SCHEMA)

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                yield self._db
            except BaseException:
                self._db.execute("ROLLBACK")
                raise
            self._db.execute("COMMIT")

    def _one(self, sql: str, *args: Any) -> sqlite3.Row | None:
        with self._lock:
            row: sqlite3.Row | None = self._db.execute(sql, args).fetchone()
            return row

    def _all(self, sql: str, *args: Any) -> list[sqlite3.Row]:
        with self._lock:
            return list(self._db.execute(sql, args).fetchall())

    def _exec(self, sql: str, *args: Any) -> int:
        with self._lock:
            return self._db.execute(sql, args).rowcount

    # ---- dynamic clients -------------------------------------------------------------------

    def add_client(
        self,
        client_id: str,
        kind: str,
        name: str,
        redirect_uris: list[str],
        grant_types: list[str],
        now: float,
        ip: str | None,
    ) -> None:
        self._exec(
            "INSERT OR REPLACE INTO clients VALUES (?,?,?,?,?,?,?,?)",
            client_id,
            kind,
            name,
            json.dumps(redirect_uris),
            json.dumps(grant_types),
            now,
            None,
            ip,
        )

    def get_client(self, client_id: str) -> sqlite3.Row | None:
        return self._one("SELECT * FROM clients WHERE client_id=?", client_id)

    def touch_client(self, client_id: str, now: float) -> None:
        self._exec("UPDATE clients SET last_used_at=? WHERE client_id=?", now, client_id)

    def count_dcr_clients(self) -> int:
        row = self._one("SELECT COUNT(*) AS n FROM clients WHERE kind='dcr'")
        return int(row["n"]) if row else 0

    def prune_dcr_clients(self, now: float, unused_ttl: int, idle_ttl: int) -> int:
        return self._exec(
            "DELETE FROM clients WHERE kind='dcr' AND ("
            "(last_used_at IS NULL AND created_at < ?) OR (last_used_at IS NOT NULL AND last_used_at < ?))",
            now - unused_ttl,
            now - idle_ttl,
        )

    # ---- sessions and pending authorization requests ---------------------------------------

    def add_session(self, id_hash: str, user_id: str, csrf: str, expires_at: float) -> None:
        self._exec("INSERT INTO sessions VALUES (?,?,?,?)", id_hash, user_id, csrf, expires_at)

    def get_session(self, id_hash: str, now: float) -> sqlite3.Row | None:
        return self._one("SELECT * FROM sessions WHERE id_hash=? AND expires_at>?", id_hash, now)

    def delete_session(self, id_hash: str) -> None:
        self._exec("DELETE FROM sessions WHERE id_hash=?", id_hash)
        self._exec("DELETE FROM pending WHERE session_hash=?", id_hash)

    def add_pending(self, pid: str, session_hash: str, params: dict[str, Any], expires_at: float) -> None:
        self._exec("INSERT INTO pending VALUES (?,?,?,?)", pid, session_hash, json.dumps(params), expires_at)

    def take_pending(self, pid: str, session_hash: str, now: float) -> dict[str, Any] | None:
        with self.tx() as db:
            row = db.execute(
                "SELECT params FROM pending WHERE id=? AND session_hash=? AND expires_at>?",
                (pid, session_hash, now),
            ).fetchone()
            db.execute("DELETE FROM pending WHERE id=?", (pid,))
        return json.loads(row["params"]) if row else None

    # ---- grants (consent) ------------------------------------------------------------------

    def _grant(self, row: sqlite3.Row) -> Grant:
        return Grant(
            id=row["id"],
            user_id=row["user_id"],
            client_id=row["client_id"],
            resource=row["resource"],
            scopes=_scopes(row["scopes"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            revoked_at=row["revoked_at"],
        )

    def active_grant(self, user_id: str, client_id: str, resource: str) -> Grant | None:
        row = self._one(
            "SELECT * FROM grants WHERE user_id=? AND client_id=? AND resource=? AND revoked_at IS NULL",
            user_id,
            client_id,
            resource,
        )
        return self._grant(row) if row else None

    def get_grant(self, grant_id: str) -> Grant | None:
        row = self._one("SELECT * FROM grants WHERE id=?", grant_id)
        return self._grant(row) if row else None

    def upsert_grant(
        self, grant_id: str, user_id: str, client_id: str, resource: str, scopes: frozenset[str], now: float
    ) -> Grant:
        existing = self.active_grant(user_id, client_id, resource)
        if existing:
            merged = existing.scopes | scopes
            self._exec(
                "UPDATE grants SET scopes=?, updated_at=? WHERE id=?", join_scopes(merged), now, existing.id
            )
            grant = self.get_grant(existing.id)
            assert grant is not None
            return grant
        self._exec(
            "INSERT INTO grants VALUES (?,?,?,?,?,?,?,NULL)",
            grant_id,
            user_id,
            client_id,
            resource,
            join_scopes(scopes),
            now,
            now,
        )
        grant = self.get_grant(grant_id)
        assert grant is not None
        return grant

    def grants_for_user(self, user_id: str) -> list[Grant]:
        rows = self._all(
            "SELECT * FROM grants WHERE user_id=? AND revoked_at IS NULL ORDER BY updated_at DESC", user_id
        )
        return [self._grant(r) for r in rows]

    def revoke_grant(self, grant_id: str, now: float) -> bool:
        with self.tx() as db:
            changed = db.execute(
                "UPDATE grants SET revoked_at=? WHERE id=? AND revoked_at IS NULL", (now, grant_id)
            ).rowcount
            db.execute(
                "UPDATE refresh_tokens SET revoked_at=? WHERE grant_id=? AND revoked_at IS NULL",
                (now, grant_id),
            )
            db.execute(
                "UPDATE access_tokens SET revoked_at=? WHERE grant_id=? AND revoked_at IS NULL",
                (now, grant_id),
            )
            db.execute("UPDATE codes SET used_at=COALESCE(used_at, ?) WHERE grant_id=?", (now, grant_id))
        return changed > 0

    # ---- authorization codes ---------------------------------------------------------------

    def add_code(self, code_hash: str, **fields: Any) -> None:
        self._exec(
            "INSERT INTO codes VALUES (?,?,?,?,?,?,?,?,?,NULL)",
            code_hash,
            fields["grant_id"],
            fields["client_id"],
            fields["user_id"],
            fields["redirect_uri"],
            fields["code_challenge"],
            fields["resource"],
            fields["scopes"],
            fields["expires_at"],
        )

    def get_code(self, code_hash: str) -> sqlite3.Row | None:
        return self._one("SELECT * FROM codes WHERE code_hash=?", code_hash)

    def mark_code_used(self, code_hash: str, now: float) -> bool:
        """Atomic: only one caller can redeem a code."""
        return (
            self._exec("UPDATE codes SET used_at=? WHERE code_hash=? AND used_at IS NULL", now, code_hash)
            == 1
        )

    def revoke_family(self, family: str, now: float) -> None:
        """Everything issued from one authorization code (used when a code or refresh token is replayed)."""
        with self.tx() as db:
            db.execute(
                "UPDATE refresh_tokens SET revoked_at=? WHERE family=? AND revoked_at IS NULL", (now, family)
            )
            db.execute(
                "UPDATE access_tokens SET revoked_at=? WHERE family=? AND revoked_at IS NULL", (now, family)
            )
            # Tokens exchanged from those tokens inherit the family.

    # ---- refresh tokens --------------------------------------------------------------------

    def add_refresh(self, token_hash: str, **f: Any) -> None:
        self._exec(
            "INSERT INTO refresh_tokens VALUES (?,?,?,?,?,?,?,?,NULL,NULL)",
            token_hash,
            f["family"],
            f["grant_id"],
            f["client_id"],
            f["user_id"],
            f["resource"],
            f["scopes"],
            f["expires_at"],
        )

    def get_refresh(self, token_hash: str) -> sqlite3.Row | None:
        return self._one("SELECT * FROM refresh_tokens WHERE token_hash=?", token_hash)

    def rotate_refresh(self, token_hash: str, now: float) -> bool:
        return (
            self._exec(
                "UPDATE refresh_tokens SET rotated_at=? "
                "WHERE token_hash=? AND rotated_at IS NULL AND revoked_at IS NULL",
                now,
                token_hash,
            )
            == 1
        )

    # ---- access tokens ---------------------------------------------------------------------

    def add_access(self, jti: str, **f: Any) -> None:
        self._exec(
            "INSERT INTO access_tokens VALUES (?,?,?,?,?,?,?,?,NULL)",
            jti,
            f.get("grant_id"),
            f.get("family"),
            f.get("parent_jti"),
            f["client_id"],
            f["subject"],
            f["resource"],
            f["expires_at"],
        )

    def get_access(self, jti: str) -> sqlite3.Row | None:
        return self._one("SELECT * FROM access_tokens WHERE jti=?", jti)

    def revoke_access(self, jti: str, now: float) -> None:
        """Revokes a token and every token exchanged from it, transitively."""
        with self.tx() as db:
            db.execute(
                """
                WITH RECURSIVE tree(jti) AS (
                    SELECT ? UNION SELECT a.jti FROM access_tokens a JOIN tree t ON a.parent_jti = t.jti
                )
                UPDATE access_tokens SET revoked_at=? WHERE jti IN tree AND revoked_at IS NULL
                """,
                (jti, now),
            )

    def revoked_unexpired(self, now: float) -> list[str]:
        rows = self._all(
            "SELECT jti FROM access_tokens WHERE revoked_at IS NOT NULL AND expires_at>? ORDER BY jti", now
        )
        return [r["jti"] for r in rows]

    def prune(self, now: float) -> None:
        with self.tx() as db:
            db.execute("DELETE FROM access_tokens WHERE expires_at<?", (now - 60,))
            db.execute("DELETE FROM codes WHERE expires_at<?", (now - 3600,))
            db.execute("DELETE FROM refresh_tokens WHERE expires_at<?", (now,))
            db.execute("DELETE FROM sessions WHERE expires_at<?", (now,))
            db.execute("DELETE FROM pending WHERE expires_at<?", (now,))
            db.execute("DELETE FROM assertion_jtis WHERE expires_at<?", (now,))

    # ---- client assertion replay -----------------------------------------------------------

    def use_assertion_jti(self, key: str, expires_at: float) -> bool:
        try:
            self._exec("INSERT INTO assertion_jtis VALUES (?,?)", key, expires_at)
        except sqlite3.IntegrityError:
            return False
        return True

    # ---- audit log -------------------------------------------------------------------------

    def audit(
        self,
        ts: float,
        event: str,
        *,
        user_id: str | None = None,
        client_id: str | None = None,
        resource: str | None = None,
        **detail: Any,
    ) -> None:
        with self.tx() as db:
            row = db.execute("SELECT hash FROM audit ORDER BY seq DESC LIMIT 1").fetchone()
            prev = row["hash"] if row else GENESIS
            record = {
                "ts": ts,
                "event": event,
                "user_id": user_id,
                "client_id": client_id,
                "resource": resource,
                "detail": detail,
            }
            digest = _audit_hash(prev, record)
            db.execute(
                "INSERT INTO audit (ts, event, user_id, client_id, resource, detail, prev_hash, hash) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (ts, event, user_id, client_id, resource, json.dumps(detail, sort_keys=True), prev, digest),
            )

    def _event(self, row: sqlite3.Row) -> AuditEvent:
        return AuditEvent(
            seq=row["seq"],
            ts=row["ts"],
            event=row["event"],
            user_id=row["user_id"],
            client_id=row["client_id"],
            resource=row["resource"],
            detail=json.loads(row["detail"]),
            prev_hash=row["prev_hash"],
            hash=row["hash"],
        )

    def audit_events(self, user_id: str | None = None, limit: int = 200) -> list[AuditEvent]:
        if user_id is None:
            rows = self._all("SELECT * FROM audit ORDER BY seq DESC LIMIT ?", limit)
        else:
            rows = self._all("SELECT * FROM audit WHERE user_id=? ORDER BY seq DESC LIMIT ?", user_id, limit)
        return [self._event(r) for r in reversed(rows)]

    def verify_audit(self) -> tuple[bool, int, str | None]:
        """Recomputes the hash chain. Returns (ok, events checked, first bad seq or head hash)."""
        prev = GENESIS
        count = 0
        for row in self._all("SELECT * FROM audit ORDER BY seq"):
            e = self._event(row)
            record = {
                "ts": e.ts,
                "event": e.event,
                "user_id": e.user_id,
                "client_id": e.client_id,
                "resource": e.resource,
                "detail": e.detail,
            }
            if e.prev_hash != prev or _audit_hash(prev, record) != e.hash:
                return False, count, f"seq {e.seq}"
            prev = e.hash
            count += 1
        return True, count, prev
