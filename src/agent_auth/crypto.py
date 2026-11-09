"""Keys, signing and the small hashing helpers shared by the server and the guard."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import time
from pathlib import Path
from typing import Any

import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
from jwt.algorithms import ECAlgorithm

ACCESS_TOKEN_TYP = "at+jwt"  # RFC 9068
SIGNING_ALG = "ES256"
JWT_BEARER = "urn:ietf:params:oauth:client-assertion-type:jwt-bearer"


def b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def pkce_s256(verifier: str) -> str:
    return b64url(hashlib.sha256(verifier.encode()).digest())


def random_token(nbytes: int = 32) -> str:
    return secrets.token_urlsafe(nbytes)


def sha256_hex(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def jwk_thumbprint(jwk: dict[str, Any]) -> str:
    """RFC 7638 thumbprint, used as the key id."""
    if jwk["kty"] == "EC":
        members = {k: jwk[k] for k in ("crv", "kty", "x", "y")}
    elif jwk["kty"] == "RSA":
        members = {k: jwk[k] for k in ("e", "kty", "n")}
    elif jwk["kty"] == "OKP":
        members = {k: jwk[k] for k in ("crv", "kty", "x")}
    else:  # pragma: no cover - only the types above are accepted anywhere
        raise ValueError(f"unsupported key type {jwk['kty']}")
    canonical = json.dumps(members, separators=(",", ":"), sort_keys=True)
    return b64url(hashlib.sha256(canonical.encode()).digest())


class SigningKey:
    """An ES256 key. The private part never leaves this object."""

    def __init__(self, private_key: ec.EllipticCurvePrivateKey) -> None:
        if not isinstance(private_key.curve, ec.SECP256R1):
            raise ValueError("signing key must be P-256")
        self._key = private_key
        public = ECAlgorithm.to_jwk(private_key.public_key(), as_dict=True)
        self.kid = jwk_thumbprint(public)
        self.public_jwk: dict[str, Any] = {**public, "kid": self.kid, "use": "sig", "alg": SIGNING_ALG}

    @classmethod
    def generate(cls) -> SigningKey:
        return cls(ec.generate_private_key(ec.SECP256R1()))

    @classmethod
    def from_pem(cls, pem: bytes) -> SigningKey:
        key = serialization.load_pem_private_key(pem, password=None)
        if not isinstance(key, ec.EllipticCurvePrivateKey):
            raise ValueError("signing key must be an EC private key")
        return cls(key)

    @classmethod
    def load_or_create(cls, path: Path) -> SigningKey:
        """Reads a PEM key, creating it (mode 0600) the first time."""
        if path.exists():
            return cls.from_pem(path.read_bytes())
        key = cls.generate()
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as fh:
            fh.write(key.private_pem())
        return key

    def private_pem(self) -> bytes:
        return self._key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )

    def sign(self, claims: dict[str, Any], typ: str) -> str:
        return jwt.encode(claims, self._key, algorithm=SIGNING_ALG, headers={"kid": self.kid, "typ": typ})

    def jwks(self) -> dict[str, Any]:
        return {"keys": [self.public_jwk]}


def client_assertion(client_id: str, key: SigningKey, audience: str, lifetime: int = 60) -> str:
    """A private_key_jwt client assertion (RFC 7523), for confidential clients."""
    now = int(time.time())
    claims = {
        "iss": client_id,
        "sub": client_id,
        "aud": audience,
        "iat": now,
        "exp": now + lifetime,
        "jti": random_token(16),
    }
    return key.sign(claims, typ="JWT")
