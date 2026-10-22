"""API-key authentication and scope-based authorization.

Authentication answers "who is calling": the key must match one of the configured keys, or the
answer is 401. Authorization answers "what may they do": a known client without the scope an
endpoint needs gets 403. The two are kept apart because the right reaction differs: a client
with a bad key should fix the key, a client with a good key should ask for more rights.
"""

import hashlib
import hmac
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field, SecretStr

Scope = Literal["chat", "admin"]


class ApiKeyConfig(BaseModel):
    """One configured key. A client may have several, which makes rotation painless:
    add the new key, move the client over, remove the old one."""

    client: str = Field(pattern=r"^[a-z0-9_-]{1,64}$")
    # Long enough to be unguessable; SecretStr keeps it out of reprs, logs and tracebacks
    key: SecretStr = Field(min_length=16)
    # chat: the model endpoints and one's own conversations. admin: usage, cost and cache
    # statistics, and reading any conversation.
    scopes: frozenset[Scope] = frozenset({"chat"})


@dataclass(frozen=True)
class Client:
    name: str
    scopes: frozenset[Scope]

    def has(self, scope: Scope) -> bool:
        return scope in self.scopes


class Authenticator:
    def __init__(self, keys: Sequence[ApiKeyConfig]) -> None:
        # Only digests are kept: the plain keys do not stay in memory after startup, so a
        # memory dump or a stray debug print of this object reveals nothing usable.
        self._entries = [
            (_digest(config.key.get_secret_value()), Client(config.client, config.scopes))
            for config in keys
        ]

    @property
    def configured(self) -> bool:
        return bool(self._entries)

    def authenticate(self, presented: str | None) -> Client | None:
        if not presented:
            return None
        digest = _digest(presented)
        found: Client | None = None
        for stored, client in self._entries:
            # Constant-time comparison, and no early exit: how long the check takes must not
            # reveal which prefix of a guess was right or which entry matched
            if hmac.compare_digest(stored, digest):
                found = client
        return found


def _digest(key: str) -> bytes:
    # Hashing first gives both sides of the comparison the same length, so even the length
    # of a configured key cannot be probed
    return hashlib.sha256(key.encode()).digest()
