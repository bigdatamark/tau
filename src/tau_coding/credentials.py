# [MODULE DOC]
# Local credential storage for provider credentials, kept in a JSON file
# (credentials.json under Tau home) with owner-only permissions (0600). Each
# named entry is either a bare string (legacy API key), an api_key object, or a
# refreshable oauth object (access/refresh tokens, expiry, optional account id
# and metadata). Provides typed credential records, a store protocol for setup
# integrations, and the file-backed implementation with atomic writes.
"""Local credential storage for Tau provider credentials."""

from __future__ import annotations

from dataclasses import dataclass, field
from json import dumps, loads
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Literal, Protocol

from tau_agent.types import JSONValue
from tau_coding.paths import TauPaths


class CredentialStoreError(ValueError):
    # [CLASS DOC]
    # Error (a ValueError) raised for invalid credential names/values or malformed
    # credentials file contents.
    """Raised when Tau credential storage cannot be read or written."""


@dataclass(frozen=True, slots=True)
class OAuthCredential:
    # [CLASS DOC]
    # Immutable OAuth credential: access token, refresh token, expiry (integer,
    # typically epoch milliseconds), optional account id, and a mapping of
    # non-secret provider-specific JSON metadata (default empty).
    """Refreshable OAuth credential persisted under Tau home.

    ``account_id`` remains optional so legacy OpenAI Codex credentials load
    unchanged while device-code providers can persist only the metadata they
    actually receive. Provider-specific, non-secret values live in ``metadata``.
    """

    access: str
    refresh: str
    expires: int
    account_id: str | None = None
    metadata: dict[str, JSONValue] = field(default_factory=dict)

    def to_json(self) -> dict[str, JSONValue]:
        # [FUNCTION DOC]
        # Serialize to a JSON-compatible mapping with type "oauth", access, refresh and
        # expires; account_id is included only when set and metadata (shallow-copied)
        # only when non-empty. No side effects.
        """Serialize this OAuth credential to JSON-compatible data."""
        result: dict[str, JSONValue] = {
            "type": "oauth",
            "access": self.access,
            "refresh": self.refresh,
            "expires": self.expires,
        }
        if self.account_id is not None:
            result["account_id"] = self.account_id
        if self.metadata:
            result["metadata"] = dict(self.metadata)
        return result


@dataclass(frozen=True, slots=True)
class ApiKeyCredential:
    # [CLASS DOC]
    # Immutable API-key credential holding a single key string.
    """API-key credential persisted under Tau home."""

    key: str

    def to_json(self) -> dict[str, JSONValue]:
        # [FUNCTION DOC]
        # Serialize to a mapping with type "api_key" and the key. No side effects.
        """Serialize this API key credential to JSON-compatible data."""
        return {"type": "api_key", "key": self.key}


type StoredCredential = str | ApiKeyCredential | OAuthCredential
type StoredCredentialKind = Literal["api_key", "oauth"]


class CredentialStore(Protocol):
    # [CLASS DOC]
    # Structural protocol for mutable credential stores used by setup integrations:
    # get a string credential by name, set one, delete one, and list names
    # optionally filtered by prefix.
    """Mutable credential-store operations used by setup integrations."""

    def get(self, name: str) -> str | None:
        # [FUNCTION DOC]
        # Protocol: return the API-key value for a name, or None.
        ...

    def set(self, name: str, value: str) -> None:
        # [FUNCTION DOC]
        # Protocol: store an API-key value under a name.
        ...

    def delete(self, name: str) -> None:
        # [FUNCTION DOC]
        # Protocol: remove a named credential.
        ...

    def names(self, *, prefix: str | None = None) -> tuple[str, ...]:
        # [FUNCTION DOC]
        # Protocol: return stored names (keyword prefix filter optional).
        ...


class FileCredentialStore:
    # [CLASS DOC]
    # JSON-file-backed credential store. Every operation reloads the whole file and
    # every mutation rewrites it atomically; there is no in-memory caching or
    # locking (concurrent writers may lose updates).
    """Small JSON-backed provider credential store under Tau home."""

    def __init__(self, path: Path | None = None) -> None:
        # [FUNCTION DOC]
        # Store the file path, defaulting to the standard credentials path under the
        # default Tau home. No I/O.
        self.path = path or credentials_path()

    def get(self, name: str) -> str | None:
        # [FUNCTION DOC]
        # Return the API-key value for a name: a bare string entry or an api_key
        # entry's key; returns None for missing names or OAuth entries. Reads the file;
        # raises CredentialStoreError or a JSON decode error if the file is malformed.
        # KNOWN ISSUE: the name is not stripped here, while set strips it, so lookups
        # with surrounding whitespace miss entries stored via set.
        """Return a stored API-key credential value by name."""
        credential = self._load().get(name)
        if isinstance(credential, str):
            return credential
        if isinstance(credential, ApiKeyCredential):
            return credential.key
        return None

    def set(self, name: str, value: str) -> None:
        # [FUNCTION DOC]
        # Store an API key: name and value are stripped; raises CredentialStoreError if
        # either is empty. The value is saved as a bare string entry (legacy format),
        # replacing any existing entry including an OAuth one. Rewrites the file.
        """Store an API-key credential value by name."""
        name = _validate_credential_name(name)
        value = value.strip()
        if not value:
            raise CredentialStoreError("Credential value must not be empty")
        data = self._load()
        data[name] = value
        self._save(data)

    def set_api_key(self, name: str, value: str) -> None:
        # [FUNCTION DOC]
        # Alias of set; stores an API key under a name with the same validation and
        # side effects.
        """Store an API-key credential value by name."""
        self.set(name, value)

    def get_oauth(self, name: str) -> OAuthCredential | None:
        # [FUNCTION DOC]
        # Return the OAuth credential stored under a name, or None if missing or not an
        # OAuth entry. Reads the file; raises on malformed files as for get.
        """Return a stored OAuth credential by name."""
        credential = self._load().get(name)
        if isinstance(credential, OAuthCredential):
            return credential
        return None

    def set_oauth(self, name: str, credential: OAuthCredential) -> None:
        # [FUNCTION DOC]
        # Store an OAuth credential under a stripped name after validating it (non-empty
        # access/refresh tokens, non-blank account id if given, positive expiry, string
        # metadata keys with JSON values). Raises CredentialStoreError on invalid input.
        # Rewrites the file.
        """Store a refreshable OAuth credential by name."""
        name = _validate_credential_name(name)
        _validate_oauth_credential(credential)
        data = self._load()
        data[name] = credential
        self._save(data)

    def delete(self, name: str) -> None:
        # [FUNCTION DOC]
        # Remove a named credential if present (no error if absent) and rewrite the
        # file (the file is written even when nothing changed, creating it if missing).
        # Name is not stripped.
        """Delete a stored credential value by name."""
        data = self._load()
        data.pop(name, None)
        self._save(data)

    def names(self, *, prefix: str | None = None) -> tuple[str, ...]:
        # [FUNCTION DOC]
        # Return the sorted tuple of stored credential names, optionally only those
        # starting with the keyword prefix. Values are never exposed. Reads the file.
        """Return credential names without exposing their values."""
        names = tuple(sorted(self._load()))
        if prefix is None:
            return names
        return tuple(name for name in names if name.startswith(prefix))

    def _load(self) -> dict[str, StoredCredential]:
        # [FUNCTION DOC]
        # HELPER: Read and parse the credentials file into a mapping from name to stored
        # credential. Returns an empty mapping if the file does not exist. Raises
        # CredentialStoreError if the top level is not an object or a value is invalid,
        # and a JSON decode error (a ValueError) for invalid JSON; OSError on read errors.
        if not self.path.exists():
            return {}
        raw = loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise CredentialStoreError("Tau credentials must be a JSON object")
        credentials: dict[str, StoredCredential] = {}
        for key, value in raw.items():
            if not isinstance(key, str):
                raise CredentialStoreError("Tau credential names must be strings")
            credentials[key] = _credential_from_json(value)
        return credentials

    def _save(self, data: dict[str, StoredCredential]) -> None:
        # [FUNCTION DOC]
        # HELPER: Atomically persist the mapping: ensures the parent dir exists,
        # serializes each credential (indent 2, sorted keys, trailing newline), writes to
        # a hidden temp file in the same directory created with mode 0600, renames it
        # over the target and re-applies 0600. The temp file is removed if it still
        # exists afterwards (i.e. on failure). Raises OSError on filesystem errors.
        self.path.parent.mkdir(parents=True, exist_ok=True)
        raw = {key: _credential_to_json(value) for key, value in data.items()}
        content = dumps(raw, indent=2, sort_keys=True) + "\n"
        temporary_path: Path | None = None
        try:
            with NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=self.path.parent,
                prefix=f".{self.path.name}.",
                delete=False,
            ) as handle:
                temporary_path = Path(handle.name)
                temporary_path.chmod(0o600)
                handle.write(content)
                handle.flush()
            temporary_path.replace(self.path)
            self.path.chmod(0o600)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()


def credentials_path(paths: TauPaths | None = None) -> Path:
    # [FUNCTION DOC]
    # Return the credentials.json path inside the Tau home of the given path layout
    # (or a default layout). No I/O.
    """Return Tau's local provider credential path."""
    return (paths or TauPaths()).home / "credentials.json"


def _validate_credential_name(name: str) -> str:
    # [FUNCTION DOC]
    # HELPER: Strip a credential name and return it; raises CredentialStoreError if
    # it becomes empty.
    normalized = name.strip()
    if not normalized:
        raise CredentialStoreError("Credential name must not be empty")
    return normalized


def _validate_oauth_credential(credential: OAuthCredential) -> None:
    # [FUNCTION DOC]
    # HELPER: Validate an OAuth credential: non-blank access and refresh tokens,
    # account id either None or non-blank, expiry greater than 0, and valid metadata.
    # Raises CredentialStoreError on the first failure; returns nothing.
    if not credential.access.strip():
        raise CredentialStoreError("OAuth access token must not be empty")
    if not credential.refresh.strip():
        raise CredentialStoreError("OAuth refresh token must not be empty")
    if credential.account_id is not None and not credential.account_id.strip():
        raise CredentialStoreError("OAuth account id must not be empty")
    if credential.expires <= 0:
        raise CredentialStoreError("OAuth expiry must be greater than 0")
    _validate_oauth_metadata(credential.metadata)


def _credential_from_json(value: object) -> StoredCredential:
    # [FUNCTION DOC]
    # HELPER: Parse one stored JSON value into a credential. Strings are returned
    # as-is (legacy API key). Objects must have type "api_key" (requires a non-blank
    # key, returned stripped) or "oauth" (requires a positive non-boolean integer
    # expires, optional non-blank string account_id, object metadata with valid JSON
    # values, and non-blank access/refresh which are stripped). Raises
    # CredentialStoreError for any other shape. Note account_id is not stripped.
    if isinstance(value, str):
        return value
    if not isinstance(value, dict):
        raise CredentialStoreError("Tau credential values must be strings or objects")

    credential_type = value.get("type")
    if credential_type not in {"api_key", "oauth"}:
        raise CredentialStoreError("Tau credential object type must be api_key or oauth")
    if credential_type == "api_key":
        key = _string_field(value, "key", credential_type)
        return ApiKeyCredential(key=key)

    expires = value.get("expires")
    if not isinstance(expires, int) or isinstance(expires, bool) or expires <= 0:
        raise CredentialStoreError("Tau oauth credential expires must be a positive integer")
    account_id = value.get("account_id")
    if account_id is not None and (not isinstance(account_id, str) or not account_id.strip()):
        raise CredentialStoreError("Tau oauth credential account_id must be a non-empty string")
    metadata = value.get("metadata", {})
    if not isinstance(metadata, dict):
        raise CredentialStoreError("Tau oauth credential metadata must be an object")
    _validate_oauth_metadata(metadata)
    return OAuthCredential(
        access=_string_field(value, "access", credential_type),
        refresh=_string_field(value, "refresh", credential_type),
        expires=expires,
        account_id=account_id,
        metadata=dict(metadata),
    )


def _credential_to_json(value: StoredCredential) -> str | dict[str, JSONValue]:
    # [FUNCTION DOC]
    # HELPER: Convert a stored credential to its JSON form: strings unchanged,
    # credential records via their to_json method.
    if isinstance(value, str):
        return value
    return value.to_json()


def _validate_oauth_metadata(metadata: dict[Any, Any]) -> None:
    # [FUNCTION DOC]
    # HELPER: Ensure every metadata key is a non-blank string and every value is a
    # JSON-compatible value; raises CredentialStoreError otherwise.
    for key, value in metadata.items():
        if not isinstance(key, str) or not key.strip():
            raise CredentialStoreError("Tau oauth credential metadata keys must be strings")
        if not _is_json_value(value):
            raise CredentialStoreError("Tau oauth credential metadata values must be JSON values")


def _is_json_value(value: object) -> bool:
    # [FUNCTION DOC]
    # HELPER: Recursively check whether a value is JSON-compatible: None, string,
    # boolean, int, float, a list of such values, or a mapping with string keys and
    # such values. Returns a boolean. Does not reject NaN/infinity floats.
    if value is None or isinstance(value, str | bool | int | float):
        return True
    if isinstance(value, list):
        return all(_is_json_value(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and _is_json_value(item) for key, item in value.items())
    return False


def _string_field(
    value: dict[Any, Any],
    field_name: str,
    credential_type: StoredCredentialKind,
) -> str:
    # [FUNCTION DOC]
    # HELPER: Fetch a required string field from a credential object; raises
    # CredentialStoreError naming the credential type and field if it is missing,
    # non-string or blank. Returns the stripped string.
    field = value.get(field_name)
    if not isinstance(field, str) or not field.strip():
        raise CredentialStoreError(
            f"Tau {credential_type} credential field must be a non-empty string: {field_name}"
        )
    return field.strip()
