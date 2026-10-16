# [MODULE DOC]
# User-home session index management for coding sessions (tau_coding layer;
# the portable harness knows nothing about these file locations).
# Sessions are JSONL transcripts stored per project under the Tau home's
# sessions directory; this module maintains JSONL metadata indexes: one
# index.jsonl per project session directory, a legacy global index.jsonl and a
# global archive.jsonl for sessions hidden from the resume picker.
# Defines id/name validation, the pydantic SessionRecordModel (on-disk row),
# the CodingSessionRecord dataclass, SessionManager and a dedup helper.
# Constants: max id length 128 UTF-8 bytes, reserved ids "default"/"index",
# Windows reserved device stems (aux, con, nul, prn, com1-9, lpt1-9), and the
# allowed id pattern (alphanumeric start/end; letters, digits, dot, dash,
# underscore inside). InferenceProviderMode is "automatic" or "fixed".
"""User-home session management for Tau coding sessions."""

from __future__ import annotations

import re
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict

from tau_coding.paths import TauPaths

_MAX_SESSION_ID_BYTES = 128
_RESERVED_SESSION_IDS = frozenset({"default", "index"})
_WINDOWS_RESERVED_FILE_STEMS = frozenset(
    {"aux", "con", "nul", "prn"}
    | {f"com{index}" for index in range(1, 10)}
    | {f"lpt{index}" for index in range(1, 10)}
)
_SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?$")

InferenceProviderMode = Literal["automatic", "fixed"]


def normalize_session_name(value: str) -> str:
    # [FUNCTION DOC]
    # Trims a user-supplied session name and validates it.
    # Outputs: the trimmed name. Raises ValueError if empty after trimming or if it
    # contains carriage return, newline or tab characters. Pure.
    """Return a trimmed, single-line session name or raise ValueError."""
    name = value.strip()
    if not name:
        raise ValueError("Session name cannot be empty")
    if any(char in name for char in "\r\n\t"):
        raise ValueError("Session name must be a single line.")
    return name


def validate_session_id(session_id: str) -> None:
    # [FUNCTION DOC]
    # Validates a custom session id is safe and portable as a file name.
    # Raises ValueError if it fails the id pattern, exceeds 128 UTF-8 bytes, equals
    # a reserved id (case-insensitive), or its part before the first dot is a
    # Windows reserved device name (case-insensitive). Returns nothing.
    """Reject custom session ids that are unsafe as file names."""
    if not _SESSION_ID_PATTERN.fullmatch(session_id):
        raise ValueError(
            "Session id must be non-empty, contain only alphanumeric characters, '-', '_', "
            "and '.', and start and end with an alphanumeric character"
        )
    if len(session_id.encode("utf-8")) > _MAX_SESSION_ID_BYTES:
        raise ValueError(f"Session id must be at most {_MAX_SESSION_ID_BYTES} bytes")
    normalized_id = session_id.casefold()
    if normalized_id in _RESERVED_SESSION_IDS:
        raise ValueError(f"Session id is reserved: {session_id}")
    if normalized_id.partition(".")[0] in _WINDOWS_RESERVED_FILE_STEMS:
        raise ValueError(f"Session id is not a portable file name: {session_id}")


class SessionRecordModel(BaseModel):
    # [CLASS DOC]
    # Pydantic model for one index row (extra fields ignored): id, transcript path
    # and cwd as strings, model id, optional provider name, optional inference
    # provider and its mode, optional title, creation and update timestamps
    # (seconds since epoch).
    """JSON-serializable coding-session metadata."""

    model_config = ConfigDict(extra="ignore")

    id: str
    path: str
    cwd: str
    model: str
    provider_name: str | None = None
    inference_provider: str | None = None
    inference_provider_mode: InferenceProviderMode | None = None
    title: str | None = None
    created_at: float
    updated_at: float


@dataclass(frozen=True, slots=True)
class CodingSessionRecord:
    # [CLASS DOC]
    # Frozen, slotted dataclass of session metadata with Path-typed path and cwd;
    # inference provider mode defaults to "automatic".
    """Metadata for one durable coding session."""

    id: str
    path: Path
    cwd: Path
    model: str
    title: str | None
    created_at: float
    updated_at: float
    provider_name: str | None = None
    inference_provider: str | None = None
    inference_provider_mode: InferenceProviderMode = "automatic"

    @classmethod
    def from_model(cls, model: SessionRecordModel) -> CodingSessionRecord:
        # [FUNCTION DOC]
        # Class method converting an index row model to a record, converting strings to
        # paths. When the stored mode is missing it is inferred: "fixed" if an
        # inference provider is set, else "automatic". Pure.
        """Convert a JSON model to a record."""
        return cls(
            id=model.id,
            path=Path(model.path),
            cwd=Path(model.cwd),
            model=model.model,
            title=model.title,
            created_at=model.created_at,
            updated_at=model.updated_at,
            provider_name=model.provider_name,
            inference_provider=model.inference_provider,
            inference_provider_mode=(
                model.inference_provider_mode
                or ("fixed" if model.inference_provider is not None else "automatic")
            ),
        )

    def to_model(self) -> SessionRecordModel:
        # [FUNCTION DOC]
        # Converts the record back into a SessionRecordModel with string paths. Pure.
        """Convert this record to a JSON model."""
        return SessionRecordModel(
            id=self.id,
            path=str(self.path),
            cwd=str(self.cwd),
            model=self.model,
            title=self.title,
            created_at=self.created_at,
            updated_at=self.updated_at,
            provider_name=self.provider_name,
            inference_provider=self.inference_provider,
            inference_provider_mode=self.inference_provider_mode,
        )


class SessionManager:
    # [CLASS DOC]
    # Creates, indexes, lists, archives, restores and resumes coding sessions
    # under the user's Tau home (paths default to a fresh TauPaths). All index
    # operations read and rewrite whole JSONL files (no locking).
    """Create, index, list, and resume user-home coding sessions."""

    def __init__(self, paths: TauPaths | None = None) -> None:
        # [FUNCTION DOC]
        # Stores the given TauPaths or constructs default TauPaths.
        self.paths = paths or TauPaths()

    @property
    def index_path(self) -> Path:
        # [FUNCTION DOC]
        # Property: path of the legacy global index (sessions dir / index.jsonl).
        """Return the legacy global session metadata index path."""
        return self.paths.sessions_dir / "index.jsonl"

    def project_index_path(self, cwd: Path) -> Path:
        # [FUNCTION DOC]
        # Returns the index.jsonl path inside the project's session directory for the
        # given cwd (as computed by TauPaths). No I/O.
        """Return the session metadata index path for a project cwd."""
        return self.paths.project_session_dir(cwd) / "index.jsonl"

    @property
    def archive_index_path(self) -> Path:
        # [FUNCTION DOC]
        # Property: path of the global archive index (sessions dir / archive.jsonl).
        """Return the metadata index for sessions hidden from the resume picker."""
        return self.paths.sessions_dir / "archive.jsonl"

    def list_sessions(self, cwd: Path | None = None) -> list[CodingSessionRecord]:
        # [FUNCTION DOC]
        # Returns indexed (non-archived) records sorted by updated time, newest first.
        # With a cwd, reads that project's index plus legacy-index rows with that
        # resolved cwd; otherwise aggregates the legacy index and every project index.
        # Records are deduplicated by id keeping the latest update.
        # Side effects: file reads. Raises on malformed index rows (validation error).
        """Return indexed sessions, newest updated first.

        When `cwd` is provided, only sessions for that resolved working directory
        are returned. Without `cwd`, records are aggregated across project
        indexes and the legacy global index.
        """
        records = self._read_project_records(cwd) if cwd is not None else self._read_all_records()
        return sorted(records, key=lambda record: record.updated_at, reverse=True)

    def get_session(self, session_id: str) -> CodingSessionRecord | None:
        # [FUNCTION DOC]
        # Returns the record with the given id from all active indexes, or none.
        """Return a session record by id, if present."""
        for record in self._read_all_records():
            if record.id == session_id:
                return record
        return None

    def latest_session_for_cwd(self, cwd: Path) -> CodingSessionRecord | None:
        # [FUNCTION DOC]
        # Returns the most recently updated session for the working directory, or
        # none when there are none.
        """Return the most recently updated session for a working directory."""
        records = self.list_sessions(cwd)
        return records[0] if records else None

    def list_archived_sessions(self, cwd: Path | None = None) -> list[CodingSessionRecord]:
        # [FUNCTION DOC]
        # Returns deduplicated archived records, optionally only those whose resolved
        # cwd equals the resolved given cwd, newest first. Reads the archive index.
        """Return archived sessions, optionally restricted to one project."""
        records = _deduplicate_records(self._read_index(self.archive_index_path))
        if cwd is not None:
            resolved_cwd = cwd.resolve()
            records = [record for record in records if record.cwd.resolve() == resolved_cwd]
        return sorted(records, key=lambda record: record.updated_at, reverse=True)

    def archive_session(self, session_id: str) -> bool:
        # [FUNCTION DOC]
        # Hides a session from the resume index without touching its transcript.
        # Finds all active records with the id; returns false if none. Otherwise adds
        # them to the archive (replacing archived entries with the same id), removes
        # them from their project index and from the legacy index, returns true.
        # Side effects: rewrites index files.
        """Move a session out of the resume index without touching its transcript."""
        matches = [record for record in self._read_all_records() if record.id == session_id]
        if not matches:
            return False
        self._add_to_archive(matches)
        for record in matches:
            self._remove(record)
            if self.index_path.exists():
                self._remove_from_index(self.index_path, session_id)
        return True

    def archive_project(self, cwd: Path) -> bool:
        # [FUNCTION DOC]
        # Archives every active session whose resolved cwd matches the given project.
        # Returns false if none; otherwise appends them to the archive, rewrites the
        # project index and the legacy index (if it exists) without that project's
        # rows, and returns true. Transcripts are untouched.
        """Move a project's sessions out of resume indexes without touching files."""
        resolved_cwd = cwd.resolve()
        records = [
            record for record in self._read_all_records() if record.cwd.resolve() == resolved_cwd
        ]
        if not records:
            return False
        self._add_to_archive(records)
        project_path = self.project_index_path(resolved_cwd)
        project_records = [
            record
            for record in self._read_index(project_path)
            if record.cwd.resolve() != resolved_cwd
        ]
        self._write_index(project_path, project_records)
        if self.index_path.exists():
            legacy_records = [
                record
                for record in self._read_index(self.index_path)
                if record.cwd.resolve() != resolved_cwd
            ]
            self._write_index(self.index_path, legacy_records)
        return True

    def unarchive_session(self, session_id: str) -> bool:
        # [FUNCTION DOC]
        # Restores archived records with the id into their project indexes (upsert)
        # and removes the id from the archive. Returns whether any matched.
        """Restore one archived session to the active resume index."""
        matches = [record for record in self.list_archived_sessions() if record.id == session_id]
        if not matches:
            return False
        for record in matches:
            self._upsert(record)
        self._remove_from_archive(session_id)
        return True

    def unarchive_project(self, cwd: Path) -> bool:
        # [FUNCTION DOC]
        # Restores all archived sessions of the resolved project cwd to their project
        # index and rewrites the archive without them. Returns whether any matched.
        """Restore all archived sessions for a project to the active index."""
        resolved_cwd = cwd.resolve()
        matches = [
            record
            for record in self.list_archived_sessions()
            if record.cwd.resolve() == resolved_cwd
        ]
        if not matches:
            return False
        for record in matches:
            self._upsert(record)
        archived = [
            record
            for record in self._read_index(self.archive_index_path)
            if record.cwd.resolve() != resolved_cwd
        ]
        self._write_index(self.archive_index_path, archived)
        return True

    def create_session(
        self,
        *,
        cwd: Path,
        model: str,
        provider_name: str | None = None,
        inference_provider: str | None = None,
        inference_provider_mode: InferenceProviderMode | None = None,
        title: str | None = None,
        session_id: str | None = None,
    ) -> CodingSessionRecord:
        # [FUNCTION DOC]
        # Prepares a new session record (see prepare_session) and indexes it.
        # Outputs: the record. Raises ValueError for invalid/reserved ids.
        # Side effects: creates the project session directory and writes the index;
        # does not create the transcript file.
        """Create and index a new session record."""
        record = self.prepare_session(
            cwd=cwd,
            model=model,
            provider_name=provider_name,
            inference_provider=inference_provider,
            inference_provider_mode=inference_provider_mode,
            title=title,
            session_id=session_id,
        )
        self.index_session(record)
        return record

    def create_session_exclusive(
        self,
        *,
        cwd: Path,
        model: str,
        provider_name: str | None = None,
        inference_provider: str | None = None,
        inference_provider_mode: InferenceProviderMode | None = None,
        title: str | None = None,
        session_id: str | None = None,
    ) -> CodingSessionRecord:
        # [FUNCTION DOC]
        # Like create_session but refuses to reuse an id: raises RuntimeError if the id
        # is already indexed or the transcript file already exists, creating the
        # transcript file exclusively (empty) as an atomic reservation. If indexing then
        # fails it removes the index entry and the file (best effort) and re-raises;
        # other file-creation errors also trigger best-effort cleanup and re-raise.
        # Outputs: the indexed record.
        """Atomically reserve and index a session transcript without overwriting."""
        record = self.prepare_session(
            cwd=cwd,
            model=model,
            provider_name=provider_name,
            inference_provider=inference_provider,
            inference_provider_mode=inference_provider_mode,
            title=title,
            session_id=session_id,
        )
        if self.get_session(record.id) is not None:
            raise RuntimeError(f"Session already exists with id '{record.id}'")
        try:
            with record.path.open("x", encoding="utf-8"):
                pass
        except FileExistsError as exc:
            raise RuntimeError(f"Session already exists with id '{record.id}'") from exc
        except Exception:
            with suppress(OSError):
                record.path.unlink(missing_ok=True)
            raise
        try:
            return self.index_session(record)
        except Exception:
            with suppress(Exception):
                self._remove(record)
            with suppress(OSError):
                record.path.unlink(missing_ok=True)
            raise

    def prepare_session(
        self,
        *,
        cwd: Path,
        model: str,
        provider_name: str | None = None,
        inference_provider: str | None = None,
        inference_provider_mode: InferenceProviderMode | None = None,
        title: str | None = None,
        session_id: str | None = None,
    ) -> CodingSessionRecord:
        # [FUNCTION DOC]
        # Builds metadata for a new session without indexing it.
        # Behavior: resolves cwd; uses the given id or a random hex UUID; validates the
        # id; rejects (ValueError) an id equal case-insensitively to the project's
        # "default-<dir name>" id; transcript path = project session dir / id.jsonl;
        # creates that directory; mode defaults to "fixed" when an inference provider
        # is given, else "automatic"; created and updated times are now.
        # Outputs: the record. Side effects: directory creation.
        """Return metadata for a session without adding it to the resume index."""
        now = time()
        resolved_cwd = cwd.resolve()
        record_id = uuid4().hex if session_id is None else session_id
        validate_session_id(record_id)
        project_session_dir = self.paths.project_session_dir(resolved_cwd)
        default_session_id = f"default-{project_session_dir.name}"
        if record_id.casefold() == default_session_id.casefold():
            raise ValueError(f"Session id is reserved: {record_id}")
        path = project_session_dir / f"{record_id}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        return CodingSessionRecord(
            id=record_id,
            path=path,
            cwd=resolved_cwd,
            model=model,
            provider_name=provider_name,
            inference_provider=inference_provider,
            inference_provider_mode=(
                inference_provider_mode
                or ("fixed" if inference_provider is not None else "automatic")
            ),
            title=title,
            created_at=now,
            updated_at=now,
        )

    def index_session(self, record: CodingSessionRecord) -> CodingSessionRecord:
        # [FUNCTION DOC]
        # Upserts a prepared record into its project index and returns it.
        """Add a prepared session record to the resume index."""
        self._upsert(record)
        return record

    def get_or_create_default_session(
        self, *, cwd: Path, model: str, provider_name: str | None = None
    ) -> CodingSessionRecord:
        # [FUNCTION DOC]
        # Returns the project's default session record ("default-<project dir
        # name>"), creating and indexing one titled "Default session" at TauPaths'
        # default session path when it is not indexed. Does not create the transcript.
        """Return the default project session, creating an index record when needed."""
        resolved_cwd = cwd.resolve()
        project_hash = self.paths.project_session_dir(resolved_cwd).name
        session_id = f"default-{project_hash}"
        existing = self.get_session(session_id)
        if existing is not None:
            return existing

        now = time()
        path = self.paths.default_session_path(resolved_cwd)
        record = CodingSessionRecord(
            id=session_id,
            path=path,
            cwd=resolved_cwd,
            model=model,
            provider_name=provider_name,
            title="Default session",
            created_at=now,
            updated_at=now,
        )
        self._upsert(record)
        return record

    def touch_session(
        self,
        session_id: str,
        *,
        model: str | None = None,
        provider_name: str | None = None,
        inference_provider: str | None = None,
        inference_provider_mode: InferenceProviderMode | None = None,
        preserve_inference_provider: bool = True,
        title: str | None = None,
    ) -> CodingSessionRecord | None:
        # [FUNCTION DOC]
        # Updates an indexed session's metadata and bumps updated time to now.
        # Inputs: id; optional new model (falsy keeps old), provider name, title (none
        # keeps old); inference provider and mode, applied only when
        # preserve_inference_provider is false (provider then replaced even by none;
        # mode replaced only when given).
        # Outputs: the updated record, or none if the id is unknown.
        # Side effects: rewrites the project index.
        # KNOWN ISSUE: clearing the inference provider without passing a mode keeps the
        # old mode, so a record can end up "fixed" with no inference provider.
        """Update a session's last-used metadata."""
        existing = self.get_session(session_id)
        if existing is None:
            return None
        updated = CodingSessionRecord(
            id=existing.id,
            path=existing.path,
            cwd=existing.cwd,
            model=model or existing.model,
            provider_name=provider_name if provider_name is not None else existing.provider_name,
            inference_provider=(
                existing.inference_provider if preserve_inference_provider else inference_provider
            ),
            inference_provider_mode=(
                existing.inference_provider_mode
                if preserve_inference_provider or inference_provider_mode is None
                else inference_provider_mode
            ),
            title=title if title is not None else existing.title,
            created_at=existing.created_at,
            updated_at=time(),
        )
        self._upsert(updated)
        return updated

    def _read_index(self, path: Path) -> list[CodingSessionRecord]:
        # [FUNCTION DOC]
        # HELPER: Reads a JSONL index file into records, splitting only on newline
        # characters (not other Unicode line separators) and skipping blank lines.
        # Returns empty list if the file is missing. Raises validation errors for
        # malformed rows and OS errors on read failure.
        if not path.exists():
            return []

        records: list[CodingSessionRecord] = []
        # Split on newlines only: str.splitlines() would also split on characters
        # like U+2028 that appear unescaped inside JSON string values.
        for line in path.read_text(encoding="utf-8").split("\n"):
            stripped = line.strip()
            if not stripped:
                continue
            model = SessionRecordModel.model_validate_json(stripped)
            records.append(CodingSessionRecord.from_model(model))
        return records

    def _read_project_records(self, cwd: Path) -> list[CodingSessionRecord]:
        # [FUNCTION DOC]
        # HELPER: Records from the project's index plus legacy-index rows whose cwd
        # equals the resolved cwd, deduplicated by id.
        resolved_cwd = cwd.resolve()
        records = self._read_index(self.project_index_path(resolved_cwd))
        records.extend(
            record for record in self._read_index(self.index_path) if record.cwd == resolved_cwd
        )
        return _deduplicate_records(records)

    def _read_all_records(self) -> list[CodingSessionRecord]:
        # [FUNCTION DOC]
        # HELPER: Records from the legacy index and every "*/index.jsonl" under the
        # sessions directory, deduplicated by id.
        records = self._read_index(self.index_path)
        for index_path in self.paths.sessions_dir.glob("*/index.jsonl"):
            records.extend(self._read_index(index_path))
        return _deduplicate_records(records)

    def _write_index(self, path: Path, records: list[CodingSessionRecord]) -> None:
        # [FUNCTION DOC]
        # HELPER: Creates the parent directory and overwrites the index file with one
        # JSON row per record (trailing newline when non-empty; empty file otherwise).
        path.parent.mkdir(parents=True, exist_ok=True)
        content = "\n".join(record.to_model().model_dump_json() for record in records)
        if content:
            content += "\n"
        path.write_text(content, encoding="utf-8")

    def _upsert(self, record: CodingSessionRecord) -> None:
        # [FUNCTION DOC]
        # HELPER: Rewrites the record's project index with any same-id row replaced by
        # the record, appended at the end.
        path = self.project_index_path(record.cwd)
        records = [item for item in self._read_index(path) if item.id != record.id]
        records.append(record)
        self._write_index(path, records)

    def _remove(self, record: CodingSessionRecord) -> None:
        # [FUNCTION DOC]
        # HELPER: Removes the record's id from its project index.
        self._remove_from_index(self.project_index_path(record.cwd), record.id)

    def _remove_from_index(self, path: Path, session_id: str) -> None:
        # [FUNCTION DOC]
        # HELPER: Rewrites the given index without rows having the session id.
        records = [item for item in self._read_index(path) if item.id != session_id]
        self._write_index(path, records)

    def _add_to_archive(self, records: list[CodingSessionRecord]) -> None:
        # [FUNCTION DOC]
        # HELPER: Rewrites the archive index as its deduplicated rows minus any with
        # the incoming ids, followed by the incoming records.
        archived = [
            item
            for item in _deduplicate_records(self._read_index(self.archive_index_path))
            if all(item.id != record.id for record in records)
        ]
        archived.extend(records)
        self._write_index(self.archive_index_path, archived)

    def _remove_from_archive(self, session_id: str) -> None:
        # [FUNCTION DOC]
        # HELPER: Removes the id from the archive index.
        self._remove_from_index(self.archive_index_path, session_id)


def _deduplicate_records(records: list[CodingSessionRecord]) -> list[CodingSessionRecord]:
    # [FUNCTION DOC]
    # HELPER: Collapses records by id, keeping the one with the greatest updated
    # time (later duplicates win ties), preserving first-seen id order.
    by_id: dict[str, CodingSessionRecord] = {}
    for record in records:
        existing = by_id.get(record.id)
        if existing is None or record.updated_at >= existing.updated_at:
            by_id[record.id] = record
    return list(by_id.values())
