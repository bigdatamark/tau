# [MODULE DOC]
# Best-effort startup notices for the Tau CLI (tau_coding layer):
# 1) update checks: looks up the latest stable "tau-ai" release on PyPI at most once
#    per day (cached in <Tau home>/cache/update-check.json, 1.5s timeout), skippable
#    via TAU_NO_UPDATE_CHECK or the CI environment variable;
# 2) one-time release notes: remembers the last-seen installed version in
#    <Tau home>/cache/release-notes-state.json and, after an upgrade, returns the
#    bundled release-note entries between the old and new versions.
# Every failure (network, JSON, disk, version parsing) is swallowed so startup never
# blocks. The fetcher, clock, environment and paths are injectable for tests.
"""Best-effort PyPI update checks for the Tau CLI."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from os import environ
from pathlib import Path
from typing import Any

from packaging.version import InvalidVersion, Version

from tau_ai.http import get_json
from tau_coding.paths import TauPaths

PYPI_PACKAGE_NAME = "tau-ai"
PYPI_JSON_URL = f"https://pypi.org/pypi/{PYPI_PACKAGE_NAME}/json"
UPDATE_CHECK_INTERVAL = timedelta(days=1)
UPDATE_CHECK_TIMEOUT_SECONDS = 1.5
UPDATE_CHECK_ENV_DISABLE = "TAU_NO_UPDATE_CHECK"
RELEASE_NOTES_STATE_FILENAME = "release-notes-state.json"
RELEASE_NOTES_PATH = Path(__file__).resolve().parent / "data" / "release-notes" / "releases.json"

Fetcher = Callable[[str, float], dict[str, Any]]
Clock = Callable[[], datetime]


@dataclass(frozen=True, slots=True)
class UpdateNotice:
    # [CLASS DOC]
    # Immutable notice that a newer version exists: installed version, latest version,
    # and package name (default "tau-ai").
    """A user-facing update notice."""

    current_version: str
    latest_version: str
    package_name: str = PYPI_PACKAGE_NAME

    @property
    def message(self) -> str:
        # [FUNCTION DOC]
        # Property: one-line guidance naming the latest and installed versions and telling
        # the user to run "tau update".
        """Return concise update guidance."""
        return (
            f"Tau {self.latest_version} is available (installed: {self.current_version}). "
            "Run `tau update` to upgrade."
        )


@dataclass(frozen=True, slots=True)
class UpdateCheckResult:
    # [CLASS DOC]
    # Immutable cached lookup result: when the check happened (timezone-aware) and the
    # latest stable version found, or None if none was found.
    """Cached latest-version lookup result."""

    checked_at: datetime
    latest_version: str | None


@dataclass(frozen=True, slots=True)
class ReleaseNoteSection:
    # [CLASS DOC]
    # Immutable named group of release-note bullet strings (e.g. "Added").
    """A named release-note section."""

    title: str
    items: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ReleaseNotesEntry:
    # [CLASS DOC]
    # Immutable release notes for one version: version string, optional date string,
    # and ordered sections.
    """Structured release notes for one Tau version."""

    version: str
    date: str | None
    sections: tuple[ReleaseNoteSection, ...]

    @property
    def transcript_items(self) -> tuple[str, ...]:
        # [FUNCTION DOC]
        # Property: all bullet items of all sections flattened in order.
        """Return flattened note text for compact in-app display."""
        return tuple(item for section in self.sections for item in section.items)


@dataclass(frozen=True, slots=True)
class ReleaseNotesNotice:
    # [CLASS DOC]
    # Immutable notice shown once after an upgrade: current version, previously seen
    # version, and the release-note entries in between (ascending version order).
    """Release notes shown once after the installed Tau version changes."""

    current_version: str
    previous_version: str
    entries: tuple[ReleaseNotesEntry, ...]

    @property
    def notes(self) -> tuple[str, ...]:
        # [FUNCTION DOC]
        # Property: all bullet items across all entries, flattened in order.
        """Return all release note items included in this notice."""
        return tuple(item for entry in self.entries for item in entry.transcript_items)

    @property
    def message(self) -> str:
        # [FUNCTION DOC]
        # Property: Markdown text "Tau updated to <version>" followed by each entry's
        # formatted sections separated by blank lines, or a single "See the changelog"
        # bullet when there are no entries. Entry versions themselves are not printed.
        """Return a compact markdown release-notes block for the transcript."""
        if self.entries:
            version_blocks = [self._format_entry(entry) for entry in self.entries]
            body = "\n\n".join(version_blocks)
        else:
            body = "- See the changelog for details."
        return f"Tau updated to {self.current_version}\n\n{body}"

    def _format_entry(self, entry: ReleaseNotesEntry) -> str:
        # [FUNCTION DOC]
        # HELPER: Format one entry as blocks of a bold section title followed by "- item"
        # bullet lines, blocks separated by blank lines.
        section_blocks: list[str] = []
        for section in entry.sections:
            bullets = "\n".join(f"- {item}" for item in section.items)
            section_blocks.append(f"**{section.title}**\n{bullets}")
        return "\n\n".join(section_blocks)


def startup_update_notice(
    current_version: str,
    *,
    fetcher: Fetcher | None = None,
    cache_path: Path | None = None,
    now: Clock | None = None,
    env: Mapping[str, str] | None = None,
) -> UpdateNotice | None:
    # [FUNCTION DOC]
    # Return an UpdateNotice when a newer stable release than the current version is
    # known, else None.
    # Inputs: installed version string; keyword overrides for the fetcher (url, timeout
    # to JSON object), cache path, clock, and environment mapping.
    # Behavior: return None if checks are disabled by the environment; use a cached
    # result younger than one day if available; otherwise fetch from PyPI (any exception
    # returns None) and write the result (including a None version) to the cache.
    # Returns None when no version is known, when it is not newer, or when either
    # version string is invalid. Raises nothing.
    # Side effects: may perform an HTTP request and write the cache file.
    """Return an update notice when PyPI has a newer stable Tau release.

    This function is intentionally best-effort: cache, network, JSON, and version
    parsing failures all become quiet no-ops so startup can continue.
    """
    environment = environ if env is None else env
    if _update_check_disabled(environment):
        return None

    current_time = (now or _utc_now)()
    cached_result = _cached_update_check_result(cache_path, current_time)
    if cached_result is None:
        try:
            latest_version = fetch_latest_pypi_version(fetcher=fetcher)
        except Exception:  # noqa: BLE001 - update checks must never block startup
            return None
        _write_update_check_cache(cache_path, current_time, latest_version)
    else:
        latest_version = cached_result.latest_version

    if latest_version is None:
        return None

    try:
        if Version(latest_version) <= Version(current_version):
            return None
    except InvalidVersion:
        return None

    return UpdateNotice(current_version=current_version, latest_version=latest_version)


def startup_release_notes_notice(
    current_version: str,
    *,
    state_path: Path | None = None,
    release_notes: tuple[ReleaseNotesEntry, ...] | None = None,
) -> ReleaseNotesNotice | None:
    # [FUNCTION DOC]
    # Return release notes once after the installed version increases, else None.
    # Reads the last-seen version from the state file (any error is treated as no
    # state), then always records the current version. Returns None on first run (no
    # previous state), when the version did not increase, or on invalid versions.
    # Otherwise loads bundled notes unless supplied (load failure means no entries) and
    # returns a notice with entries in the half-open range previous < v <= current.
    # Inputs: current version; optional state path and preloaded notes.
    # Side effects: reads and writes the state file. Raises nothing.
    """Return release notes once after Tau starts with a newer installed version.

    The first run only records the current version so fresh installs do not see an
    upgrade banner. Read/write failures are quiet no-ops so startup can continue.
    """
    path = state_path or default_release_notes_state_path()
    try:
        previous_version = _read_last_seen_version(path)
    except Exception:  # noqa: BLE001 - release-note state should not block startup
        previous_version = None

    _write_release_notes_state(path, current_version)
    if previous_version is None:
        return None

    try:
        if Version(current_version) <= Version(previous_version):
            return None
    except InvalidVersion:
        return None

    if release_notes is None:
        try:
            release_notes = load_release_notes()
        except Exception:  # noqa: BLE001 - a broken bundled file must not block startup
            release_notes = ()

    entries = release_notes_between(
        previous_version,
        current_version,
        release_notes,
    )
    return ReleaseNotesNotice(
        current_version=current_version,
        previous_version=previous_version,
        entries=entries,
    )


def load_release_notes(path: Path | None = None) -> tuple[ReleaseNotesEntry, ...]:
    # [FUNCTION DOC]
    # Read and parse the release-notes JSON file (default: the bundled data file).
    # Output: tuple of entries in file order.
    # Raises OSError for unreadable files, JSON errors for invalid JSON, and ValueError
    # if the top level is not an array or any entry is malformed.
    """Load structured release notes from the repo-owned JSON file."""
    data = json.loads((path or RELEASE_NOTES_PATH).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("release notes must be a JSON array")
    return tuple(_parse_release_notes_entry(item) for item in data)


def release_notes_between(
    previous_version: str,
    current_version: str,
    release_notes: tuple[ReleaseNotesEntry, ...],
) -> tuple[ReleaseNotesEntry, ...]:
    # [FUNCTION DOC]
    # Select entries whose version v satisfies previous < v <= current, skipping entries
    # with invalid versions, sorted ascending by version. Returns an empty tuple if
    # either bound is invalid. Pure.
    """Return release-note entries newer than previous_version up to current_version."""
    try:
        previous = Version(previous_version)
        current = Version(current_version)
    except InvalidVersion:
        return ()

    selected: list[ReleaseNotesEntry] = []
    for entry in release_notes:
        try:
            parsed = Version(entry.version)
        except InvalidVersion:
            continue
        if previous < parsed <= current:
            selected.append(entry)
    return tuple(sorted(selected, key=lambda entry: Version(entry.version)))


def fetch_latest_pypi_version(*, fetcher: Fetcher | None = None) -> str | None:
    # [FUNCTION DOC]
    # Fetch PyPI's JSON metadata and return the newest stable version as a string.
    # Prefers the maximum non-prerelease, non-dev version from "releases" that has at
    # least one file (or non-list file data); falls back to "info.version" if it is
    # stable; else returns None.
    # Input: optional fetcher (defaults to an HTTP GET via the provider HTTP helper).
    # Raises whatever the fetcher raises, and InvalidVersion if info.version is
    # malformed. Side effects: network request.
    """Fetch the latest stable Tau version from PyPI."""
    data = (fetcher or _httpx_fetch_json)(PYPI_JSON_URL, UPDATE_CHECK_TIMEOUT_SECONDS)
    releases = data.get("releases")
    if isinstance(releases, dict):
        versions = _stable_release_versions(releases)
        if versions:
            return str(max(versions))

    info = data.get("info")
    if isinstance(info, dict):
        version = info.get("version")
        if isinstance(version, str):
            parsed = Version(version)
            if not parsed.is_prerelease and not parsed.is_devrelease:
                return version
    return None


def default_update_check_cache_path(paths: TauPaths | None = None) -> Path:
    # [FUNCTION DOC]
    # Return <Tau home>/cache/update-check.json for the given (or default) Tau paths.
    """Return the on-disk cache path for startup update checks."""
    return (paths or TauPaths()).home / "cache" / "update-check.json"


def default_release_notes_state_path(paths: TauPaths | None = None) -> Path:
    # [FUNCTION DOC]
    # Return <Tau home>/cache/release-notes-state.json for the given (or default) paths.
    """Return the on-disk state path for one-time release notes."""
    return (paths or TauPaths()).home / "cache" / RELEASE_NOTES_STATE_FILENAME


def _parse_release_notes_entry(data: Any) -> ReleaseNotesEntry:
    # [FUNCTION DOC]
    # HELPER: Validate and convert one JSON release-note object: required string
    # "version", optional string "date", and "sections" as an object mapping titles to
    # lists of strings (order preserved). Raises ValueError describing the first
    # problem found.
    if not isinstance(data, dict):
        raise ValueError("release note entry must be an object")
    version = data.get("version")
    if not isinstance(version, str):
        raise ValueError("release note entry missing version")
    date = data.get("date")
    if date is not None and not isinstance(date, str):
        raise ValueError("release note date must be a string")
    raw_sections = data.get("sections")
    if not isinstance(raw_sections, dict):
        raise ValueError("release note sections must be an object")

    sections: list[ReleaseNoteSection] = []
    for title, raw_items in raw_sections.items():
        if not isinstance(title, str):
            raise ValueError("release note section title must be a string")
        if not isinstance(raw_items, list) or not all(isinstance(item, str) for item in raw_items):
            raise ValueError("release note section items must be strings")
        sections.append(ReleaseNoteSection(title=title, items=tuple(raw_items)))
    return ReleaseNotesEntry(version=version, date=date, sections=tuple(sections))


def _stable_release_versions(releases: dict[Any, Any]) -> list[Version]:
    # [FUNCTION DOC]
    # HELPER: From PyPI's releases mapping, collect parsed versions that are strings,
    # valid, not pre/dev releases, and not explicitly empty file lists (yanked or
    # placeholder). Returns a list of versions.
    versions: list[Version] = []
    for version_text, files in releases.items():
        if not isinstance(version_text, str):
            continue
        if isinstance(files, list) and not files:
            continue
        try:
            parsed = Version(version_text)
        except InvalidVersion:
            continue
        if parsed.is_prerelease or parsed.is_devrelease:
            continue
        versions.append(parsed)
    return versions


def _httpx_fetch_json(url: str, timeout_seconds: float) -> dict[str, Any]:
    # [FUNCTION DOC]
    # HELPER: Default fetcher: GET the URL with the given timeout, following redirects,
    # and return the decoded JSON object. Raises ValueError if the body is not an
    # object; HTTP/network errors propagate.
    data = get_json(url, timeout=timeout_seconds, follow_redirects=True)
    if not isinstance(data, dict):
        raise ValueError("PyPI response must be a JSON object")
    return data


def _cached_update_check_result(cache_path: Path | None, now: datetime) -> UpdateCheckResult | None:
    # [FUNCTION DOC]
    # HELPER: Load and parse the cache file (default path if none given); return None
    # for any read/parse failure or when the entry is older than the check interval.
    # KNOWN ISSUE: a checked_at timestamp in the future (clock skew) is always treated
    # as fresh until the clock catches up.
    path = cache_path or default_update_check_cache_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        result = _parse_cached_result(data)
    except Exception:  # noqa: BLE001 - corrupt caches should be ignored
        return None
    if now - result.checked_at > UPDATE_CHECK_INTERVAL:
        return None
    return result


def _parse_cached_result(data: Any) -> UpdateCheckResult:
    # [FUNCTION DOC]
    # HELPER: Validate the cache JSON object (string "checked_at" in ISO format,
    # optional string "latest_version") and return an UpdateCheckResult; naive
    # timestamps are assumed UTC. Raises ValueError on invalid shape or timestamp.
    if not isinstance(data, dict):
        raise ValueError("cache must be a JSON object")
    checked_at = data.get("checked_at")
    latest_version = data.get("latest_version")
    if not isinstance(checked_at, str):
        raise ValueError("cache missing checked_at")
    if latest_version is not None and not isinstance(latest_version, str):
        raise ValueError("cache latest_version must be a string")
    parsed_checked_at = datetime.fromisoformat(checked_at)
    if parsed_checked_at.tzinfo is None:
        parsed_checked_at = parsed_checked_at.replace(tzinfo=UTC)
    return UpdateCheckResult(checked_at=parsed_checked_at, latest_version=latest_version)


def _write_update_check_cache(
    cache_path: Path | None,
    checked_at: datetime,
    latest_version: str | None,
) -> None:
    # [FUNCTION DOC]
    # HELPER: Write the check time (as UTC ISO-8601) and latest version as indented JSON
    # to the cache file, creating parent directories. OS errors are ignored.
    path = cache_path or default_update_check_cache_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "checked_at": checked_at.astimezone(UTC).isoformat(),
                    "latest_version": latest_version,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    except OSError:
        return


def _read_last_seen_version(path: Path) -> str | None:
    # [FUNCTION DOC]
    # HELPER: Read "last_seen_version" from the release-notes state JSON. Returns the
    # string or None. Raises on missing file, invalid JSON, or wrong types.
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("release-note state must be a JSON object")
    version = data.get("last_seen_version")
    if version is not None and not isinstance(version, str):
        raise ValueError("release-note last_seen_version must be a string")
    return version


def _write_release_notes_state(path: Path, current_version: str) -> None:
    # [FUNCTION DOC]
    # HELPER: Write the current version as "last_seen_version" JSON to the state file,
    # creating parent directories. OS errors are ignored.
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"last_seen_version": current_version}, indent=2) + "\n",
            encoding="utf-8",
        )
    except OSError:
        return


def _update_check_disabled(env: Mapping[str, str]) -> bool:
    # [FUNCTION DOC]
    # HELPER: Return True if TAU_NO_UPDATE_CHECK is set to anything other than empty,
    # "0", "false" or "no" (case-insensitive), or if CI is set to any non-empty value.
    # KNOWN ISSUE: CI set to a falsy-looking string such as "false" or "0" still
    # disables the check, unlike the TAU_NO_UPDATE_CHECK handling.
    value = env.get(UPDATE_CHECK_ENV_DISABLE)
    if value is not None and value.strip().lower() not in {"", "0", "false", "no"}:
        return True
    return bool(env.get("CI"))


def _utc_now() -> datetime:
    # [FUNCTION DOC]
    # HELPER: Return the current timezone-aware UTC datetime (default clock).
    return datetime.now(UTC)
