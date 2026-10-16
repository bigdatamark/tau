# [MODULE DOC]
# Tests for tau_coding.update_check: the startup PyPI update notice (newest stable release, cached
# for about a day, failures ignored, disabled by TAU_NO_UPDATE_CHECK or CI), release-notes loading
# from the shared releases JSON, the post-upgrade "what's new" notice (state file of last seen
# version, combining skipped versions, tolerating missing/malformed notes files), the version range
# filter, and PyPI latest-version parsing. Uses injected fetchers, clocks and tmp_path files.
from datetime import UTC, datetime, timedelta

from tau_coding.update_check import (
    PYPI_JSON_URL,
    UPDATE_CHECK_TIMEOUT_SECONDS,
    ReleaseNoteSection,
    ReleaseNotesEntry,
    fetch_latest_pypi_version,
    load_release_notes,
    release_notes_between,
    startup_release_notes_notice,
    startup_update_notice,
)


def test_startup_update_notice_reports_newer_stable_release(tmp_path) -> None:
    # [FUNCTION DOC]
    # Test: with releases 0.1.0, 0.2.0 and a 0.3.0rc1 pre-release, installed 0.1.0 gets a notice for
    # 0.2.0 with the upgrade hint; the fetcher is called once with the PyPI URL and the timeout.
    calls: list[tuple[str, float]] = []

    def fetcher(url: str, timeout: float) -> dict[str, object]:
        # [FUNCTION DOC]
        # HELPER: Fake fetcher recording (url, timeout) and returning a releases mapping.
        calls.append((url, timeout))
        return {"releases": {"0.1.0": [{}], "0.2.0": [{}], "0.3.0rc1": [{}]}}

    notice = startup_update_notice(
        "0.1.0",
        fetcher=fetcher,
        cache_path=tmp_path / "update-check.json",
        now=lambda: datetime(2026, 1, 1, tzinfo=UTC),
        env={},
    )

    assert notice is not None
    assert notice.current_version == "0.1.0"
    assert notice.latest_version == "0.2.0"
    assert "Tau 0.2.0 is available (installed: 0.1.0)" in notice.message
    assert "Run `tau update` to upgrade" in notice.message
    assert calls == [(PYPI_JSON_URL, UPDATE_CHECK_TIMEOUT_SECONDS)]


def test_startup_update_notice_is_quiet_when_current(tmp_path) -> None:
    # [FUNCTION DOC]
    # Test: no notice when the installed version equals the latest release.
    notice = startup_update_notice(
        "0.2.0",
        fetcher=lambda _url, _timeout: {"releases": {"0.2.0": [{}]}},
        cache_path=tmp_path / "update-check.json",
        env={},
    )

    assert notice is None


def test_startup_update_notice_uses_fresh_cache(tmp_path) -> None:
    # [FUNCTION DOC]
    # Test: a cache checked 12 hours earlier is used (fetcher would raise) and yields a notice for 0.2.0.
    cache_path = tmp_path / "update-check.json"
    cache_path.write_text(
        '{"checked_at":"2026-01-01T00:00:00+00:00","latest_version":"0.2.0"}\n',
        encoding="utf-8",
    )

    notice = startup_update_notice(
        "0.1.0",
        fetcher=lambda _url, _timeout: (_ for _ in ()).throw(AssertionError("no fetch")),
        cache_path=cache_path,
        now=lambda: datetime(2026, 1, 1, 12, tzinfo=UTC),
        env={},
    )

    assert notice is not None
    assert notice.latest_version == "0.2.0"


def test_startup_update_notice_uses_fresh_empty_cache(tmp_path) -> None:
    # [FUNCTION DOC]
    # Test: a fresh cache with a null latest version yields no notice and no fetch.
    cache_path = tmp_path / "update-check.json"
    cache_path.write_text(
        '{"checked_at":"2026-01-01T00:00:00+00:00","latest_version":null}\n',
        encoding="utf-8",
    )

    notice = startup_update_notice(
        "0.1.0",
        fetcher=lambda _url, _timeout: (_ for _ in ()).throw(AssertionError("no fetch")),
        cache_path=cache_path,
        now=lambda: datetime(2026, 1, 1, 12, tzinfo=UTC),
        env={},
    )

    assert notice is None


def test_startup_update_notice_refreshes_stale_cache(tmp_path) -> None:
    # [FUNCTION DOC]
    # Test: a cache two days old is refreshed from the fetcher, reporting 0.3.0.
    cache_path = tmp_path / "update-check.json"
    cache_path.write_text(
        '{"checked_at":"2026-01-01T00:00:00+00:00","latest_version":"0.2.0"}\n',
        encoding="utf-8",
    )

    notice = startup_update_notice(
        "0.1.0",
        fetcher=lambda _url, _timeout: {"releases": {"0.3.0": [{}]}},
        cache_path=cache_path,
        now=lambda: datetime(2026, 1, 1, tzinfo=UTC) + timedelta(days=2),
        env={},
    )

    assert notice is not None
    assert notice.latest_version == "0.3.0"


def test_startup_update_notice_ignores_failures(tmp_path) -> None:
    # [FUNCTION DOC]
    # Test: a fetcher raising TimeoutError produces no notice instead of an exception.
    def broken_fetcher(_url: str, _timeout: float) -> dict[str, object]:
        # [FUNCTION DOC]
        # HELPER: Fetcher that always raises TimeoutError("offline").
        raise TimeoutError("offline")

    assert (
        startup_update_notice(
            "0.1.0",
            fetcher=broken_fetcher,
            cache_path=tmp_path / "update-check.json",
            env={},
        )
        is None
    )


def test_startup_update_notice_can_be_disabled(tmp_path) -> None:
    # [FUNCTION DOC]
    # Test: TAU_NO_UPDATE_CHECK=1 disables the check entirely (no fetch, no notice).
    notice = startup_update_notice(
        "0.1.0",
        fetcher=lambda _url, _timeout: (_ for _ in ()).throw(AssertionError("no fetch")),
        cache_path=tmp_path / "update-check.json",
        env={"TAU_NO_UPDATE_CHECK": "1"},
    )

    assert notice is None


def test_startup_update_notice_skips_ci(tmp_path) -> None:
    # [FUNCTION DOC]
    # Test: CI=true skips the check entirely (no fetch, no notice).
    notice = startup_update_notice(
        "0.1.0",
        fetcher=lambda _url, _timeout: (_ for _ in ()).throw(AssertionError("no fetch")),
        cache_path=tmp_path / "update-check.json",
        env={"CI": "true"},
    )

    assert notice is None


def test_load_release_notes_reads_shared_json(tmp_path) -> None:
    # [FUNCTION DOC]
    # Test: a releases JSON list is parsed into release-notes entries with ordered titled sections.
    notes_path = tmp_path / "releases.json"
    notes_path.write_text(
        """
        [
          {
            "version": "0.1.2",
            "date": "2026-07-03",
            "sections": {"New": ["Feature"], "Fixed": ["Fix"]}
          }
        ]
        """,
        encoding="utf-8",
    )

    notes = load_release_notes(notes_path)

    assert notes == (
        ReleaseNotesEntry(
            version="0.1.2",
            date="2026-07-03",
            sections=(
                ReleaseNoteSection(title="New", items=("Feature",)),
                ReleaseNoteSection(title="Fixed", items=("Fix",)),
            ),
        ),
    )


def test_startup_release_notes_notice_records_first_seen_version(tmp_path) -> None:
    # [FUNCTION DOC]
    # Test: with no state file, the first run shows no notice but records the current version as last seen.
    state_path = tmp_path / "release-notes-state.json"

    notice = startup_release_notes_notice(
        "0.1.2",
        state_path=state_path,
        release_notes=(
            ReleaseNotesEntry(
                version="0.1.2",
                date=None,
                sections=(ReleaseNoteSection(title="New", items=("New TUI release notes",)),),
            ),
        ),
    )

    assert notice is None
    assert '"last_seen_version": "0.1.2"' in state_path.read_text(encoding="utf-8")


def test_startup_release_notes_notice_reports_upgrade_once(tmp_path) -> None:
    # [FUNCTION DOC]
    # Test: upgrading from 0.1.1 to 0.1.2 produces one notice with versions, flattened notes and a
    # Markdown message; a second call returns None because the state was updated.
    state_path = tmp_path / "release-notes-state.json"
    state_path.write_text('{"last_seen_version":"0.1.1"}\n', encoding="utf-8")
    release_notes = (
        ReleaseNotesEntry(
            version="0.1.2",
            date=None,
            sections=(ReleaseNoteSection(title="New", items=("New feature", "Bug fix")),),
        ),
    )

    notice = startup_release_notes_notice(
        "0.1.2",
        state_path=state_path,
        release_notes=release_notes,
    )

    assert notice is not None
    assert notice.previous_version == "0.1.1"
    assert notice.current_version == "0.1.2"
    assert notice.notes == ("New feature", "Bug fix")
    assert notice.message == "Tau updated to 0.1.2\n\n**New**\n- New feature\n- Bug fix"

    second_notice = startup_release_notes_notice(
        "0.1.2",
        state_path=state_path,
        release_notes=release_notes,
    )
    assert second_notice is None


def test_startup_release_notes_notice_survives_missing_release_notes_file(
    tmp_path, monkeypatch
) -> None:
    # [FUNCTION DOC]
    # Test (regression #313): a missing default releases file still yields an upgrade notice with no entries.
    # Regression test for issue #313: a wheel missing releases.json crashed
    # startup with FileNotFoundError instead of skipping the notice.
    import tau_coding.update_check as update_check_module

    monkeypatch.setattr(
        update_check_module, "RELEASE_NOTES_PATH", tmp_path / "missing" / "releases.json"
    )
    state_path = tmp_path / "release-notes-state.json"
    state_path.write_text('{"last_seen_version":"0.1.1"}\n', encoding="utf-8")

    notice = startup_release_notes_notice("0.1.2", state_path=state_path)

    assert notice is not None
    assert notice.entries == ()


def test_startup_release_notes_notice_survives_malformed_release_notes_file(
    tmp_path, monkeypatch
) -> None:
    # [FUNCTION DOC]
    # Test: an unparseable default releases file still yields an upgrade notice with no entries.
    import tau_coding.update_check as update_check_module

    broken_path = tmp_path / "releases.json"
    broken_path.write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(update_check_module, "RELEASE_NOTES_PATH", broken_path)
    state_path = tmp_path / "release-notes-state.json"
    state_path.write_text('{"last_seen_version":"0.1.1"}\n', encoding="utf-8")

    notice = startup_release_notes_notice("0.1.2", state_path=state_path)

    assert notice is not None
    assert notice.entries == ()


def test_startup_release_notes_notice_combines_skipped_versions(tmp_path) -> None:
    # [FUNCTION DOC]
    # Test: upgrading 0.1.0 -> 0.1.2 combines notes from 0.1.1 and 0.1.2 in ascending version order
    # and excludes the future 0.1.3.
    state_path = tmp_path / "release-notes-state.json"
    state_path.write_text('{"last_seen_version":"0.1.0"}\n', encoding="utf-8")
    release_notes = (
        ReleaseNotesEntry(
            version="0.1.2",
            date=None,
            sections=(ReleaseNoteSection(title="New", items=("Second change",)),),
        ),
        ReleaseNotesEntry(
            version="0.1.1",
            date=None,
            sections=(ReleaseNoteSection(title="Fixed", items=("First change",)),),
        ),
        ReleaseNotesEntry(
            version="0.1.3",
            date=None,
            sections=(ReleaseNoteSection(title="New", items=("Future change",)),),
        ),
    )

    notice = startup_release_notes_notice(
        "0.1.2",
        state_path=state_path,
        release_notes=release_notes,
    )

    assert notice is not None
    assert notice.notes == ("First change", "Second change")


def test_release_notes_between_ignores_future_versions() -> None:
    # [FUNCTION DOC]
    # Test: the range filter is exclusive of the previous and inclusive of the current version.
    entries = (
        ReleaseNotesEntry(version="0.1.1", date=None, sections=()),
        ReleaseNotesEntry(version="0.1.2", date=None, sections=()),
        ReleaseNotesEntry(version="0.1.3", date=None, sections=()),
    )

    assert release_notes_between("0.1.1", "0.1.2", entries) == (entries[1],)


def test_fetch_latest_pypi_version_falls_back_to_info_version() -> None:
    # [FUNCTION DOC]
    # Test: without a releases mapping, the latest version falls back to the info version.
    latest = fetch_latest_pypi_version(
        fetcher=lambda _url, _timeout: {"info": {"version": "0.4.0"}}
    )

    assert latest == "0.4.0"


def test_fetch_latest_pypi_version_skips_malformed_release_versions() -> None:
    # [FUNCTION DOC]
    # Test: unparseable release keys are skipped when selecting the latest version.
    latest = fetch_latest_pypi_version(
        fetcher=lambda _url, _timeout: {"releases": {"0.3.0": [{}], "wat": [{}]}}
    )

    assert latest == "0.3.0"


def test_fetch_latest_pypi_version_rejects_malformed_versions() -> None:
    # [FUNCTION DOC]
    # Test: an unparseable info version raises the packaging InvalidVersion exception.
    try:
        fetch_latest_pypi_version(fetcher=lambda _url, _timeout: {"info": {"version": "wat"}})
    except Exception as exc:
        assert exc.__class__.__name__ == "InvalidVersion"
    else:
        raise AssertionError("expected InvalidVersion")


def test_load_release_notes_resolves_default_path() -> None:
    # [FUNCTION DOC]
    # Test: loading release notes from the default path finds at least one entry, each with a version.
    # Reads the real bundled releases file.
    """Regression: load_release_notes() resolves the module-level RELEASE_NOTES_PATH
    correctly in both dev and installed layouts."""
    entries = load_release_notes()
    assert len(entries) > 0, "should find at least one release entry"
    assert all(entry.version for entry in entries), "every entry must have a version"
