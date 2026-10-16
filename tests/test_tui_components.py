# [MODULE DOC]
# Focused Textual tests for the extension component (widget-hosting) seam of the TUI app, driven
# through the host bridge helper from the sibling TUI app tests: above/below-prompt slot widgets,
# ordering, main-view replacement and open/close transcript restore, same-id swaps in one tick
# without duplicate ids, host exceptions not being swallowed, sidebar sections keyed by
# (extension, key) with update/remove/reorder, sidebar unsupported when off, responsive hiding,
# live-theme rebuilds, and factory/render failure quarantine with diagnostics.
"""Focused tests for the component (widget-hosting) seam.

These drive the generic seam on the real ``TauTuiApp`` via the host bridge,
reusing the fakes/helpers that already back the broader TUI suite in
``test_tui_app`` (pytest's prepend import mode puts the tests directory on
``sys.path``, so a sibling test module is importable). They close the gaps the
seam's code review flagged: slot placement/ordering, main-view re-open, host
exceptions staying un-swallowed, and main-view open/close restoring the main
transcript.
"""

from pathlib import Path

import pytest
from textual.containers import Container
from textual.widgets import Static

from conftest import isolate_home
from tau_coding.extensions import ExtensionRuntime
from tau_coding.tui.app import PromptInput, TauTuiApp
from tau_coding.tui.config import TuiSettings
from tau_coding.tui.widgets import TranscriptView
from test_tui_app import (  # noqa: E402 - sibling test module (see docstring)
    FakeSession,
    _component_bridge,
)


@pytest.mark.anyio
async def test_component_above_prompt_slot_mounts_into_above_slot() -> None:
    # [FUNCTION DOC]
    # Async test: an above-prompt slot widget mounts in the above-prompt container only and is tracked by key.
    app = TauTuiApp(FakeSession())

    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        bridge = _component_bridge(app)

        bridge.set_slot_widget(
            "top",
            lambda theme: Static("above", id="ext-above"),
            placement="above_prompt",
        )
        await pilot.pause()

        above = app.query_one("#above-prompt-slot", Container)
        assert above.query("#ext-above")
        # It must not have leaked into the other placement's slot.
        assert not app.query_one("#below-prompt-slot", Container).query("#ext-above")
        assert "top" in app._extension_slot_widgets


@pytest.mark.anyio
async def test_component_multiple_slot_widgets_mount_in_call_order() -> None:
    # [FUNCTION DOC]
    # Async test: two below-prompt slot widgets mount in call order and are tracked in that order.
    app = TauTuiApp(FakeSession())

    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        bridge = _component_bridge(app)

        bridge.set_slot_widget("a", lambda theme: Static("a", id="ext-a"), placement="below_prompt")
        bridge.set_slot_widget("b", lambda theme: Static("b", id="ext-b"), placement="below_prompt")
        await pilot.pause()

        slot = app.query_one("#below-prompt-slot", Container)
        assert [child.id for child in slot.children] == ["ext-a", "ext-b"]
        assert list(app._extension_slot_widgets) == ["a", "b"]


@pytest.mark.anyio
async def test_component_second_main_view_replaces_first() -> None:
    # [FUNCTION DOC]
    # Async test: opening a second main view closes the first handle; the main slot holds only the second.
    app = TauTuiApp(FakeSession())

    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        bridge = _component_bridge(app)

        first = bridge.open_main_view(lambda handle, theme: Static("one", id="ext-view-one"))
        await pilot.pause()
        second = bridge.open_main_view(lambda handle, theme: Static("two", id="ext-view-two"))
        await pilot.pause()

        # Opening a second view closes the first handle and leaves exactly one
        # child mounted in the main slot.
        assert not first.is_open
        assert second.is_open
        assert app._extension_main_view is second
        slot = app.query_one("#main-slot", Container)
        assert len(slot.children) == 1
        assert slot.query("#ext-view-two")
        assert not slot.query("#ext-view-one")


@pytest.mark.anyio
async def test_component_slot_replace_same_id_same_tick_no_duplicate() -> None:
    # [FUNCTION DOC]
    # Async test: removing and re-adding a same-id slot widget in one tick leaves one widget, no
    # reported failures, and the mounted widget equals the tracked one.
    """Replacing a slot widget with a same-id widget in one tick must sequence.

    The old widget's ``remove()`` is deferred; mounting the replacement before it
    drains would leave two widgets sharing an id (``DuplicateIds``). The swap
    must await the removal, so exactly one widget survives with no failure.
    """
    app = TauTuiApp(FakeSession())

    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        bridge = _component_bridge(app)

        bridge.set_slot_widget(
            "k", lambda theme: Static("one", id="dup-slot"), placement="below_prompt"
        )
        await pilot.pause()
        # Same tick: unmount then re-register the SAME id (the reload analog).
        bridge.set_slot_widget("k", None, placement="below_prompt")
        bridge.set_slot_widget(
            "k", lambda theme: Static("two", id="dup-slot"), placement="below_prompt"
        )
        await pilot.pause()
        await pilot.pause()

        slot = app.query_one("#below-prompt-slot", Container)
        assert len(slot.query("#dup-slot")) == 1
        assert app._extension_component_failures_reported == set()
        assert app._extension_slot_mounted["k"] is app._extension_slot_widgets["k"]


@pytest.mark.anyio
async def test_component_slot_rapid_a_b_c_last_wins() -> None:
    # [FUNCTION DOC]
    # Async test: three same-id slot swaps in one tick leave only the last-built widget mounted.
    """Three same-id slot swaps in one tick collapse to the last, no duplicates."""
    app = TauTuiApp(FakeSession())

    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        bridge = _component_bridge(app)

        made: dict[str, Static] = {}

        def factory(label: str):  # noqa: ANN202
            # [FUNCTION DOC]
            # HELPER: Return a widget factory for the label that records each built widget in the shared map.
            def build(theme):  # noqa: ANN001, ANN202
                # [FUNCTION DOC]
                # HELPER: Build a Static with the label and a fixed id, record it and return it (theme ignored).
                widget = made[label] = Static(label, id="abc-slot")
                return widget

            return build

        bridge.set_slot_widget("k", factory("pre"), placement="below_prompt")
        await pilot.pause()
        bridge.set_slot_widget("k", factory("A"), placement="below_prompt")
        bridge.set_slot_widget("k", factory("B"), placement="below_prompt")
        bridge.set_slot_widget("k", factory("C"), placement="below_prompt")
        await pilot.pause()
        await pilot.pause()

        slot = app.query_one("#below-prompt-slot", Container)
        surviving = slot.query("#abc-slot")
        assert len(surviving) == 1
        assert surviving.first() is made["C"]
        assert app._extension_component_failures_reported == set()


@pytest.mark.anyio
async def test_component_second_main_view_same_id_no_duplicate() -> None:
    # [FUNCTION DOC]
    # Async test: opening a second same-id main view before the first drains leaves one mounted view,
    # which is the second handle's widget, with no failures.
    """Opening a second same-id main view while one is open must not collide."""
    app = TauTuiApp(FakeSession())

    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        bridge = _component_bridge(app)

        bridge.open_main_view(lambda handle, theme: Static("one", id="dup-view"))
        await pilot.pause()
        # A second view sharing the viewer id, opened before the first drains.
        second = bridge.open_main_view(lambda handle, theme: Static("two", id="dup-view"))
        await pilot.pause()
        await pilot.pause()

        slot = app.query_one("#main-slot", Container)
        assert len(slot.query("#dup-view")) == 1
        assert app._extension_main_view is second
        assert app._extension_main_view_mounted is second.widget
        assert app._extension_component_failures_reported == set()


@pytest.mark.anyio
async def test_component_main_view_rapid_a_b_c_last_wins() -> None:
    # [FUNCTION DOC]
    # Async test: three same-id main-view opens in one tick mount only the last; earlier handles are closed.
    """Three same-id main-view opens in one tick mount exactly the last one."""
    app = TauTuiApp(FakeSession())

    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        bridge = _component_bridge(app)

        bridge.open_main_view(lambda handle, theme: Static("pre", id="abc-view"))
        await pilot.pause()
        a = bridge.open_main_view(lambda handle, theme: Static("A", id="abc-view"))
        b = bridge.open_main_view(lambda handle, theme: Static("B", id="abc-view"))
        c = bridge.open_main_view(lambda handle, theme: Static("C", id="abc-view"))
        await pilot.pause()
        await pilot.pause()

        slot = app.query_one("#main-slot", Container)
        surviving = slot.query("#abc-view")
        assert len(surviving) == 1
        assert surviving.first() is c.widget
        assert app._extension_main_view is c
        assert not a.is_open
        assert not b.is_open
        assert c.is_open
        assert app._extension_component_failures_reported == set()


@pytest.mark.anyio
async def test_component_host_exception_is_not_swallowed() -> None:
    # [FUNCTION DOC]
    # Async test: with an extension widget tracked, a core RuntimeError passed to the app's exception
    # handler reaches Textual's default handling (stored on the app) and is re-raised by run_test on exit.
    app = TauTuiApp(FakeSession())

    with pytest.raises(RuntimeError, match="core-bug"):
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            bridge = _component_bridge(app)
            bridge.set_slot_widget("k", lambda theme: Static("x", id="ext-host"))
            await pilot.pause()
            # An extension widget is tracked, so the quarantine path is live.
            assert app._tracked_extension_widgets()

            # A core/host exception whose traceback touches no extension widget
            # must reach Textual's default handler rather than being quarantined
            # and swallowed. run_test re-raises the stored exception on exit.
            try:
                raise RuntimeError("core-bug")
            except RuntimeError as exc:
                app._handle_exception(exc)
            assert app._exception is not None  # reached super()._handle_exception
            assert "core-bug" in str(app._exception)


@pytest.mark.anyio
async def test_component_main_view_open_close_restores_transcript() -> None:
    # [FUNCTION DOC]
    # Async test: opening a main view hides the transcript and shows the main slot; closing the handle
    # restores the transcript, hides the slot and focuses the prompt input.
    app = TauTuiApp(FakeSession())

    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        bridge = _component_bridge(app)

        # Opening a main view hides #transcript and shows only #main-slot.
        handle = bridge.open_main_view(lambda h, theme: Static("main", id="ext-main-view"))
        await pilot.pause()
        displayed = [
            pane_id for pane_id in ("#transcript", "#main-slot") if app.query_one(pane_id).display
        ]
        assert displayed == ["#main-slot"]

        # Closing restores #transcript, hides the slot, and refocuses the prompt.
        handle.close()
        await pilot.pause()
        assert app.query_one("#transcript", TranscriptView).display
        assert not app.query_one("#main-slot", Container).display
        assert app.query_one("#prompt", PromptInput).has_focus


@pytest.mark.anyio
async def test_sidebar_sections_mount_update_and_remove_by_owned_key() -> None:
    # [FUNCTION DOC]
    # Async test: sidebar sections mount in order, an identical update keeps the same widget, a changed
    # update rewrites title/body in place, and removal leaves only the remaining section.
    app = TauTuiApp(FakeSession())

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        bridge = _component_bridge(app)

        bridge.set_sidebar_section("alpha", "status", title="Alpha", content=["first"])
        bridge.set_sidebar_section("beta", "status", title="Beta", content=["second"])
        await pilot.pause()

        slot = app.query_one("#sidebar-extension-sections", Container)
        assert len(slot.children) == 2
        assert [
            section.query_one(".extension-sidebar-title", Static).render().plain
            for section in slot.children
        ] == [
            "Alpha",
            "Beta",
        ]

        alpha_section = slot.children[0]
        bridge.set_sidebar_section("alpha", "status", title="Alpha", content=["first"])
        await pilot.pause()
        assert slot.children[0] is alpha_section

        bridge.set_sidebar_section("alpha", "status", title="Alpha updated", content=["updated"])
        await pilot.pause()

        assert slot.children[0] is alpha_section
        assert [
            section.query_one(".extension-sidebar-title", Static).render().plain
            for section in slot.children
        ] == [
            "Alpha updated",
            "Beta",
        ]
        alpha_body = slot.children[0].query_one(".extension-sidebar-body", Container)
        assert alpha_body.query_one(Static).render().plain == "updated"

        bridge.remove_sidebar_section("alpha", "status")
        await pilot.pause()
        await pilot.pause()
        assert len(slot.children) == 1
        assert (
            slot.children[0].query_one(".extension-sidebar-title", Static).render().plain == "Beta"
        )


@pytest.mark.anyio
async def test_sidebar_removing_absent_key_does_not_schedule_reconciliation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # [FUNCTION DOC]
    # Async test: removing an unknown sidebar section does not schedule a swap coroutine.
    app = TauTuiApp(FakeSession())

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        scheduled = False

        def schedule(coro):  # noqa: ANN001, ANN202
            # [FUNCTION DOC]
            # HELPER: Replacement scheduler that flags it was called and closes the coroutine unrun.
            nonlocal scheduled
            scheduled = True
            coro.close()

        monkeypatch.setattr(app, "_schedule_extension_swap", schedule)
        _component_bridge(app).remove_sidebar_section("ext", "absent")

        assert scheduled is False


@pytest.mark.anyio
async def test_sidebar_same_local_key_is_isolated_between_extensions() -> None:
    # [FUNCTION DOC]
    # Async test: the same local key from two extensions yields two distinct tracked sections.
    app = TauTuiApp(FakeSession())

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        bridge = _component_bridge(app)
        bridge.set_sidebar_section("one", "status", title="One", content=["1"])
        bridge.set_sidebar_section("two", "status", title="Two", content=["2"])
        await pilot.pause()

        assert list(app._extension_sidebar_widgets) == [("one", "status"), ("two", "status")]


@pytest.mark.anyio
async def test_sidebar_remove_then_readd_moves_section_to_end() -> None:
    # [FUNCTION DOC]
    # Async test: removing and re-adding a section in the same tick moves it to the end of the order.
    app = TauTuiApp(FakeSession())

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        bridge = _component_bridge(app)
        bridge.set_sidebar_section("ext", "a", title="A", content=["a"])
        bridge.set_sidebar_section("ext", "b", title="B", content=["b"])
        await pilot.pause()
        bridge.remove_sidebar_section("ext", "a")
        bridge.set_sidebar_section("ext", "a", title="A", content=["a"])
        await pilot.pause()
        await pilot.pause()

        slot = app.query_one("#sidebar-extension-sections", Container)
        assert [
            section.query_one(".extension-sidebar-title", Static).render().plain
            for section in slot.children
        ] == [
            "B",
            "A",
        ]


@pytest.mark.anyio
async def test_sidebar_off_is_unsupported_and_does_not_call_factory() -> None:
    # [FUNCTION DOC]
    # Async test: with the sidebar position "off", the bridge reports no sidebar support, the content
    # factory is never called, nothing is tracked and the sidebar is hidden.
    app = TauTuiApp(FakeSession(), tui_settings=TuiSettings(sidebar_position="off"))
    called = False

    def factory(theme):  # noqa: ANN001, ANN202
        # [FUNCTION DOC]
        # HELPER: Content factory that flags it was called and returns a Static.
        nonlocal called
        called = True
        return Static("hidden")

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        bridge = _component_bridge(app)
        assert bridge.supports_sidebar is False
        bridge.set_sidebar_section("ext", "status", title="Status", content=factory)
        await pilot.pause()

        assert called is False
        assert app._extension_sidebar_widgets == {}
        assert app.query_one("#sidebar").display is False


@pytest.mark.anyio
async def test_responsive_hidden_sidebar_keeps_sections_for_resize() -> None:
    # [FUNCTION DOC]
    # Async test: at narrow width the sidebar is hidden but sections stay mounted; after resizing wider
    # the sidebar is displayed with sections intact.
    app = TauTuiApp(FakeSession())

    async with app.run_test(size=(80, 30)) as pilot:
        await pilot.pause()
        bridge = _component_bridge(app)
        assert bridge.supports_sidebar is True
        bridge.set_sidebar_section("ext", "status", title="Status", content=["ready"])
        await pilot.pause()
        assert app.query_one("#sidebar").display is False
        assert app.query_one("#sidebar-extension-sections", Container).children

        await pilot.resize_terminal(width=120, height=40)
        await pilot.pause()
        assert app.query_one("#sidebar").display is True
        assert app.query_one("#sidebar-extension-sections", Container).children


@pytest.mark.anyio
@pytest.mark.parametrize("position", ["left", "right"])
async def test_sidebar_sections_work_in_both_sidebar_positions(position: str) -> None:
    # [FUNCTION DOC]
    # Async parametrized test: sections mount for left and right sidebar positions; the app has the
    # right-sidebar class only for "right".
    app = TauTuiApp(FakeSession(), tui_settings=TuiSettings(sidebar_position=position))

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        bridge = _component_bridge(app)
        bridge.set_sidebar_section("ext", "status", title="Status", content=["ready"])
        await pilot.pause()

        assert app.query_one("#sidebar-extension-sections", Container).children
        assert app.has_class("-sidebar-right") is (position == "right")


@pytest.mark.anyio
async def test_sidebar_factory_rebuilds_with_live_theme(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # [FUNCTION DOC]
    # Async test: a sidebar content factory is rebuilt with the new theme when the TUI theme changes
    # (dark then light). Isolates HOME so theme persistence writes to tmp_path.
    isolate_home(monkeypatch, tmp_path)
    app = TauTuiApp(FakeSession())
    themes: list[str] = []

    def factory(theme):  # noqa: ANN001, ANN202
        # [FUNCTION DOC]
        # HELPER: Factory recording the theme name and returning a Static showing it.
        themes.append(theme.name)
        return Static(theme.name, classes="theme-body")

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        bridge = _component_bridge(app)
        bridge.set_sidebar_section("ext", "theme", title="Theme", content=factory)
        await pilot.pause()
        app._set_tui_theme("tau-light")
        await pilot.pause()
        await pilot.pause()

        assert themes == ["tau-dark", "tau-light"]
        assert (
            app.query_one("#sidebar-extension-sections .theme-body", Static).render().plain
            == "tau-light"
        )


@pytest.mark.anyio
async def test_sidebar_factory_failure_keeps_app_running() -> None:
    # [FUNCTION DOC]
    # Async test: a sidebar factory that raises is reported as a component failure, not tracked, and
    # the app keeps running.
    app = TauTuiApp(FakeSession())

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        bridge = _component_bridge(app)

        def explode(theme):  # noqa: ANN001, ANN202
            # [FUNCTION DOC]
            # HELPER: Factory that always raises RuntimeError.
            raise RuntimeError("sidebar factory exploded")

        bridge.set_sidebar_section("ext", "bad", title="Bad", content=explode)
        await pilot.pause()

        assert app.is_running
        assert app._extension_sidebar_widgets == {}
        assert "sidebar:ext:bad" in app._extension_component_failures_reported


class _SidebarRenderCrash(Static):
    # [CLASS DOC]
    # HELPER: Static widget whose render always raises, used to simulate an extension render crash.
    def render(self):  # noqa: ANN201
        # [FUNCTION DOC]
        # HELPER: Raise RuntimeError("sidebar render exploded").
        raise RuntimeError("sidebar render exploded")


@pytest.mark.anyio
async def test_sidebar_render_failure_is_quarantined_and_diagnosed() -> None:
    # [FUNCTION DOC]
    # Async test: a sidebar widget crashing at render time is quarantined (untracked) while the app
    # keeps running, and a diagnostic naming the extension and section is added to the runtime.
    session = FakeSession()
    runtime = ExtensionRuntime()
    session.extension_runtime = runtime
    app = TauTuiApp(session)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        bridge = _component_bridge(app)
        bridge.set_sidebar_section(
            "broken-extension",
            "bad",
            title="Bad",
            content=lambda theme: _SidebarRenderCrash("bad"),
        )
        await pilot.pause()
        await pilot.pause()

        assert app.is_running
        assert ("broken-extension", "bad") not in app._extension_sidebar_widgets
        assert any(
            diagnostic.name == "broken-extension"
            and "sidebar:broken-extension:bad" in diagnostic.message
            for diagnostic in runtime.diagnostics
        )


@pytest.mark.anyio
async def test_component_clear_removes_sidebar_sections() -> None:
    # [FUNCTION DOC]
    # Async test: clearing components removes all sidebar contributions, tracked/mounted widgets and children.
    app = TauTuiApp(FakeSession())

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        bridge = _component_bridge(app)
        bridge.set_sidebar_section("ext", "status", title="Status", content=["ready"])
        await pilot.pause()
        assert app.query_one("#sidebar-extension-sections", Container).children

        bridge.clear_components()
        await pilot.pause()
        await pilot.pause()

        assert app._extension_sidebar_contributions == {}
        assert app._extension_sidebar_widgets == {}
        assert app._extension_sidebar_mounted == {}
        assert not app.query_one("#sidebar-extension-sections", Container).children
