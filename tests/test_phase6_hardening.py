# [MODULE DOC]
# Phase 6 hardening regression tests for extension runtime lifecycle: dynamic provider
# model refresh coalescing and generation retirement, local-backend operation
# cancellation bounds and replacement staleness, and llama.cpp integration state
# persistence (stale model references, exact model names, secrets kept out of state
# files and reprs). All async tests run on the asyncio backend via anyio.
"""Phase 6 lifecycle, race, ownership, and persistence regressions."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest

from tau_agent.harness import SimpleCancellationToken
from tau_coding.credentials import FileCredentialStore
from tau_coding.extensions import (
    DynamicProvider,
    DynamicProviderRegistry,
    ExtensionRuntime,
    LocalBackend,
    LocalBackendRegistry,
    LocalBackendStatus,
    LocalConfigureResult,
    LocalConfigureSpec,
    LocalModel,
    LocalOperationContext,
    NoAuth,
    OpenAICompatibleTransport,
    ProviderModel,
    ProviderModelSnapshot,
)
from tau_coding.extensions.builtins.llama_cpp.service import (
    LlamaCppService,
)
from tau_coding.extensions.builtins.llama_cpp.state import (
    LLAMA_CPP_CREDENTIAL_PREFIX,
    LlamaCppIntegrationState,
    LlamaCppStateError,
    LlamaCppStateStore,
    LlamaCppStoredModel,
)
from tau_coding.local_backends import (
    LOCAL_OPERATION_CANCELLATION_TIMEOUT_SECONDS,
    LocalOperationResult,
)
from tau_coding.paths import TauPaths

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    # [FUNCTION DOC]
    # HELPER: Fixture selecting the asyncio backend for anyio-marked tests.
    return "asyncio"


def _provider(provider_id: str = "second", *, refresh_models=None) -> DynamicProvider:
    # [FUNCTION DOC]
    # HELPER: Build a DynamicProvider with the given id (default "second") exposing one
    # model "model" as default, an OpenAI-compatible transport at a fake base URL with no
    # auth, and an optional refresh-models callback.
    return DynamicProvider(
        id=provider_id,
        display_name=provider_id,
        models=(ProviderModel("model"),),
        default_model="model",
        transport=OpenAICompatibleTransport(
            base_url="http://second.test/v1",
            auth=NoAuth(),
        ),
        refresh_models=refresh_models,
    )


def _context(action: str, backend_id: str = "second") -> LocalOperationContext:
    # [FUNCTION DOC]
    # HELPER: Build a LocalOperationContext for the given action and backend id with a fresh
    # cancellation token, fixed generation id "generation", source "test:second", an
    # always-current predicate and a no-op progress callback.
    return LocalOperationContext(
        signal=SimpleCancellationToken(),
        action=action,  # type: ignore[arg-type]
        generation_id="generation",
        backend_id=backend_id,
        source_id="test:second",
        _is_current=lambda: True,
        _progress=lambda _: None,
    )


async def test_refresh_stress_keeps_latest_snapshot_and_closes_generation() -> None:
    # [FUNCTION DOC]
    # Twenty concurrent refreshes of one provider coalesce into a single callback call; all
    # report "published", the effective provider exposes "model-1", and closing the runtime
    # removes the effective provider.
    calls = 0
    entered = asyncio.Event()
    release = asyncio.Event()

    async def refresh(context) -> ProviderModelSnapshot:
        # [FUNCTION DOC]
        # HELPER: Refresh callback that counts calls, signals entry, waits for release, and
        # returns a snapshot with a model named after the call count.
        nonlocal calls
        calls += 1
        entered.set()
        await release.wait()
        model = f"model-{calls}"
        return ProviderModelSnapshot((ProviderModel(model),), model)

    runtime = ExtensionRuntime()
    registry = runtime.provider_registry
    registry.register("test:second", _provider(refresh_models=refresh))
    waiters = [asyncio.create_task(registry.refresh("second")) for _ in range(20)]
    await entered.wait()
    release.set()
    results = await asyncio.gather(*waiters)

    assert calls == 1
    assert {result.status for result in results} == {"published"}
    assert registry.effective("second").definition.models[0].id == "model-1"  # type: ignore[union-attr]
    await runtime.aclose()
    assert registry.effective("second") is None


async def test_refresh_from_retired_generation_cannot_publish() -> None:
    # [FUNCTION DOC]
    # Re-registers the provider with a blocking refresh callback, starts a refresh, resets
    # the runtime for reload while it is blocked, then releases it: the result is
    # "cancelled" and nothing is published.
    entered = asyncio.Event()
    release = asyncio.Event()

    async def refresh(context) -> ProviderModelSnapshot:
        # [FUNCTION DOC]
        # HELPER: Refresh callback that signals entry, blocks until released, returns a "stale"
        # snapshot.
        entered.set()
        await release.wait()
        return ProviderModelSnapshot((ProviderModel("stale"),), "stale")

    runtime = ExtensionRuntime()
    registry = runtime.provider_registry
    registry.register("test:second", _provider())
    # Replace the callback while keeping the test provider's public shape.
    current = registry.effective("second").definition  # type: ignore[union-attr]
    registry.register(
        "test:second",
        DynamicProvider(
            id=current.id,
            display_name=current.display_name,
            models=current.models,
            default_model=current.default_model,
            transport=current.transport,
            refresh_models=refresh,
        ),
    )
    pending = asyncio.create_task(registry.refresh("second"))
    await entered.wait()
    runtime.reset_for_reload()
    release.set()
    result = await pending

    assert result.status == "cancelled"
    assert registry.effective("second") is None
    await runtime.aclose()


async def test_local_retire_bounds_cancellation_resistant_backend() -> None:
    # [FUNCTION DOC]
    # A local backend operation that swallows cancellation keeps running; closing the
    # registry must still finish within roughly the cancellation timeout (under 1 second,
    # not before half the timeout), leaving the operation pending; once released, its result
    # is marked stale.
    providers = DynamicProviderRegistry(generation_id="generation")
    providers.register("test:second", _provider())
    registry = LocalBackendRegistry(providers, generation_id="generation")
    entered = asyncio.Event()
    release = asyncio.Event()

    async def operation(context: LocalOperationContext) -> LocalOperationResult:
        # [FUNCTION DOC]
        # HELPER: Backend status/refresh operation that ignores CancelledError and loops until
        # the release event is set, then returns a ready status.
        entered.set()
        while not release.is_set():
            try:
                await release.wait()
            except asyncio.CancelledError:
                # Simulate a backend finishing network cleanup later. The
                # registry must not inject another cancellation into this path.
                continue
        return LocalOperationResult(backend_status=LocalBackendStatus(state="ready"))

    backend = LocalBackend(
        id="second",
        provider_id="second",
        display_name="Second",
        configure_spec=LocalConfigureSpec(),
        configure=lambda values, context: LocalConfigureResult(),
        status=operation,
        refresh=operation,
    )
    registry.register("test:second", backend)
    pending = asyncio.create_task(registry.refresh("second"))
    await entered.wait()

    started = asyncio.get_running_loop().time()
    close_task = asyncio.create_task(registry.aclose())
    await asyncio.sleep(LOCAL_OPERATION_CANCELLATION_TIMEOUT_SECONDS / 2)
    assert not close_task.done()
    await close_task
    elapsed = asyncio.get_running_loop().time() - started
    assert elapsed < 1.0
    assert pending.done() is False

    release.set()
    result = await asyncio.wait_for(pending, timeout=0.5)
    assert result.stale is True
    await providers.aclose()


async def test_local_replacement_discards_late_result_and_new_operation_wins() -> None:
    # [FUNCTION DOC]
    # Replacing a backend registration while its old refresh is running makes the old
    # result stale; a subsequent refresh uses the new backend and reports model "new".
    providers = DynamicProviderRegistry(generation_id="generation")
    providers.register("test:second", _provider())
    registry = LocalBackendRegistry(providers, generation_id="generation")
    old_started = asyncio.Event()
    old_release = asyncio.Event()

    async def old_refresh(context: LocalOperationContext) -> LocalBackendStatus:
        # [FUNCTION DOC]
        # HELPER: Blocking refresh returning a ready status with model "old" after release.
        old_started.set()
        await old_release.wait()
        return LocalBackendStatus(state="ready", models=(LocalModel("old"),))

    async def new_refresh(context: LocalOperationContext) -> LocalBackendStatus:
        # [FUNCTION DOC]
        # HELPER: Immediate refresh returning a ready status with model "new".
        return LocalBackendStatus(state="ready", models=(LocalModel("new"),))

    def backend(refresh):
        # [FUNCTION DOC]
        # HELPER: Build a LocalBackend "second" whose status and refresh both use the given
        # callback and whose configure returns an empty result.
        return LocalBackend(
            id="second",
            provider_id="second",
            display_name="Second",
            configure_spec=LocalConfigureSpec(),
            configure=lambda values, context: LocalConfigureResult(),
            status=refresh,
            refresh=refresh,
        )

    registry.register("test:second", backend(old_refresh))
    old_task = asyncio.create_task(registry.refresh("second"))
    await old_started.wait()
    registry.register("test:second", backend(new_refresh))
    old_release.set()
    assert (await old_task).stale is True
    fresh = await registry.refresh("second")
    assert fresh.backend_status is not None
    assert fresh.backend_status.models[0].id == "new"
    await registry.aclose()
    await providers.aclose()


async def test_missing_model_reference_survives_refresh_and_restart(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Saves llama.cpp state whose selected model is absent from the server's model list.
    # With a mock HTTP server (health ok, models list without it), the provider has no
    # default model, refresh reports stale status while preserving the stored selection, and
    # a restarted service (fresh state store) still reports stale with no default model.
    endpoint = "http://local.test:8080"
    paths = TauPaths(home=tmp_path / "tau", agents_home=tmp_path / "agents")
    state_store = LlamaCppStateStore(paths=paths)
    state_store.save(
        LlamaCppIntegrationState(
            endpoint=endpoint,
            selected_model="returning-model",
            models=(LlamaCppStoredModel("other-model"),),
        )
    )
    credentials = FileCredentialStore(paths.home / "credentials.json")
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={"status": "ok"}
                if request.url.path == "/health"
                else {"data": [{"id": "other-model"}]},
            )
        )
    )
    service = LlamaCppService(
        state_store=state_store,
        credential_store=credentials,
        environment={},
        client=client,
    )
    assert service.provider().default_model is None
    result = await service.refresh(_context("refresh", "llama.cpp"))
    assert result.backend_status is not None
    assert result.backend_status.stale is True
    assert state_store.active().selected_model == "returning-model"  # type: ignore[union-attr]

    restarted = LlamaCppService(
        state_store=LlamaCppStateStore(paths=paths),
        credential_store=credentials,
        environment={},
        client=client,
    )
    assert restarted.provider().default_model is None
    restarted_status = await restarted.status(_context("refresh", "llama.cpp"))
    assert restarted_status.stale is True
    assert restarted.provider().default_model is None
    await client.aclose()


@pytest.mark.parametrize("selected_model", ["", " returning-model"])
def test_state_rejects_non_exact_stale_model_references(selected_model: str) -> None:
    # [FUNCTION DOC]
    # Integration state construction raises LlamaCppStateError ("non-empty exact") for an
    # empty or whitespace-padded selected model.
    with pytest.raises(LlamaCppStateError, match="non-empty exact"):
        LlamaCppIntegrationState(endpoint="http://local.test", selected_model=selected_model)


async def test_secret_state_and_diagnostics_never_contain_key(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Configuring an endpoint with an API key while the server is offline stores the key
    # only in the credential store under the llama.cpp prefix; the state file, the result
    # repr and the provider repr never contain it, and the state file references it by
    # credential ref.
    secret = "phase-six-secret"
    paths = TauPaths(home=tmp_path / "tau", agents_home=tmp_path / "agents")
    state_store = LlamaCppStateStore(paths=paths)
    credentials = FileCredentialStore(paths.home / "credentials.json")

    async def dispatch(request: httpx.Request) -> httpx.Response:
        # [FUNCTION DOC]
        # HELPER: Mock transport handler that always raises a connection error ("offline").
        raise httpx.ConnectError("offline", request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(dispatch))
    service = LlamaCppService(
        state_store=state_store,
        credential_store=credentials,
        environment={},
        client=client,
    )
    result = await service.configure(
        {"endpoint": "http://127.0.0.1:8080", "api_key": secret},
        _context("configure", "llama.cpp"),
    )
    serialized = state_store.path.read_text(encoding="utf-8")
    assert secret not in serialized
    assert secret not in repr(result)
    assert secret not in repr(service.provider())
    refs = credentials.names(prefix=LLAMA_CPP_CREDENTIAL_PREFIX)
    assert len(refs) == 1
    assert credentials.get(refs[0]) == secret
    payload = json.loads(serialized)
    assert payload["endpoints"]["http://127.0.0.1:8080"]["credential_ref"] == refs[0]
    await client.aclose()
