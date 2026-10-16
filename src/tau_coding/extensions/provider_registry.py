# [MODULE DOC]
# Process-local, generation-scoped registry for extension-defined (dynamic)
# model providers. Composes immutable durable provider configs with layered
# dynamic definitions registered per source (extension); the most recently
# registered layer wins. Supports coalesced, timeout-bounded, cancellable model
# list refreshes whose snapshots are published only if the layer is still
# current, bounded secret-free diagnostics, and retirement/close that cancels
# and drains background discovery tasks. A module-level set keeps cancelled
# discovery tasks strongly referenced until they actually finish.
"""Generation-local layered registry for extension-defined providers."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Awaitable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from functools import partial
from itertools import count
from os import environ
from types import MappingProxyType
from typing import Literal
from uuid import uuid4

from tau_agent.harness import SimpleCancellationToken
from tau_coding.extensions.providers import (
    CredentialReader,
    DynamicProvider,
    ProviderModelSnapshot,
    ProviderRefreshContext,
    resolve_provider_auth,
)
from tau_coding.provider_config import ProviderConfig

DEFAULT_PROVIDER_REFRESH_TIMEOUT_SECONDS = 10.0
PROVIDER_DISCOVERY_CANCELLATION_TIMEOUT_SECONDS = 0.25
MAX_PROVIDER_REFRESH_DIAGNOSTICS = 100

# asyncio keeps only weak task references. This process-owned supervisor keeps
# cancellation-requested discovery and its generation registry strongly
# reachable through cooperative drain or bounded containment.
_SUPERVISED_DISCOVERY_TASKS: set[asyncio.Task[ProviderModelSnapshot]] = set()


@dataclass(frozen=True, slots=True)
class ProviderLayerToken:
    # [CLASS DOC]
    # Frozen identity of one registered layer: provider id, source id, generation id
    # and a unique layer id. Required to publish into exactly that layer.
    """Identity required to publish work into one exact source layer."""

    provider_id: str
    source_id: str
    generation_id: str
    layer_id: str


@dataclass(frozen=True, slots=True)
class DynamicProviderLayer:
    # [CLASS DOC]
    # Frozen record of one registered dynamic provider with its token and global
    # registration order.
    """One registered dynamic provider definition and its ownership token."""

    token: ProviderLayerToken
    provider: DynamicProvider
    registration_order: int


@dataclass(frozen=True, slots=True)
class EffectiveProvider:
    # [CLASS DOC]
    # Frozen view of the definition currently in effect for a provider id: either a
    # durable config or a dynamic provider, its source id ("durable" for baseline),
    # and the layer token when dynamic.
    """The complete effective provider definition for one provider id."""

    definition: ProviderConfig | DynamicProvider = field(repr=False)
    source_id: str
    layer_token: ProviderLayerToken | None = None

    @property
    def dynamic(self) -> bool:
        # [FUNCTION DOC]
        # Property: true when the effective definition comes from a dynamic layer.
        """Return whether the effective definition is process-local."""
        return self.layer_token is not None


@dataclass(frozen=True, slots=True)
class ProviderRefreshDiagnostic:
    # [CLASS DOC]
    # Frozen diagnostic for a failed refresh: layer token, reason (cancelled,
    # failed, timed_out) and a fixed message that never includes exception text.
    """Bounded secret-free diagnostic for one failed layer generation."""

    token: ProviderLayerToken
    reason: Literal["cancelled", "failed", "timed_out"]
    message: str


@dataclass(frozen=True, slots=True)
class ProviderRefreshResult:
    # [CLASS DOC]
    # Frozen refresh outcome: status (published, unavailable, cancelled, failed,
    # timed_out, stale), the provider definition now current (if any) and the token.
    """Outcome returned to one refresh caller."""

    status: Literal["published", "unavailable", "cancelled", "failed", "timed_out", "stale"]
    provider: DynamicProvider | None = field(repr=False)
    token: ProviderLayerToken | None


@dataclass(frozen=True, slots=True)
class ProviderRegistryCloseResult:
    # [CLASS DOC]
    # Frozen close outcome: whether all work drained and how many discovery tasks
    # were still running (contained) after the bounded wait.
    """Whether close drained callbacks or boundedly contained remaining work."""

    drained: bool
    contained_discovery_tasks: int = 0


@dataclass(eq=False, slots=True)
class _RefreshOperation:
    # [CLASS DOC]
    # HELPER: mutable record of one shared refresh: its cancellation signal, task,
    # number of waiting callers and whether task cancellation was already requested.
    # Compared by identity.
    """One owned refresh operation, independently of coalescing eligibility."""

    signal: SimpleCancellationToken
    task: asyncio.Task[ProviderRefreshResult]
    waiters: int = 0
    cancellation_requested: bool = False


class _NoCredentials:
    # [CLASS DOC]
    # HELPER: credential reader that knows no credentials.
    def get(self, name: str) -> str | None:
        # [FUNCTION DOC]
        # HELPER: ignore the name and return nothing.
        del name
        return None


class DynamicProviderRegistry:
    # [CLASS DOC]
    # Registry for one runtime generation. Holds the durable providers by name,
    # generation id, credential reader, environment mapping, per-provider ordered
    # layer lists, counters, coalesced refresh operations keyed by (token, network
    # policy), refresh revisions per token, discovery tasks with cancellation
    # deadlines, bounded diagnostics, and a retired flag. Nothing is persisted.
    """Compose immutable durable baselines with generation-owned dynamic layers.

    The registry is process-local. It has no persistence methods: dynamic
    definitions and refresh snapshots can only live for this registry's runtime
    generation.
    """

    def __init__(
        self,
        durable_providers: Sequence[ProviderConfig] = (),
        *,
        generation_id: str | None = None,
        credentials: CredentialReader | None = None,
        environment: Mapping[str, str] | None = None,
    ) -> None:
        # [FUNCTION DOC]
        # Index durable providers by name (ValueError on duplicate ids), set the
        # generation id (random hex when omitted), credentials (empty reader by default)
        # and environment (process environment by default), and initialize empty state.
        durable: dict[str, ProviderConfig] = {}
        for provider in durable_providers:
            if provider.name in durable:
                raise ValueError(f"Duplicate durable provider id: {provider.name}")
            durable[provider.name] = provider
        self._durable = MappingProxyType(durable)
        self._generation_id = generation_id or uuid4().hex
        self._credentials = credentials if credentials is not None else _NoCredentials()
        self._environment = environment if environment is not None else environ
        self._layers: dict[str, list[DynamicProviderLayer]] = {}
        self._order = count(1)
        self._layer_sequence = count(1)
        self._refresh_operations: dict[tuple[ProviderLayerToken, bool], _RefreshOperation] = {}
        self._owned_refresh_operations: set[_RefreshOperation] = set()
        self._refresh_revisions: dict[ProviderLayerToken, int] = {}
        self._discovery_tasks: set[asyncio.Task[ProviderModelSnapshot]] = set()
        self._discovery_cancellation_deadlines: dict[
            asyncio.Task[ProviderModelSnapshot], float
        ] = {}
        self._diagnostics: dict[ProviderLayerToken, ProviderRefreshDiagnostic] = {}
        self._retired = False

    @property
    def generation_id(self) -> str:
        # [FUNCTION DOC]
        # Property: this registry's generation identity.
        """Return this staged registry's generation identity."""
        return self._generation_id

    @property
    def credentials(self) -> CredentialReader:
        # [FUNCTION DOC]
        # Property: the credential reader used for provider auth.
        """Return the credential reader shared with runtime construction."""
        return self._credentials

    @property
    def environment(self) -> Mapping[str, str]:
        # [FUNCTION DOC]
        # Property: the environment mapping used for provider auth.
        """Return the environment snapshot shared with runtime construction."""
        return self._environment

    @property
    def durable_providers(self) -> tuple[ProviderConfig, ...]:
        # [FUNCTION DOC]
        # Property: tuple of the durable provider configs in insertion order.
        """Return the exact complete durable baseline objects."""
        return tuple(self._durable.values())

    @property
    def diagnostics(self) -> tuple[ProviderRefreshDiagnostic, ...]:
        # [FUNCTION DOC]
        # Property: tuple of recorded refresh diagnostics in insertion order.
        """Return bounded diagnostics in deterministic insertion order."""
        return tuple(self._diagnostics.values())

    def register(self, source_id: str, provider: DynamicProvider) -> ProviderLayerToken:
        # [FUNCTION DOC]
        # Register or replace the layer for (provider id, source). Raises RuntimeError
        # when retired and ValueError for an invalid source id. Creates a new token,
        # removes any existing layer from the same source, appends the new layer last
        # (highest precedence) and cancels refresh work of replaced layers. Returns the
        # new token.
        """Atomically register or replace one source's complete provider layer."""
        self._assert_active()
        normalized_source = _source_id(source_id)
        # Provider construction has already performed complete validation. Do
        # not mutate current state before this point.
        token = ProviderLayerToken(
            provider_id=provider.id,
            source_id=normalized_source,
            generation_id=self._generation_id,
            layer_id=f"{self._generation_id}:{next(self._layer_sequence)}",
        )
        layer = DynamicProviderLayer(
            token=token,
            provider=provider,
            registration_order=next(self._order),
        )
        existing = self._layers.get(provider.id, ())
        replaced = [item for item in existing if item.token.source_id == normalized_source]
        self._layers[provider.id] = [
            item for item in existing if item.token.source_id != normalized_source
        ] + [layer]
        for old_layer in replaced:
            self._cancel_token(old_layer.token)
        return token

    def unregister(self, provider_id: str, source_id: str) -> bool:
        # [FUNCTION DOC]
        # Remove the given source's layer for a provider, dropping the provider entry if
        # no layers remain, and cancel its refresh work. Returns whether anything was
        # removed. Raises ValueError for an invalid source id; works even when retired.
        """Remove one source layer while preserving all other definitions."""
        normalized_source = _source_id(source_id)
        layers = self._layers.get(provider_id)
        if not layers:
            return False
        removed = [item for item in layers if item.token.source_id == normalized_source]
        if not removed:
            return False
        remaining = [item for item in layers if item.token.source_id != normalized_source]
        if remaining:
            self._layers[provider_id] = remaining
        else:
            self._layers.pop(provider_id, None)
        for layer in removed:
            self._cancel_token(layer.token)
        return True

    def update(self, source_id: str, provider: DynamicProvider) -> bool:
        # [FUNCTION DOC]
        # Replace the provider definition of an existing source layer in place, keeping
        # its token and order (so in-flight work stays valid). Raises RuntimeError when
        # retired. Returns false if no such layer exists.
        """Publish a new definition without invalidating paired local backends.

        Snapshot updates are not source-layer replacements: preserving the
        provider token lets a backend operation finish while its provider's
        model list changes. Registration remains the API for adding or
        replacing a source layer and retains its cancellation semantics.
        """
        self._assert_active()
        normalized_source = _source_id(source_id)
        layers = self._layers.get(provider.id)
        if not layers:
            return False
        for index, layer in enumerate(layers):
            if layer.token.source_id == normalized_source:
                layers[index] = DynamicProviderLayer(
                    token=layer.token,
                    provider=provider,
                    registration_order=layer.registration_order,
                )
                return True
        return False

    def unregister_source(self, source_id: str) -> None:
        # [FUNCTION DOC]
        # Unregister the source's layers from every provider id (cancelling their work).
        """Remove every layer and cancel every task owned by one source."""
        normalized_source = _source_id(source_id)
        for provider_id in tuple(self._layers):
            self.unregister(provider_id, normalized_source)

    def effective(self, provider_id: str) -> EffectiveProvider | None:
        # [FUNCTION DOC]
        # Return the effective provider: the last dynamic layer if any, else the durable
        # config with source "durable", else nothing.
        """Return the latest complete active layer, then the durable baseline."""
        layers = self._layers.get(provider_id)
        if layers:
            latest = layers[-1]
            return EffectiveProvider(
                definition=latest.provider,
                source_id=latest.token.source_id,
                layer_token=latest.token,
            )
        durable = self._durable.get(provider_id)
        if durable is None:
            return None
        return EffectiveProvider(definition=durable, source_id="durable")

    def effective_providers(self) -> tuple[EffectiveProvider, ...]:
        # [FUNCTION DOC]
        # Return effective providers for all ids: durable ids first in order, then
        # dynamic-only ids in insertion order. Pure.
        """Return a deterministic composed view without modifying durable settings."""
        ids = list(self._durable)
        ids.extend(provider_id for provider_id in self._layers if provider_id not in self._durable)
        return tuple(
            effective
            for provider_id in ids
            if (effective := self.effective(provider_id)) is not None
        )

    def layers(self, provider_id: str) -> tuple[DynamicProviderLayer, ...]:
        # [FUNCTION DOC]
        # Return the provider's active dynamic layers, lowest to highest precedence.
        """Return active dynamic layers in precedence order."""
        return tuple(self._layers.get(provider_id, ()))

    def layer_token(self, provider_id: str, source_id: str) -> ProviderLayerToken | None:
        # [FUNCTION DOC]
        # Return the token of the given source's layer for the provider, or nothing.
        # Note: the source id is compared as given, without normalization.
        """Return one source's exact active layer identity, if registered."""
        for layer in self._layers.get(provider_id, ()):
            if layer.token.source_id == source_id:
                return layer.token
        return None

    async def refresh(
        self,
        provider_id: str,
        *,
        allow_network: bool = True,
        timeout_seconds: float = DEFAULT_PROVIDER_REFRESH_TIMEOUT_SECONDS,
    ) -> ProviderRefreshResult:
        # [FUNCTION DOC]
        # Refresh the effective dynamic layer's model list. Raises ValueError for a
        # non-boolean network policy or non-positive/non-finite timeout. Returns
        # "unavailable" when retired, missing, durable, or lacking a refresh callback.
        # Otherwise joins an existing live operation with the same token and network
        # policy or starts a new task (bumping the token's revision). Waits on it
        # shielded under the caller's own timeout: returns the operation's result, or
        # on timeout records a diagnostic and returns "timed_out"; when the last waiter
        # times out the shared operation is detached and cancelled. Caller cancellation
        # does not cancel the shared work. Side effects: tasks, state mutation.
        """Refresh the effective dynamic layer under caller-specific policy.

        Callers coalesce only when their layer token and network policy match,
        while each keeps its own timeout. Caller cancellation does not cancel
        shared work; generation retirement, source removal, replacement, or
        :meth:`cancel_refresh` owns cancellation.
        """
        if not isinstance(allow_network, bool):
            raise ValueError("Provider refresh network policy must be a boolean")
        if (
            not isinstance(timeout_seconds, int | float)
            or isinstance(timeout_seconds, bool)
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("Provider refresh timeout must be greater than 0")
        effective = self.effective(provider_id)
        if (
            self._retired
            or effective is None
            or not isinstance(effective.definition, DynamicProvider)
            or effective.layer_token is None
            or effective.definition.refresh_models is None
        ):
            provider = (
                effective.definition
                if effective is not None and isinstance(effective.definition, DynamicProvider)
                else None
            )
            return ProviderRefreshResult(
                "unavailable",
                provider,
                effective.layer_token if effective else None,
            )

        token = effective.layer_token
        key = (token, allow_network)
        operation = self._refresh_operations.get(key)
        if operation is not None and (operation.task.done() or operation.task.cancelling()):
            self._detach_refresh_operation(key, operation)
            operation = None
        if operation is None:
            signal = SimpleCancellationToken()
            revision = self._refresh_revisions.get(token, 0) + 1
            self._refresh_revisions[token] = revision
            task = asyncio.create_task(
                self._run_refresh(
                    token,
                    effective.definition,
                    signal,
                    allow_network=allow_network,
                    revision=revision,
                ),
                name=f"tau-provider-refresh:{token.source_id}:{provider_id}",
            )
            operation = _RefreshOperation(signal=signal, task=task)
            self._refresh_operations[key] = operation
            self._owned_refresh_operations.add(operation)
            task.add_done_callback(partial(self._finish_refresh, key, operation))
        operation.waiters += 1
        timed_out = False
        try:
            async with asyncio.timeout(timeout_seconds):
                return await asyncio.shield(operation.task)
        except TimeoutError:
            timed_out = True
            self._record_diagnostic(token, "timed_out", "provider model refresh timed out")
            return ProviderRefreshResult("timed_out", self._current_provider(token), token)
        finally:
            operation.waiters -= 1
            if timed_out and operation.waiters == 0:
                # Detach before returning so an immediate retry cannot join an
                # already-cancelling operation. The owned-operation set keeps
                # the old task supervised until its actual completion.
                self._cancel_operation(key, operation)

    def cancel_refresh(self, provider_id: str, source_id: str | None = None) -> bool:
        # [FUNCTION DOC]
        # Cancel refresh operations for all layers of the provider, or only the given
        # source's layer. Returns whether any operation was cancelled.
        """Cancel refresh work for matching active provider layers."""
        cancelled = False
        for layer in self._layers.get(provider_id, ()):
            if source_id is None or layer.token.source_id == source_id:
                cancelled = self._cancel_token(layer.token) or cancelled
        return cancelled

    def retire(self) -> None:
        # [FUNCTION DOC]
        # Idempotently mark the generation retired, clear all layers, cancel all keyed
        # and owned refresh operations and request cancellation of discovery tasks.
        # Synchronous; requires a running loop if discovery tasks exist.
        """Synchronously invalidate this generation and cancel all owned work."""
        if self._retired:
            return
        self._retired = True
        self._layers.clear()
        for key, operation in tuple(self._refresh_operations.items()):
            self._cancel_operation(key, operation)
        for operation in tuple(self._owned_refresh_operations):
            self._request_refresh_cancellation(operation)
        for task in tuple(self._discovery_tasks):
            self._request_discovery_cancellation(task)

    async def aclose(self) -> ProviderRegistryCloseResult:
        # [FUNCTION DOC]
        # Retire, await all owned refresh tasks (ignoring their exceptions), then cancel
        # and wait for discovery tasks up to the containment deadline. Returns whether
        # everything drained and how many discovery tasks are still running.
        """Retire, drain cooperative work, and report bounded containment.

        Discovery receives at most one task cancellation. Close waits only for
        the remainder of the fixed containment window measured from that first
        request. A callback still running afterward is not reported as drained;
        a process-owned supervisor retains its task and generation owner until
        actual completion.
        """
        self.retire()
        refresh_tasks = tuple(operation.task for operation in self._owned_refresh_operations)
        if refresh_tasks:
            await asyncio.gather(*refresh_tasks, return_exceptions=True)
        discovery_tasks = tuple(self._discovery_tasks)
        if discovery_tasks:
            await asyncio.gather(
                *(self._cancel_and_drain_discovery(task) for task in discovery_tasks),
                return_exceptions=True,
            )
        contained = sum(not task.done() for task in self._discovery_tasks)
        return ProviderRegistryCloseResult(
            drained=contained == 0,
            contained_discovery_tasks=contained,
        )

    async def _run_refresh(
        self,
        token: ProviderLayerToken,
        provider: DynamicProvider,
        signal: SimpleCancellationToken,
        *,
        allow_network: bool,
        revision: int,
    ) -> ProviderRefreshResult:
        # [FUNCTION DOC]
        # HELPER: body of a refresh task. Starts a discovery task (tracked, with a done
        # callback) and awaits it shielded; validates the snapshot type (TypeError
        # otherwise) and builds a candidate provider with it. On cancellation, cancels
        # the signal, drains discovery and returns "cancelled" with a diagnostic; on
        # any other exception returns "failed" with a categorical diagnostic. Returns
        # "stale" if the signal was cancelled, the token is no longer current, or a newer
        # revision started; otherwise replaces that layer's provider with the candidate
        # and returns "published".
        callback_task: asyncio.Task[ProviderModelSnapshot] | None = None
        try:
            callback_task = asyncio.create_task(
                self._discover_snapshot(
                    provider,
                    signal,
                    allow_network=allow_network,
                ),
                name=f"tau-provider-discovery:{token.source_id}:{token.provider_id}",
            )
            self._discovery_tasks.add(callback_task)
            callback_task.add_done_callback(self._finish_discovery)
            snapshot = await asyncio.shield(callback_task)
            if not isinstance(snapshot, ProviderModelSnapshot):
                raise TypeError("refresh must return ProviderModelSnapshot")
            candidate = provider.with_snapshot(snapshot)
        except asyncio.CancelledError:
            signal.cancel()
            if callback_task is not None:
                await self._cancel_and_drain_discovery(callback_task)
            self._record_diagnostic(token, "cancelled", "provider model refresh was cancelled")
            return ProviderRefreshResult("cancelled", self._current_provider(token), token)
        except Exception:
            # Arbitrary extension exception text may contain request data or a
            # secret. Keep diagnostics categorical and never include repr(exc).
            self._record_diagnostic(token, "failed", "provider model refresh failed")
            return ProviderRefreshResult("failed", self._current_provider(token), token)

        if (
            signal.is_cancelled()
            or not self._token_is_current(token)
            or self._refresh_revisions.get(token) != revision
        ):
            return ProviderRefreshResult("stale", self._current_provider(token), token)
        layers = self._layers[token.provider_id]
        self._layers[token.provider_id] = [
            DynamicProviderLayer(
                token=layer.token,
                provider=candidate,
                registration_order=layer.registration_order,
            )
            if layer.token == token
            else layer
            for layer in layers
        ]
        return ProviderRefreshResult("published", candidate, token)

    async def _discover_snapshot(
        self,
        provider: DynamicProvider,
        signal: SimpleCancellationToken,
        *,
        allow_network: bool,
    ) -> ProviderModelSnapshot:
        # [FUNCTION DOC]
        # HELPER: resolve the provider's auth from credentials/environment, build a
        # refresh context (signal, network policy, cached models, auth) and await the
        # provider's refresh callback, returning its snapshot. May raise anything the
        # callback raises.
        auth = await resolve_provider_auth(
            provider.auth,
            credentials=self._credentials,
            environment=self._environment,
        )
        callback = provider.refresh_models
        assert callback is not None
        context = ProviderRefreshContext(
            signal=signal,
            allow_network=allow_network,
            cached_models=provider.models,
            auth=auth,
        )
        return await _await_snapshot(callback(context))

    def _token_is_current(self, token: ProviderLayerToken) -> bool:
        # [FUNCTION DOC]
        # HELPER: true when not retired, the token belongs to this generation and it is
        # still among the provider's active layers.
        if self._retired or token.generation_id != self._generation_id:
            return False
        return any(layer.token == token for layer in self._layers.get(token.provider_id, ()))

    def _current_provider(self, token: ProviderLayerToken) -> DynamicProvider | None:
        # [FUNCTION DOC]
        # HELPER: return the provider of the active layer with this token, or nothing.
        for layer in self._layers.get(token.provider_id, ()):
            if layer.token == token:
                return layer.provider
        return None

    def _cancel_token(self, token: ProviderLayerToken) -> bool:
        # [FUNCTION DOC]
        # HELPER: cancel every refresh operation for the token (any network policy) and
        # forget its revision. Returns whether any were cancelled.
        cancelled = False
        keys = {key for key in self._refresh_operations if key[0] == token}
        for key in keys:
            cancelled = self._cancel_operation(key) or cancelled
        self._refresh_revisions.pop(token, None)
        return cancelled

    def _cancel_operation(
        self,
        key: tuple[ProviderLayerToken, bool],
        expected: _RefreshOperation | None = None,
    ) -> bool:
        # [FUNCTION DOC]
        # HELPER: if the key maps to an operation (and to the expected one, when given),
        # detach it so new callers cannot join and request its cancellation. Returns
        # whether it did so.
        operation = self._refresh_operations.get(key)
        if operation is None or (expected is not None and operation is not expected):
            return False
        # Coalescing eligibility ends synchronously, before cancellation can
        # return to its caller. Identity checks keep an old done callback from
        # detaching a successor created under the same key.
        self._detach_refresh_operation(key, operation)
        self._request_refresh_cancellation(operation)
        return True

    @staticmethod
    def _request_refresh_cancellation(operation: _RefreshOperation) -> None:
        # [FUNCTION DOC]
        # HELPER: cancel the operation's signal and, once only, cancel its task if not done.
        operation.signal.cancel()
        if operation.cancellation_requested:
            return
        operation.cancellation_requested = True
        if not operation.task.done():
            operation.task.cancel()

    def _detach_refresh_operation(
        self,
        key: tuple[ProviderLayerToken, bool],
        operation: _RefreshOperation,
    ) -> None:
        # [FUNCTION DOC]
        # HELPER: remove the key only if it still maps to this exact operation.
        if self._refresh_operations.get(key) is operation:
            self._refresh_operations.pop(key, None)

    def _finish_refresh(
        self,
        key: tuple[ProviderLayerToken, bool],
        operation: _RefreshOperation,
        task: asyncio.Task[ProviderRefreshResult],
    ) -> None:
        # [FUNCTION DOC]
        # HELPER: done callback for a refresh task: detach the operation, drop it from
        # the owned set and retrieve the task result so no "never retrieved" warning occurs.
        self._detach_refresh_operation(key, operation)
        self._owned_refresh_operations.discard(operation)
        _consume_refresh_task_result(task)

    def _finish_discovery(self, task: asyncio.Task[ProviderModelSnapshot]) -> None:
        # [FUNCTION DOC]
        # HELPER: done callback for a discovery task: untrack it, drop its deadline and
        # retrieve its result.
        self._discovery_tasks.discard(task)
        self._discovery_cancellation_deadlines.pop(task, None)
        _consume_task_result(task)

    def _request_discovery_cancellation(
        self,
        task: asyncio.Task[ProviderModelSnapshot],
    ) -> float:
        # [FUNCTION DOC]
        # HELPER: request cancellation of a discovery task once, returning its deadline
        # (loop time plus the containment timeout). The first request records the
        # deadline, adds the task to the process-wide supervised set (released on
        # completion) and cancels it. Requires a running event loop.
        deadline = self._discovery_cancellation_deadlines.get(task)
        if deadline is not None:
            return deadline
        loop = asyncio.get_running_loop()
        deadline = loop.time() + PROVIDER_DISCOVERY_CANCELLATION_TIMEOUT_SECONDS
        self._discovery_cancellation_deadlines[task] = deadline
        # A callback may still be in ordinary finally cleanup when the bounded
        # close interval ends. Give it a process-rooted supervisor before the
        # cancellation request so dropping the outgoing runtime cannot let the
        # still-pending task and its generation owner be garbage-collected.
        _SUPERVISED_DISCOVERY_TASKS.add(task)
        task.add_done_callback(_release_supervised_discovery)
        task.cancel()
        return deadline

    async def _cancel_and_drain_discovery(
        self,
        task: asyncio.Task[ProviderModelSnapshot],
    ) -> bool:
        # [FUNCTION DOC]
        # HELPER: request cancellation and wait, shielded, only until the deadline.
        # Returns true if the task finished (including with an error) and false if it is
        # still running at the deadline. Re-raises if the caller itself is cancelled
        # while the task is still pending.
        """Request cancellation once and await only the containment remainder."""
        if task.done():
            return True
        deadline = self._request_discovery_cancellation(task)
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            return task.done()
        try:
            async with asyncio.timeout(remaining):
                await asyncio.shield(task)
        except TimeoutError:
            return task.done()
        except asyncio.CancelledError:
            if task.done():
                return True
            # This owner was itself cancelled. Shielding keeps the callback
            # supervised and avoids injecting a second cancellation into its
            # potentially cooperative finally cleanup.
            raise
        except Exception:
            return True
        return True

    def _record_diagnostic(
        self,
        token: ProviderLayerToken,
        reason: Literal["cancelled", "failed", "timed_out"],
        message: str,
    ) -> None:
        # [FUNCTION DOC]
        # HELPER: record at most one diagnostic per token, evicting the oldest when the
        # maximum count is reached.
        if token in self._diagnostics:
            return
        if len(self._diagnostics) >= MAX_PROVIDER_REFRESH_DIAGNOSTICS:
            oldest = next(iter(self._diagnostics))
            self._diagnostics.pop(oldest)
        self._diagnostics[token] = ProviderRefreshDiagnostic(token, reason, message)

    def _assert_active(self) -> None:
        # [FUNCTION DOC]
        # HELPER: raise RuntimeError if the registry generation is retired.
        if self._retired:
            raise RuntimeError("Dynamic provider registry generation is retired")


async def _await_snapshot(
    value: Awaitable[ProviderModelSnapshot],
) -> ProviderModelSnapshot:
    # [FUNCTION DOC]
    # HELPER: await an awaitable snapshot so it can be wrapped in a task.
    return await value


def _release_supervised_discovery(task: asyncio.Task[ProviderModelSnapshot]) -> None:
    # [FUNCTION DOC]
    # HELPER: done callback removing the task from the process-wide supervised set.
    """Release the process-owned strong reference after actual completion."""
    _SUPERVISED_DISCOVERY_TASKS.discard(task)


def _consume_refresh_task_result(task: asyncio.Task[ProviderRefreshResult]) -> None:
    # [FUNCTION DOC]
    # HELPER: retrieve a finished refresh task's exception (ignoring cancellation)
    # to mark it as observed.
    """Retrieve an owned refresh result after all publication logic completed."""
    with suppress(asyncio.CancelledError):
        task.exception()


def _consume_task_result(task: asyncio.Task[ProviderModelSnapshot]) -> None:
    # [FUNCTION DOC]
    # HELPER: retrieve a finished discovery task's exception (ignoring
    # cancellation) to mark it as observed.
    """Retrieve a detached cancelled discovery result without publishing it."""
    with suppress(asyncio.CancelledError):
        task.exception()


def _source_id(value: str) -> str:
    # [FUNCTION DOC]
    # HELPER: validate a source id: must be a non-empty string without surrounding
    # whitespace; raises ValueError otherwise. Returns it unchanged.
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Provider source id must be a non-empty string")
    normalized = value.strip()
    if normalized != value:
        raise ValueError("Provider source id must not have surrounding whitespace")
    return normalized
