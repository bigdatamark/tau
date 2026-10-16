# [PACKAGE DOC]
# Package for the trusted built-in llama.cpp local backend extension.
# Re-exports the public service/state API (endpoint normalization, discovery,
# service, state store, constants) and provides the extension setup hook that
# registers a llama.cpp provider and a provider-neutral local backend with the
# extension API.
"""Trusted built-in llama.cpp local backend."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from tau_coding.built_in_extensions import BuiltInExtensionContext
from tau_coding.credentials import FileCredentialStore, credentials_path
from tau_coding.paths import TauPaths

from .router import LLAMA_CPP_ROUTER_MAX_BUILD, LLAMA_CPP_ROUTER_MIN_BUILD
from .service import (
    DEFAULT_LLAMA_CPP_TIMEOUT_SECONDS,
    LLAMA_CPP_API_KEY_ENV,
    LLAMA_CPP_BACKEND_ID,
    LLAMA_CPP_DEFAULT_ENDPOINT,
    LLAMA_CPP_DISPLAY_NAME,
    LLAMA_CPP_ENDPOINT_ENV,
    LLAMA_CPP_PROVIDER_ID,
    LlamaCppDiscovery,
    LlamaCppEndpoint,
    LlamaCppEndpointError,
    LlamaCppError,
    LlamaCppService,
    normalize_llama_cpp_endpoint,
)
from .state import (
    LLAMA_CPP_CREDENTIAL_PREFIX,
    LLAMA_CPP_STATE_SCHEMA_VERSION,
    LlamaCppIntegrationState,
    LlamaCppStateError,
    LlamaCppStateStore,
    LlamaCppStoredModel,
)

if TYPE_CHECKING:
    from tau_coding.extensions.api import ExtensionAPI


def setup(
    tau: ExtensionAPI,
    context: BuiltInExtensionContext | None = None,
) -> None:
    # [FUNCTION DOC]
    # Extension entry point. Inputs: the extension API object and an optional
    # built-in extension context. When no context is given, builds a default one
    # from default Tau paths, a file credential store at the default credentials
    # path, and a copy of the process environment. Constructs a LlamaCppService
    # wired to a state store, credential store, environment, optional HTTP client
    # and the API's register/update provider and register-local-backend callbacks,
    # then registers the service's provider and local backend.
    # Returns nothing. Side effects: registers into the extension API; may read
    # the environment. Exceptions from registration propagate.
    """Register the trusted provider and provider-neutral local backend."""
    dependencies = context or BuiltInExtensionContext(
        paths=TauPaths(),
        credential_store=FileCredentialStore(credentials_path(TauPaths())),
        environment=dict(os.environ),
    )
    service = LlamaCppService(
        state_store=LlamaCppStateStore(paths=dependencies.paths),
        credential_store=dependencies.credential_store,
        environment=dependencies.environment,
        client=dependencies.http_client,
        register_provider=tau.register_provider,
        update_provider=tau.update_provider,
        register_backend=tau.register_local_backend,
    )
    tau.register_provider(service.provider())
    tau.register_local_backend(service.backend())


__all__ = [
    "DEFAULT_LLAMA_CPP_TIMEOUT_SECONDS",
    "LLAMA_CPP_API_KEY_ENV",
    "LLAMA_CPP_CREDENTIAL_PREFIX",
    "LLAMA_CPP_BACKEND_ID",
    "LLAMA_CPP_DEFAULT_ENDPOINT",
    "LLAMA_CPP_DISPLAY_NAME",
    "LLAMA_CPP_ENDPOINT_ENV",
    "LLAMA_CPP_PROVIDER_ID",
    "LLAMA_CPP_ROUTER_MAX_BUILD",
    "LLAMA_CPP_ROUTER_MIN_BUILD",
    "LLAMA_CPP_STATE_SCHEMA_VERSION",
    "LlamaCppDiscovery",
    "LlamaCppEndpoint",
    "LlamaCppEndpointError",
    "LlamaCppError",
    "LlamaCppIntegrationState",
    "LlamaCppService",
    "LlamaCppStateError",
    "LlamaCppStateStore",
    "LlamaCppStoredModel",
    "normalize_llama_cpp_endpoint",
    "setup",
]
