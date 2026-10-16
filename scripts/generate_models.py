#!/usr/bin/env python3
# [MODULE DOC]
# Developer script that regenerates Tau's bundled model catalog JSON from the
# models.dev API (following Pi's generator). Reads a models.dev source and an
# NVIDIA models source (URLs or local fixtures), merges them with the built-in
# source catalog, and writes the JSON snapshot (default
# src/tau_coding/data/models-dev-catalog.json). Run as a CLI.
"""Generate Tau's bundled model catalog from models.dev, following Pi."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from urllib.request import Request, urlopen

from tau_coding.catalog_loader import builtin_source_catalog
from tau_coding.models_dev import (
    MODELS_DEV_URL,
    NVIDIA_MODELS_URL,
    models_dev_catalog_document,
    nvidia_model_filter,
)

DEFAULT_OUTPUT = Path("src/tau_coding/data/models-dev-catalog.json")


def _load_source(location: str) -> object:
    # [FUNCTION DOC]
    # HELPER: load JSON from a location. If the location starts with http:// or
    # https://, performs an HTTP GET with a custom User-Agent and 30 s timeout and
    # parses the response body; otherwise opens it as a local UTF-8 file.
    # Input: location string. Output: the parsed JSON value.
    # Raises network errors (URLError/HTTPError), OSError for files, and JSON
    # decode errors. Side effects: network or file I/O.
    if location.startswith(("http://", "https://")):
        request = Request(location, headers={"User-Agent": "tau-model-catalog-generator"})
        with urlopen(request, timeout=30) as response:  # noqa: S310 - explicit build input
            return json.load(response)
    with Path(location).open(encoding="utf-8") as source_file:
        return json.load(source_file)


def main() -> None:
    # [FUNCTION DOC]
    # CLI entry point. Parses arguments: source (models.dev URL/file),
    # nvidia-source (NVIDIA models URL/file), generated-at (Unix ms, default now),
    # and output path. Loads both sources, builds the catalog document via
    # models_dev_catalog_document with the built-in source catalog and an NVIDIA
    # model filter, creates the output directory, writes pretty-printed sorted JSON
    # with a trailing newline, and prints a summary of models/providers written.
    # Returns nothing. Propagates load/write errors; argparse exits on bad args.
    # Note: a generated-at of 0 is treated as unset and replaced with now.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        default=MODELS_DEV_URL,
        help="models.dev api.json URL or a local JSON fixture",
    )
    parser.add_argument(
        "--nvidia-source",
        default=NVIDIA_MODELS_URL,
        help="NVIDIA /models URL or local JSON fixture used by Pi's live filter",
    )
    parser.add_argument(
        "--generated-at",
        type=int,
        default=None,
        help="snapshot generation time in Unix milliseconds (default: now)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"generated JSON path (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    source = _load_source(args.source)
    document = models_dev_catalog_document(
        source,
        builtin_source_catalog(),
        provider_model_filters={
            "nvidia": nvidia_model_filter(source, _load_source(args.nvidia_source))
        },
        generated_at=args.generated_at or int(time.time() * 1000),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    providers = document["providers"]
    model_count = sum(len(provider["models"]) for provider in providers.values())
    print(f"Wrote {model_count} models for {len(providers)} providers to {args.output}")


if __name__ == "__main__":
    main()
