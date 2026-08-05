"""Vector-index migration CLI.

Rebuilds every ChromaDB collection under a new embedding model.
Dimension is hard-coupled to the model: once a collection has been
written with 768-dim vectors (``nomic-embed-text``), querying it
with 1024-dim vectors (``bge-m3``) returns garbage. Swap → rebuild.

Re-embedding is recoverable from the DuckDB raw_* tables (that's the
authoritative copy), so this CLI takes the simple path: drop every
collection, write a new ``.embedding_meta.json``, then call
:meth:`src.core.chromadb.indexer.Indexer.full_reindex`. No dual-write
because doubling the disk footprint isn't worth it when the source
is intact.

Safety:

* Defaults to dry-run. Pass ``--apply`` to actually wipe collections.
* For remote embedders, prints a token + dollar estimate before any
  remote API call so you can cancel if a fresh user's mailbox would
  cost more than expected.
* Refuses to run when the Ollama backend is missing the requested
  model — preflight error rather than thousands of failed embeds.

Usage::

    python -m src.core.chromadb.migrate --to-model bge-m3
    python -m src.core.chromadb.migrate --to-model bge-m3 --apply
    python -m src.core.chromadb.migrate --to-model text-embedding-3-large \\
        --provider openai --apply

sensitivity_tier: 3 (re-embeds all raw user data)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.core.chromadb.engine import COLLECTION_NAMES, DEFAULT_DB_PATH, VectorEngine
from src.core.chromadb.indexer import Indexer
from src.core.chromadb.meta import current_meta, read_meta, write_meta
from src.core.sqlite.engine import DatabaseEngine
from src.models.embedding_provider import (
    MODEL_DIMENSIONS,
    EmbeddingProvider,
    OllamaEmbeddingProvider,
    OpenAIEmbeddingProvider,
)

logger = logging.getLogger(__name__)

# Rough token-per-character heuristic (matches indexer.APPROX_CHARS_PER_TOKEN).
APPROX_CHARS_PER_TOKEN = 4

# OpenAI text-embedding-3-large list price (as of 2026).
OPENAI_EMBED_USD_PER_M_TOKENS = 0.13


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------


def preflight_ollama(model: str) -> None:
    """Confirm the requested model is pulled in Ollama.

    sensitivity_tier: N/A
    """
    try:
        import ollama
    except ImportError as exc:
        raise SystemExit(
            f"ollama SDK unavailable: {exc}",
        ) from exc
    try:
        listing = ollama.list()
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            f"could not query ollama: {exc}\n"
            f"is ollama running? try `ollama serve`",
        ) from exc
    names = {m.get("name") or m.get("model") for m in listing.get("models", [])}
    if model not in names and f"{model}:latest" not in names:
        raise SystemExit(
            f"ollama does not have '{model}' pulled.\n"
            f"run: `ollama pull {model}` and retry.",
        )


def estimate_corpus_size(duck: DatabaseEngine) -> tuple[int, int]:
    """Sum of (rows, characters) across embedded tables.

    Counts the same tables the indexer reads from. Approximation is
    good enough for an order-of-magnitude cost estimate.

    sensitivity_tier: 1
    """
    queries = [
        ("raw_messages", "SELECT COALESCE(SUM(LENGTH(content)),0) AS c, "
         "COUNT(*) AS n FROM raw_messages"),
        ("raw_calendar_events", "SELECT COALESCE(SUM(LENGTH(COALESCE(title,'')) "
         "+ LENGTH(COALESCE(description,''))),0) AS c, COUNT(*) AS n "
         "FROM raw_calendar_events"),
        ("raw_notes", "SELECT COALESCE(SUM(LENGTH(COALESCE(title,''))+"
         "LENGTH(COALESCE(content,''))),0) AS c, COUNT(*) AS n FROM raw_notes"),
        ("raw_contacts", "SELECT COALESCE(SUM(LENGTH(COALESCE(name,''))+"
         "LENGTH(COALESCE(notes,''))),0) AS c, COUNT(*) AS n FROM raw_contacts"),
        ("raw_health_metrics", "SELECT COALESCE(SUM(LENGTH(COALESCE(metric_type,'')) "
         "+ 32),0) AS c, COUNT(*) AS n FROM raw_health_metrics"),
    ]
    rows = 0
    chars = 0
    for _, sql in queries:
        try:
            res = duck.query(sql)
        except Exception as exc:  # noqa: BLE001
            logger.debug("estimate query failed: %s", exc)
            continue
        if res:
            rows += int(res[0].get("n") or 0)
            chars += int(res[0].get("c") or 0)
    return rows, chars


def format_cost_estimate(
    rows: int, chars: int, provider: str,
) -> str:
    """Render a human-readable cost line.

    sensitivity_tier: N/A
    """
    tokens = chars // APPROX_CHARS_PER_TOKEN
    if provider == "remote_openai" or provider == "openai":
        usd = (tokens / 1_000_000) * OPENAI_EMBED_USD_PER_M_TOKENS
        return (
            f"~{rows:,} records / ~{tokens:,} tokens — "
            f"OpenAI cost estimate: ~${usd:.2f}"
        )
    return (
        f"~{rows:,} records / ~{tokens:,} tokens — "
        f"local provider (no $ cost, but expect a few minutes)"
    )


# ---------------------------------------------------------------------------
# Build the target provider
# ---------------------------------------------------------------------------


def build_target_provider(
    model: str,
    provider_kind: str,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    dimensions: int | None = None,
) -> EmbeddingProvider:
    """Instantiate the destination embedding provider.

    sensitivity_tier: 1
    """
    if provider_kind == "ollama":
        return OllamaEmbeddingProvider(model=model)
    if provider_kind == "openai":
        if not api_key:
            raise SystemExit(
                "--provider openai needs --api-key (or OPENAI_API_KEY env var)",
            )
        return OpenAIEmbeddingProvider(
            api_key=api_key,
            model=model,
            base_url=base_url,
            dimensions=dimensions,
        )
    raise SystemExit(f"unknown --provider: {provider_kind}")


# ---------------------------------------------------------------------------
# Core migration
# ---------------------------------------------------------------------------


def drop_all_collections(engine: VectorEngine) -> dict[str, int]:
    """Delete every domain collection so it can be recreated under the new dim.

    ChromaDB stores the expected vector dimension on the collection
    itself — ``col.delete(ids=...)`` clears the contents but the dim
    sticks, so upserts under a different model fail with
    ``InvalidArgumentError: Collection expecting embedding with dimension``.
    We use ``client.delete_collection`` instead so the next
    ``get_or_create_collection`` rebuilds it with the new embedder.

    Returns ``{collection: previous_count}`` for the audit log.

    sensitivity_tier: 3
    """
    previous: dict[str, int] = {}
    client = engine._client  # noqa: SLF001 — migration is a privileged op
    for name in COLLECTION_NAMES:
        try:
            col = client.get_collection(name)
        except Exception:  # noqa: BLE001
            # Collection doesn't exist yet — nothing to drop.
            previous[name] = 0
            continue
        previous[name] = col.count()
        client.delete_collection(name)
    return previous


def migrate(
    *,
    target_model: str,
    target_provider_kind: str,
    db_path: Path,
    api_key: str | None,
    base_url: str | None,
    dimensions: int | None,
    apply: bool,
) -> int:
    """Run the migration. Returns process exit code.

    sensitivity_tier: 3
    """
    if target_provider_kind == "ollama":
        preflight_ollama(target_model)

    target = build_target_provider(
        model=target_model,
        provider_kind=target_provider_kind,
        api_key=api_key,
        base_url=base_url,
        dimensions=dimensions,
    )

    # Probe dimension. For known models this is free; for unknown,
    # this triggers one embed call (cost: negligible).
    try:
        target_dim = target.dimension
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"failed to probe target model dimension: {exc}\n")
        return 2

    stored = read_meta(db_path)
    sys.stdout.write(
        f"current index:  {stored.provider}/{stored.model_name} "
        f"(dim={stored.dimension})\n"
        if stored else
        "current index:  (no sentinel — first migration)\n",
    )
    sys.stdout.write(
        f"target:         {target.provider_name}/{target.model_name} "
        f"(dim={target_dim})\n",
    )

    duck = DatabaseEngine()
    rows, chars = estimate_corpus_size(duck)
    sys.stdout.write(
        format_cost_estimate(rows, chars, target.provider_name) + "\n",
    )

    if not apply:
        sys.stdout.write(
            "\nDRY-RUN. Re-run with --apply to actually wipe and rebuild.\n",
        )
        return 0

    sys.stdout.write("\napplying migration…\n")
    engine = VectorEngine(db_path=db_path, embedding_fn=target)
    previous = drop_all_collections(engine)
    sys.stdout.write(
        f"dropped: {sum(previous.values())} docs across "
        f"{len([k for k, v in previous.items() if v])} collections\n",
    )

    # Wipe the BM25 FTS5 mirror so it stays in lockstep with chroma.
    # The dual-write in indexer._upsert_documents then refills it
    # alongside the vector reindex below.
    from src.core.retrieval import bm25

    try:
        bm25.init_table(duck)
        fts_removed = bm25.clear(duck)
        sys.stdout.write(f"dropped: {fts_removed} bm25 rows\n")
    except Exception as exc:  # noqa: BLE001
        sys.stdout.write(f"bm25 clear skipped: {exc}\n")

    # Write the meta *before* the long reindex so a crash mid-reindex
    # leaves the sentinel pointing at the model the partial vectors
    # were built with (rather than the old model that no longer matches).
    write_meta(
        db_path,
        current_meta(target.provider_name, target.model_name, target_dim),
    )

    indexer = Indexer(duckdb=duck, chromadb=engine)
    counts = indexer.full_reindex()
    sys.stdout.write(f"reindex complete: {counts}\n")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Entrypoint for ``python -m src.core.chromadb.migrate``.

    sensitivity_tier: N/A
    """
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild ChromaDB collections under a new embedding model. "
            "Dimension changes require a full rebuild — the source data "
            "lives in DuckDB raw_* tables and is replayed unchanged."
        ),
    )
    parser.add_argument(
        "--to-model", required=True,
        help="Target model name (e.g. bge-m3, text-embedding-3-large).",
    )
    parser.add_argument(
        "--provider", choices=("ollama", "openai"), default="ollama",
        help="Which provider hosts --to-model (default: ollama).",
    )
    parser.add_argument(
        "--api-key", default=None,
        help="Required for --provider openai. Avoid embedding in shell history.",
    )
    parser.add_argument(
        "--base-url", default=None,
        help="OpenAI-compatible endpoint override.",
    )
    parser.add_argument(
        "--dimensions", type=int, default=None,
        help=(
            "Optional dimensionality truncation (OpenAI 3-large only). "
            "Useful to match a local model's footprint."
        ),
    )
    parser.add_argument(
        "--db-path", type=Path, default=DEFAULT_DB_PATH,
        help=f"ChromaDB directory (default: {DEFAULT_DB_PATH}).",
    )
    parser.add_argument(
        "--apply", action="store_true",
        help="Actually run the migration. Without this flag, dry-run only.",
    )
    args = parser.parse_args(argv)

    expected_dim = MODEL_DIMENSIONS.get(args.to_model)
    if expected_dim is None:
        sys.stderr.write(
            f"warn: '{args.to_model}' not in MODEL_DIMENSIONS — "
            f"dimension will be probed at runtime.\n",
        )

    return migrate(
        target_model=args.to_model,
        target_provider_kind=args.provider,
        db_path=args.db_path,
        api_key=args.api_key,
        base_url=args.base_url,
        dimensions=args.dimensions,
        apply=args.apply,
    )


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
