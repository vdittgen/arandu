"""Retrieval pipeline — hybrid vector + BM25 + fusion.

Phase 4 of the embedding/vector-search overhaul. Sits above the
:mod:`src.core.chromadb` engine and below the brain agent's
``recall_context`` tool.

sensitivity_tier: 2 (handles query text and document IDs)
"""
