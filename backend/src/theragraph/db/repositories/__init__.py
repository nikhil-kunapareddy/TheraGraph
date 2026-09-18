"""Repositories: reusable, tested query logic.

Services orchestrate; repositories own SQL. Keeping queries here means the N+1s,
the index-friendly predicates and the upsert semantics live in one place that the
database test suite can exercise directly.
"""
