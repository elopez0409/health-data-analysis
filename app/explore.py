"""Exploration helpers for querying unified data into Polars DataFrames."""

import uuid
from datetime import date, datetime

import polars as pl
from sqlalchemy import text

from app.db import async_session_factory


async def load_unified(
    table: str,
    user_id: uuid.UUID | str | None = None,
    since: date | None = None,
    until: date | None = None,
) -> pl.DataFrame:
    """Load unified table data into a Polars DataFrame.

    Args:
        table: One of 'activities', 'sleep', 'heart_rate', 'daily_metrics', 'body_metrics'
        user_id: Filter by user. If None, loads all users.
        since: Only load records after this date.
        until: Only load records before this date.

    Returns:
        Polars DataFrame with all columns from the unified table.
    """
    from app.config import settings

    if user_id is None:
        user_id = settings.default_user_id
    if isinstance(user_id, str):
        user_id = uuid.UUID(user_id)

    table_name = f"unified_{table}"
    valid_tables = [
        "unified_activities",
        "unified_sleep",
        "unified_heart_rate",
        "unified_daily_metrics",
        "unified_body_metrics",
    ]
    if table_name not in valid_tables:
        raise ValueError(f"Invalid table: {table}. Must be one of: {[t.replace('unified_', '') for t in valid_tables]}")

    query = f"SELECT * FROM {table_name} WHERE user_id = :uid"
    params: dict = {"uid": user_id}

    date_col = _get_date_column(table_name)
    if since and date_col:
        query += f" AND {date_col} >= :since"
        params["since"] = since
    if until and date_col:
        query += f" AND {date_col} <= :until"
        params["until"] = until

    query += f" ORDER BY {date_col}" if date_col else ""

    async with async_session_factory() as session:
        result = await session.execute(text(query), params)
        rows = result.mappings().all()

    if not rows:
        return pl.DataFrame()

    return pl.from_dicts([dict(row) for row in rows])


def _get_date_column(table_name: str) -> str | None:
    mapping = {
        "unified_activities": "started_at",
        "unified_sleep": "sleep_date",
        "unified_heart_rate": "recorded_at",
        "unified_daily_metrics": "metric_date",
        "unified_body_metrics": "measured_at",
    }
    return mapping.get(table_name)


def load_unified_sync(
    table: str,
    user_id: uuid.UUID | str | None = None,
    since: date | None = None,
    until: date | None = None,
) -> pl.DataFrame:
    """Synchronous wrapper for use in notebooks."""
    import asyncio

    try:
        loop = asyncio.get_running_loop()
        import nest_asyncio
        nest_asyncio.apply()
        return loop.run_until_complete(load_unified(table, user_id, since, until))
    except RuntimeError:
        return asyncio.run(load_unified(table, user_id, since, until))


def load_from_postgres(
    query: str,
    connection_uri: str = "postgresql://health:health@localhost:5432/health",
) -> pl.DataFrame:
    """Direct Polars read from Postgres using connectorx (fastest path for notebooks)."""
    return pl.read_database_uri(query, uri=connection_uri)
