"""Nightly job: update per-user personal HR offsets and detect anomalies."""

from __future__ import annotations

import math
import uuid
from datetime import datetime, timezone

from sqlalchemy import select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import async_session_factory
from app.explore import load_unified
from app.logging import get_logger
from app.models.personal import HrAnomaly, PersonalHrState
from hr_selection.personal.align import (
    align_heart_rate_windows,
    compute_deltas,
    extract_profile_observations,
)
from hr_selection.personal.anomaly import (
    detect_offset_anomaly,
    detect_self_drift_anomaly,
)
from hr_selection.personal.estimator import OffsetState, batch_update_offset
from hr_selection.personal.profile import BaselineProfile, batch_update_profile
from hr_selection.personal.trusted import select_trusted_source

logger = get_logger(__name__)

CURSOR_PROVIDER = "personal_hr"
CURSOR_RESOURCE = "heart_rate"


async def _get_cursor(
    session: AsyncSession, user_id: uuid.UUID
) -> datetime | None:
    result = await session.execute(
        text(
            "SELECT last_value FROM ingestion_cursors "
            "WHERE user_id = :uid AND provider = :provider AND resource = :resource"
        ),
        {"uid": user_id, "provider": CURSOR_PROVIDER, "resource": CURSOR_RESOURCE},
    )
    row = result.first()
    if row:
        return datetime.fromisoformat(row[0])
    return None


async def _update_cursor(
    session: AsyncSession, user_id: uuid.UUID, value: datetime
) -> None:
    await session.execute(
        text(
            "INSERT INTO ingestion_cursors (user_id, provider, resource, last_value, updated_at) "
            "VALUES (:uid, :provider, :resource, :val, now()) "
            "ON CONFLICT ON CONSTRAINT uq_cursor "
            "DO UPDATE SET last_value = :val, updated_at = now()"
        ),
        {
            "uid": user_id,
            "provider": CURSOR_PROVIDER,
            "resource": CURSOR_RESOURCE,
            "val": value.isoformat(),
        },
    )


async def _load_existing_state(
    session: AsyncSession, user_id: uuid.UUID
) -> dict[str, PersonalHrState]:
    result = await session.execute(
        select(PersonalHrState).where(PersonalHrState.user_id == user_id)
    )
    rows = result.scalars().all()
    return {row.source: row for row in rows}


async def _upsert_state(
    session: AsyncSession,
    user_id: uuid.UUID,
    source: str,
    *,
    trusted_source: str,
    offset_state: OffsetState,
    profile: BaselineProfile,
) -> None:
    now = datetime.now(timezone.utc)
    values = {
        "user_id": user_id,
        "source": source,
        "trusted_source": trusted_source,
        "offset_mean": offset_state.offset_mean,
        "offset_var": offset_state.offset_var,
        "n_samples": offset_state.n_samples,
        "ci_low": offset_state.ci_low,
        "ci_high": offset_state.ci_high,
        "ewma_offset": offset_state.ewma_offset,
        "last_updated": now,
        "baseline_profile": profile.to_dict(),
    }
    stmt = pg_insert(PersonalHrState).values(**values)
    stmt = stmt.on_conflict_do_update(
        constraint="uq_personal_hr_state",
        set_={
            "trusted_source": trusted_source,
            "offset_mean": offset_state.offset_mean,
            "offset_var": offset_state.offset_var,
            "n_samples": offset_state.n_samples,
            "ci_low": offset_state.ci_low,
            "ci_high": offset_state.ci_high,
            "ewma_offset": offset_state.ewma_offset,
            "last_updated": now,
            "baseline_profile": profile.to_dict(),
        },
    )
    await session.execute(stmt)


async def _insert_anomaly(
    session: AsyncSession,
    user_id: uuid.UUID,
    source: str,
    result,
) -> None:
    stmt = pg_insert(HrAnomaly).values(
        user_id=user_id,
        source=source,
        detected_at=datetime.now(timezone.utc),
        shift_bpm=result.shift_bpm,
        prev_offset=result.prev_offset,
        new_offset=result.new_offset,
        severity=result.severity,
        resolved=False,
        details=result.to_dict(),
    )
    await session.execute(stmt)


def _offset_state_from_row(row: PersonalHrState | None) -> OffsetState:
    if row is None:
        return OffsetState()
    return OffsetState(
        offset_mean=row.offset_mean,
        offset_var=row.offset_var,
        n_samples=row.n_samples,
        ewma_offset=row.ewma_offset,
        ci_low=row.ci_low if row.ci_low is not None else float("nan"),
        ci_high=row.ci_high if row.ci_high is not None else float("nan"),
    )


def _profile_from_row(row: PersonalHrState | None) -> BaselineProfile:
    if row is None or not row.baseline_profile:
        return BaselineProfile()
    return BaselineProfile.from_dict(row.baseline_profile)


async def personal_hr_job(user_id: uuid.UUID | None = None) -> dict:
    """Run the living-model update for one user.

    Returns summary dict with counts of windows processed, sources updated,
    and anomalies detected.
    """
    from app.config import settings

    if user_id is None:
        user_id = uuid.UUID(settings.default_user_id)

    summary = {
        "user_id": str(user_id),
        "windows": 0,
        "sources_updated": 0,
        "anomalies": 0,
    }

    async with async_session_factory() as session:
        since = await _get_cursor(session, user_id)
        existing = await _load_existing_state(session, user_id)

        existing_trusted = None
        for row in existing.values():
            if row.trusted_source:
                existing_trusted = row.trusted_source
                break

    hr_df = await load_unified("heart_rate", user_id=user_id)
    if hr_df.is_empty():
        logger.info("personal_hr_no_data", user_id=str(user_id))
        return summary

    if since is not None:
        hr_df = hr_df.filter(hr_df["recorded_at"] > since)

    if hr_df.is_empty():
        logger.info("personal_hr_no_new_data", user_id=str(user_id))
        return summary

    windows = align_heart_rate_windows(hr_df)
    summary["windows"] = len(windows)
    if not windows:
        return summary

    all_sources = set()
    for win in windows:
        all_sources.update(win["sources"].keys())

    trusted = select_trusted_source(
        sorted(all_sources), existing_trusted=existing_trusted
    )
    if trusted is None:
        return summary

    deltas_by_source = compute_deltas(windows, trusted)
    is_single_source = len(all_sources) <= 1

    async with async_session_factory() as session:
        existing = await _load_existing_state(session, user_id)
        anomalies_detected = 0

        for src in all_sources:
            row = existing.get(src)
            offset_state = _offset_state_from_row(row)
            profile = _profile_from_row(row)

            obs = extract_profile_observations(windows, source=src)
            profile = batch_update_profile(profile, obs)

            if src == trusted:
                offset_state = OffsetState(
                    offset_mean=0.0,
                    offset_var=0.0,
                    n_samples=offset_state.n_samples,
                    ewma_offset=0.0,
                    ci_low=0.0,
                    ci_high=0.0,
                )
            elif is_single_source:
                recent_hr = [o["bpm"] for o in obs[-50:]]
                resting = profile.resting_mean()
                if obs and not math.isnan(resting):
                    offset_state = batch_update_offset(
                        offset_state,
                        [o["bpm"] - resting for o in obs],
                    )
                anomaly = detect_self_drift_anomaly(resting, recent_hr)
                if anomaly.is_anomaly:
                    await _insert_anomaly(session, user_id, src, anomaly)
                    anomalies_detected += 1
            else:
                src_deltas = deltas_by_source.get(src, [])
                if src_deltas:
                    pre_state = OffsetState.from_dict(offset_state.to_dict())
                    offset_state = batch_update_offset(offset_state, src_deltas)
                    recent = src_deltas[-50:]
                    anomaly = detect_offset_anomaly(pre_state, recent)
                    if anomaly.is_anomaly:
                        await _insert_anomaly(session, user_id, src, anomaly)
                        anomalies_detected += 1

            await _upsert_state(
                session,
                user_id,
                src,
                trusted_source=trusted,
                offset_state=offset_state,
                profile=profile,
            )
            summary["sources_updated"] += 1

        max_ts = hr_df["recorded_at"].max()
        if max_ts is not None:
            if hasattr(max_ts, "to_pydatetime"):
                cursor_val = max_ts.to_pydatetime()
            else:
                cursor_val = max_ts
            if cursor_val.tzinfo is None:
                cursor_val = cursor_val.replace(tzinfo=timezone.utc)
            await _update_cursor(session, user_id, cursor_val)

        await session.commit()

    summary["anomalies"] = anomalies_detected
    summary["trusted_source"] = trusted
    logger.info("personal_hr_complete", **summary)
    return summary
