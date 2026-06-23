import uuid
from datetime import date, datetime, timedelta, timezone

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.personal import HrAnomaly, PersonalHrState
from app.models.team import Athlete
from app.models.unified import UnifiedActivity, UnifiedDailyMetrics, UnifiedSleep
from app.schemas.team import (
    AnomalyOut,
    AthleteOut,
    AthleteReadiness,
    AthleteSummaryResponse,
    DailyMetricPoint,
    HrStateOut,
    RecentActivity,
    SleepPoint,
    TeamReadinessResponse,
)


def compute_status(
    metrics: UnifiedDailyMetrics | None,
    sleep: UnifiedSleep | None,
    anomaly_count: int,
    hr_state: PersonalHrState | None,
    now: datetime | None = None,
) -> tuple[str, list[str]]:
    now = now or datetime.now(timezone.utc)
    reasons: list[str] = []

    has_metrics = metrics is not None
    has_sleep = sleep is not None

    if not has_metrics and not has_sleep:
        return "no_data", ["no metrics or sleep data in recent history"]

    # Check data staleness
    cutoff = (now - timedelta(days=3)).date()
    if has_metrics and metrics.metric_date < cutoff:
        reasons.append(f"metrics stale: last data {metrics.metric_date}")
        return "red", reasons
    if has_sleep and sleep.sleep_date < cutoff:
        if not has_metrics:
            reasons.append(f"sleep stale: last data {sleep.sleep_date}")
            return "red", reasons

    is_red = False
    is_yellow = False

    # Recovery score
    if has_metrics and metrics.recovery_score is not None:
        if metrics.recovery_score < 34:
            reasons.append(f"recovery_score: {metrics.recovery_score:.0f} (red zone)")
            is_red = True
        elif metrics.recovery_score < 67:
            reasons.append(f"recovery_score: {metrics.recovery_score:.0f} (yellow zone)")
            is_yellow = True

    # Readiness score
    if has_metrics and metrics.readiness_score is not None:
        if metrics.readiness_score < 60:
            reasons.append(f"readiness_score: {metrics.readiness_score:.0f} (red zone)")
            is_red = True
        elif metrics.readiness_score < 70:
            reasons.append(f"readiness_score: {metrics.readiness_score:.0f} (yellow zone)")
            is_yellow = True

    # Sleep hours
    if has_sleep and sleep.total_seconds is not None:
        sleep_hours = sleep.total_seconds / 3600
        if sleep_hours < 5:
            reasons.append(f"sleep: {sleep_hours:.1f}h (red zone)")
            is_red = True
        elif sleep_hours < 6.5:
            reasons.append(f"sleep: {sleep_hours:.1f}h (yellow zone)")
            is_yellow = True

    # Anomalies
    if anomaly_count > 0:
        reasons.append(f"{anomaly_count} unresolved HR anomal{'y' if anomaly_count == 1 else 'ies'}")
        is_yellow = True

    if is_red:
        return "red", reasons
    if is_yellow:
        return "yellow", reasons
    return "green", reasons


async def get_team_readiness(
    session: AsyncSession, sport: str | None = None
) -> TeamReadinessResponse:
    # 1. All active athletes
    athlete_q = select(Athlete).where(Athlete.is_active.is_(True))
    if sport:
        athlete_q = athlete_q.where(Athlete.sport == sport)
    athletes = list((await session.execute(athlete_q)).scalars().all())

    if not athletes:
        return TeamReadinessResponse(
            team_size=0, reporting=0, green=0, yellow=0, red=0, no_data=0,
            as_of=datetime.now(timezone.utc), athletes=[],
        )

    user_ids = [a.user_id for a in athletes]

    # 2. Latest daily metrics per user_id
    metrics_sub = (
        select(UnifiedDailyMetrics)
        .where(UnifiedDailyMetrics.user_id.in_(user_ids))
        .distinct(UnifiedDailyMetrics.user_id)
        .order_by(UnifiedDailyMetrics.user_id, UnifiedDailyMetrics.metric_date.desc())
        .subquery()
    )
    metrics_rows = (
        await session.execute(select(UnifiedDailyMetrics).from_statement(
            select(UnifiedDailyMetrics).where(
                UnifiedDailyMetrics.id == metrics_sub.c.id
            )
        ))
    ).scalars().all()
    metrics_by_uid: dict[uuid.UUID, UnifiedDailyMetrics] = {m.user_id: m for m in metrics_rows}

    # 3. Latest sleep per user_id
    sleep_sub = (
        select(UnifiedSleep)
        .where(UnifiedSleep.user_id.in_(user_ids))
        .distinct(UnifiedSleep.user_id)
        .order_by(UnifiedSleep.user_id, UnifiedSleep.sleep_date.desc())
        .subquery()
    )
    sleep_rows = (
        await session.execute(select(UnifiedSleep).from_statement(
            select(UnifiedSleep).where(UnifiedSleep.id == sleep_sub.c.id)
        ))
    ).scalars().all()
    sleep_by_uid: dict[uuid.UUID, UnifiedSleep] = {s.user_id: s for s in sleep_rows}

    # 4. Unresolved anomaly counts
    anomaly_q = (
        select(HrAnomaly.user_id, func.count().label("cnt"))
        .where(HrAnomaly.user_id.in_(user_ids), HrAnomaly.resolved.is_(False))
        .group_by(HrAnomaly.user_id)
    )
    anomaly_rows = (await session.execute(anomaly_q)).all()
    anomaly_counts: dict[uuid.UUID, int] = {row.user_id: row.cnt for row in anomaly_rows}

    # 5. Latest HR state per user_id
    hr_sub = (
        select(PersonalHrState)
        .where(PersonalHrState.user_id.in_(user_ids))
        .distinct(PersonalHrState.user_id)
        .order_by(PersonalHrState.user_id, PersonalHrState.last_updated.desc())
        .subquery()
    )
    hr_rows = (
        await session.execute(select(PersonalHrState).from_statement(
            select(PersonalHrState).where(PersonalHrState.id == hr_sub.c.id)
        ))
    ).scalars().all()
    hr_by_uid: dict[uuid.UUID, PersonalHrState] = {h.user_id: h for h in hr_rows}

    now = datetime.now(timezone.utc)
    results: list[AthleteReadiness] = []
    counts = {"green": 0, "yellow": 0, "red": 0, "no_data": 0}

    for athlete in athletes:
        uid = athlete.user_id
        m = metrics_by_uid.get(uid)
        s = sleep_by_uid.get(uid)
        ac = anomaly_counts.get(uid, 0)
        hr = hr_by_uid.get(uid)

        status, reasons = compute_status(m, s, ac, hr, now)
        counts[status] += 1

        sleep_hours = None
        if s and s.total_seconds is not None:
            sleep_hours = round(s.total_seconds / 3600, 1)

        results.append(AthleteReadiness(
            id=athlete.id,
            name=athlete.name,
            sport=athlete.sport,
            position=athlete.position,
            jersey_number=athlete.jersey_number,
            status=status,
            status_reasons=reasons,
            readiness_score=m.readiness_score if m else None,
            recovery_score=m.recovery_score if m else None,
            hrv_rmssd=m.hrv_rmssd if m else None,
            resting_hr=m.resting_heart_rate if m else None,
            sleep_hours=sleep_hours,
            sleep_score=s.sleep_score if s else None,
            anomaly_count=ac,
            hr_offset=hr.offset_mean if hr else None,
            metrics_date=m.metric_date if m else None,
            sleep_date=s.sleep_date if s else None,
        ))

    reporting = counts["green"] + counts["yellow"] + counts["red"]

    return TeamReadinessResponse(
        team_size=len(athletes),
        reporting=reporting,
        green=counts["green"],
        yellow=counts["yellow"],
        red=counts["red"],
        no_data=counts["no_data"],
        as_of=now,
        athletes=results,
    )


async def get_athlete_summary(
    session: AsyncSession, athlete_id: uuid.UUID, days: int = 14
) -> AthleteSummaryResponse | None:
    athlete = (
        await session.execute(select(Athlete).where(Athlete.id == athlete_id))
    ).scalar_one_or_none()
    if not athlete:
        return None

    uid = athlete.user_id
    since = date.today() - timedelta(days=days)

    # Daily metrics
    metrics = (
        await session.execute(
            select(UnifiedDailyMetrics)
            .where(UnifiedDailyMetrics.user_id == uid, UnifiedDailyMetrics.metric_date >= since)
            .order_by(UnifiedDailyMetrics.metric_date)
        )
    ).scalars().all()

    daily_metric_points = [
        DailyMetricPoint(
            date=m.metric_date,
            readiness_score=m.readiness_score,
            recovery_score=m.recovery_score,
            hrv_rmssd=m.hrv_rmssd,
            resting_heart_rate=m.resting_heart_rate,
            steps=m.steps,
            strain_score=m.strain_score,
        )
        for m in metrics
    ]

    # Sleep
    sleep_rows = (
        await session.execute(
            select(UnifiedSleep)
            .where(UnifiedSleep.user_id == uid, UnifiedSleep.sleep_date >= since)
            .order_by(UnifiedSleep.sleep_date)
        )
    ).scalars().all()

    sleep_points = [
        SleepPoint(
            date=s.sleep_date,
            total_hours=round(s.total_seconds / 3600, 2) if s.total_seconds else None,
            sleep_score=s.sleep_score,
            deep_hours=round(s.deep_seconds / 3600, 2) if s.deep_seconds else None,
            rem_hours=round(s.rem_seconds / 3600, 2) if s.rem_seconds else None,
        )
        for s in sleep_rows
    ]

    # Recent activities (last N days)
    since_dt = datetime.now(timezone.utc) - timedelta(days=days)
    activities = (
        await session.execute(
            select(UnifiedActivity)
            .where(UnifiedActivity.user_id == uid, UnifiedActivity.started_at >= since_dt)
            .order_by(UnifiedActivity.started_at.desc())
            .limit(20)
        )
    ).scalars().all()

    recent = [
        RecentActivity(
            activity_type=a.activity_type,
            started_at=a.started_at,
            duration_seconds=a.duration_seconds,
            title=a.title,
        )
        for a in activities
    ]

    # Anomalies
    anomalies = (
        await session.execute(
            select(HrAnomaly)
            .where(HrAnomaly.user_id == uid, HrAnomaly.resolved.is_(False))
            .order_by(HrAnomaly.detected_at.desc())
        )
    ).scalars().all()

    anomaly_out = [
        AnomalyOut(
            source=a.source,
            detected_at=a.detected_at,
            shift_bpm=a.shift_bpm,
            severity=a.severity,
            resolved=a.resolved,
        )
        for a in anomalies
    ]

    # HR state
    hr_states = (
        await session.execute(
            select(PersonalHrState).where(PersonalHrState.user_id == uid)
        )
    ).scalars().all()

    hr_out = [
        HrStateOut(
            source=h.source,
            offset_mean=h.offset_mean,
            n_samples=h.n_samples,
            last_updated=h.last_updated,
        )
        for h in hr_states
    ]

    return AthleteSummaryResponse(
        athlete=AthleteOut.model_validate(athlete),
        daily_metrics=daily_metric_points,
        sleep=sleep_points,
        recent_activities=recent,
        anomalies=anomaly_out,
        hr_state=hr_out,
    )
