import uuid

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.models.team import Athlete
from app.schemas.team import (
    AthleteCreate,
    AthleteOut,
    AthleteUpdate,
    AthleteSummaryResponse,
    TeamReadinessResponse,
)
from app.services.readiness import get_athlete_summary, get_team_readiness

router = APIRouter(prefix="/api/team", tags=["team"])


@router.get("/readiness", response_model=TeamReadinessResponse)
async def team_readiness(
    sport: str | None = Query(None),
    session: AsyncSession = Depends(get_session),
):
    return await get_team_readiness(session, sport=sport)


@router.get("/athletes", response_model=list[AthleteOut])
async def list_athletes(
    active_only: bool = Query(True),
    session: AsyncSession = Depends(get_session),
):
    q = select(Athlete)
    if active_only:
        q = q.where(Athlete.is_active.is_(True))
    q = q.order_by(Athlete.name)
    rows = (await session.execute(q)).scalars().all()
    return [AthleteOut.model_validate(a) for a in rows]


@router.get("/athletes/{athlete_id}/summary", response_model=AthleteSummaryResponse)
async def athlete_summary(
    athlete_id: uuid.UUID,
    days: int = Query(14, ge=1, le=90),
    session: AsyncSession = Depends(get_session),
):
    result = await get_athlete_summary(session, athlete_id, days=days)
    if result is None:
        raise HTTPException(status_code=404, detail="Athlete not found")
    return result


@router.post("/athletes", response_model=AthleteOut, status_code=201)
async def create_athlete(
    body: AthleteCreate,
    session: AsyncSession = Depends(get_session),
):
    athlete = Athlete(
        name=body.name,
        sport=body.sport,
        position=body.position,
        jersey_number=body.jersey_number,
        catapult_athlete_id=body.catapult_athlete_id,
    )
    session.add(athlete)
    await session.commit()
    await session.refresh(athlete)
    return AthleteOut.model_validate(athlete)


@router.patch("/athletes/{athlete_id}", response_model=AthleteOut)
async def update_athlete(
    athlete_id: uuid.UUID,
    body: AthleteUpdate,
    session: AsyncSession = Depends(get_session),
):
    athlete = (
        await session.execute(select(Athlete).where(Athlete.id == athlete_id))
    ).scalar_one_or_none()
    if not athlete:
        raise HTTPException(status_code=404, detail="Athlete not found")

    updates = body.model_dump(exclude_unset=True)
    for field, value in updates.items():
        setattr(athlete, field, value)

    await session.commit()
    await session.refresh(athlete)
    return AthleteOut.model_validate(athlete)
