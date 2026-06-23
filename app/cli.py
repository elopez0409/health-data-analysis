import asyncio
import uuid

import typer
from rich.console import Console
from rich.table import Table

from app.config import settings
from app.schemas.common import Provider

cli_app = typer.Typer(help="Health Data Backend CLI")
console = Console()


def _import_providers():
    """Import provider modules to trigger registration."""
    import app.providers.strava.client  # noqa: F401
    import app.providers.fitbit.client  # noqa: F401
    import app.providers.oura.client  # noqa: F401
    import app.providers.withings.client  # noqa: F401
    import app.providers.whoop.client  # noqa: F401
    import app.providers.garmin.client  # noqa: F401
    import app.providers.catapult.client  # noqa: F401


async def _verify_all_async():
    from app.db import async_session_factory
    from app.providers.registry import ProviderRegistry
    from app.providers.token_store import get_token

    _import_providers()
    user_id = uuid.UUID(settings.default_user_id)

    table = Table(title="Provider Connection Status")
    table.add_column("Provider", style="bold")
    table.add_column("Status")
    table.add_column("Latency (ms)")
    table.add_column("Username")
    table.add_column("Error")

    enabled = ProviderRegistry.enabled_providers()

    async with async_session_factory() as session:
        for provider in enabled:
            if provider == Provider.CATAPULT:
                if not settings.catapult_api_token:
                    table.add_row(
                        provider.value,
                        "[yellow]NO TOKEN[/yellow]",
                        "-",
                        "-",
                        "No API token configured. Set CATAPULT_API_TOKEN in .env.",
                    )
                    continue

                client = ProviderRegistry.get_client(provider)
                status = await client.verify_connection(user_id)
            else:
                token = await get_token(session, user_id, provider)
                if not token:
                    table.add_row(
                        provider.value,
                        "[yellow]NO TOKEN[/yellow]",
                        "-",
                        "-",
                        "No token stored. Run OAuth flow first.",
                    )
                    continue

                client = ProviderRegistry.get_client(provider)
                client._access_token = token.access_token
                status = await client.verify_connection(user_id)

            if status.connected:
                table.add_row(
                    provider.value,
                    "[green]OK[/green]",
                    f"{status.latency_ms:.0f}" if status.latency_ms else "-",
                    status.username or "-",
                    "-",
                )
            else:
                table.add_row(
                    provider.value,
                    "[red]FAIL[/red]",
                    f"{status.latency_ms:.0f}" if status.latency_ms else "-",
                    "-",
                    status.error or "Unknown error",
                )

    console.print(table)


@cli_app.command("verify-all")
def verify_all():
    """Verify connectivity to all enabled providers."""
    asyncio.run(_verify_all_async())


async def _pull_async(provider: str):
    _import_providers()

    if provider == "strava":
        from app.jobs.strava_pull import strava_pull_job
        count = await strava_pull_job()
        console.print(f"[green]Strava: {count} new records ingested[/green]")
    elif provider == "fitbit":
        from app.jobs.fitbit_pull import fitbit_pull_job
        count = await fitbit_pull_job()
        console.print(f"[green]Fitbit: {count} new records ingested[/green]")
    elif provider == "catapult":
        from app.jobs.catapult_pull import catapult_pull_job
        count = await catapult_pull_job()
        console.print(f"[green]Catapult: {count} new records ingested[/green]")
    else:
        console.print(f"[red]Pull not implemented for: {provider}[/red]")


async def _personal_hr_async():
    from app.jobs.personal_hr import personal_hr_job

    summary = await personal_hr_job()
    console.print(
        f"[green]Personal HR: {summary['windows']} windows, "
        f"{summary['sources_updated']} sources updated, "
        f"{summary['anomalies']} anomalies[/green]"
    )
    if summary.get("trusted_source"):
        console.print(f"  Trusted source: {summary['trusted_source']}")


@cli_app.command("pull")
def pull(provider: str = typer.Argument(..., help="Provider to pull from")):
    """Pull data from a specific provider."""
    asyncio.run(_pull_async(provider))


async def _seed_roster_async(
    name: str,
    sport: str | None,
    position: str | None,
    jersey_number: str | None,
    catapult_id: str | None,
):
    from app.db import async_session_factory
    from app.models.team import Athlete

    async with async_session_factory() as session:
        athlete = Athlete(
            name=name,
            sport=sport,
            position=position,
            jersey_number=jersey_number,
            catapult_athlete_id=catapult_id,
        )
        session.add(athlete)
        await session.commit()
        await session.refresh(athlete)
        console.print(
            f"[green]Created athlete: {athlete.name} "
            f"(id={athlete.id}, user_id={athlete.user_id})[/green]"
        )


@cli_app.command("seed-roster")
def seed_roster(
    name: str = typer.Option(..., help="Athlete name"),
    sport: str = typer.Option(None, help="Sport (e.g. Basketball)"),
    position: str = typer.Option(None, help="Position (e.g. PG)"),
    jersey_number: str = typer.Option(None, "--jersey", help="Jersey number"),
    catapult_id: str = typer.Option(None, "--catapult-id", help="Catapult athlete ID for mapping"),
):
    """Create an athlete in the roster for demo setup."""
    asyncio.run(_seed_roster_async(name, sport, position, jersey_number, catapult_id))


@cli_app.command("personal-hr")
def personal_hr():
    """Run the personal HR living-model update (nightly job, manual trigger)."""
    asyncio.run(_personal_hr_async())


def _cli_entrypoint():
    cli_app()


if __name__ == "__main__":
    _cli_entrypoint()
