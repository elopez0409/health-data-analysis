"""Walk-forward out-of-sample backtest for personal calibration."""

from __future__ import annotations

import numpy as np

from hr_selection.personal.estimator import OffsetState, update_offset

STRATEGIES = ("uncorrected", "population_prior", "personal")


def walk_forward_backtest(
    paired,
    *,
    warmup: int = 7,
    population_mean: float | None = None,
) -> tuple[list[dict], dict]:
    """Expanding-window walk-forward backtest per user."""
    if population_mean is None:
        population_mean = float(paired["delta"].mean())

    all_nights: list[dict] = []

    for user_id, grp in paired.groupby("user_id"):
        grp = grp.sort_values("date").reset_index(drop=True)
        state = OffsetState()

        for t in range(len(grp)):
            row = grp.iloc[t]
            chest = float(row["chest"])
            wrist = float(row["wrist"])
            night = t + 1

            if t >= warmup:
                offset_est = state.offset_mean if state.n_samples > 0 else population_mean
                preds = {
                    "uncorrected": wrist,
                    "population_prior": wrist - population_mean,
                    "personal": wrist - offset_est,
                }
                for strat, pred in preds.items():
                    all_nights.append(
                        {
                            "user_id": user_id,
                            "night": night,
                            "date": str(row["date"].date()) if hasattr(row["date"], "date") else str(row["date"]),
                            "strategy": strat,
                            "predicted_hr": pred,
                            "target_chest": chest,
                            "error": abs(pred - chest),
                            "in_sample": False,
                        }
                    )

            state = update_offset(state, float(row["delta"]))

    return all_nights, {"population_mean": population_mean, "warmup": warmup}


def aggregate_backtest_metrics(nights: list[dict], holdout_start: int = 60) -> dict:
    """Aggregate OOS MAE metrics for each strategy."""
    import pandas as pd

    df = pd.DataFrame(nights)
    out: dict = {"strategies": {}, "holdout_start": holdout_start}

    for strat in STRATEGIES:
        sub = df[df["strategy"] == strat]
        cum_mae = float(sub["error"].mean())
        holdout = sub[sub["night"] >= holdout_start]
        holdout_mae = float(holdout["error"].mean()) if len(holdout) else float("nan")

        rolling = []
        for _, grp in sub.groupby("user_id"):
            grp = grp.sort_values("night")
            if len(grp) >= 14:
                rolling.append(float(grp.tail(14)["error"].mean()))
        rolling_mae = float(np.mean(rolling)) if rolling else float("nan")

        out["strategies"][strat] = {
            "cumulative_oos_mae": round(cum_mae, 4),
            "holdout_mae": round(holdout_mae, 4),
            "rolling_14night_mae": round(rolling_mae, 4),
            "n_predictions": int(len(sub)),
        }

    uncorr = out["strategies"]["uncorrected"]["cumulative_oos_mae"]
    personal = out["strategies"]["personal"]["cumulative_oos_mae"]
    if uncorr > 0:
        out["skill_pct_vs_uncorrected"] = round(100 * (uncorr - personal) / uncorr, 2)
    else:
        out["skill_pct_vs_uncorrected"] = 0.0

    return out


def cumulative_mae_curve(nights: list[dict]) -> "pd.DataFrame":
    """Return cumulative mean OOS error by night for each strategy."""
    import pandas as pd

    df = pd.DataFrame(nights)
    curves = []
    for strat in STRATEGIES:
        sub = df[df["strategy"] == strat].sort_values(["user_id", "night"])
        for user_id, grp in sub.groupby("user_id"):
            grp = grp.copy()
            grp["cumulative_mae"] = grp["error"].expanding().mean()
            grp["strategy"] = strat
            curves.append(grp[["user_id", "night", "strategy", "cumulative_mae", "error"]])
    return pd.concat(curves, ignore_index=True)
