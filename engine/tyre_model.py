"""
Tyre degradation model.

Uses real stint + lap data from OpenF1 to fit per-compound degradation curves.
Each compound gets a polynomial lap-time vs tyre-age model.
The "cliff" is detected when the degradation rate exceeds a threshold.

2025 compounds: SOFT, MEDIUM, HARD, INTERMEDIATE, WET
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
import logging

log = logging.getLogger(__name__)

# ── Compound constants (2025 Pirelli baselines) ────────────────────────────
# These are priors — they get overridden by fitted curves once we have data.

COMPOUND_BASE_LIFE = {
    "SOFT":         20,   # max recommended laps before cliff risk
    "MEDIUM":       35,
    "HARD":         50,
    "INTERMEDIATE": 30,
    "WET":          40,
}

# Approximate per-lap time loss (seconds) at nominal pace, no track-specific factor
COMPOUND_BASE_DEG = {
    "SOFT":         0.08,   # fast degrading
    "MEDIUM":       0.045,
    "HARD":         0.025,
    "INTERMEDIATE": 0.06,
    "WET":          0.04,
}

# Cliff multiplier — degradation rate jumps by this factor past optimal life
CLIFF_MULTIPLIER = 3.5

# Lap time delta for compound change (soft is fastest, hard slowest on fresh tyres)
# Referenced to MEDIUM = 0.0
COMPOUND_PACE_DELTA = {
    "SOFT":         -0.4,   # 0.4s faster per lap on fresh tyre vs medium
    "MEDIUM":        0.0,
    "HARD":         +0.5,
    "INTERMEDIATE":  0.0,   # context-dependent
    "WET":          +1.5,
}


@dataclass
class DegradationCurve:
    compound: str
    circuit: str
    # Polynomial coefficients: lap_time_delta = poly(tyre_age)
    # degree 2: [a, b, c] → a*age² + b*age + c
    coefficients: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    cliff_lap: Optional[int] = None          # lap at which cliff begins
    r_squared: float = 0.0                  # fit quality
    sample_count: int = 0
    fitted: bool = False

    def predict_delta(self, tyre_age: int) -> float:
        """Expected lap time delta (seconds) vs a fresh tyre of this compound."""
        if not self.fitted:
            # Fall back to linear baseline
            base = COMPOUND_BASE_DEG.get(self.compound, 0.05)
            cliff = COMPOUND_BASE_LIFE.get(self.compound, 30)
            if tyre_age <= cliff:
                return base * tyre_age
            else:
                cliff_delta = base * cliff
                return cliff_delta + (base * CLIFF_MULTIPLIER) * (tyre_age - cliff)

        delta = np.polyval(self.coefficients, tyre_age)
        # Apply cliff past cliff_lap
        if self.cliff_lap and tyre_age > self.cliff_lap:
            cliff_base = np.polyval(self.coefficients, self.cliff_lap)
            overshoot = tyre_age - self.cliff_lap
            deg_rate_at_cliff = COMPOUND_BASE_DEG.get(self.compound, 0.05) * CLIFF_MULTIPLIER
            delta = cliff_base + deg_rate_at_cliff * overshoot
        return float(delta)

    def is_near_cliff(self, tyre_age: int, window: int = 3) -> bool:
        """True if within `window` laps of cliff."""
        if self.cliff_lap:
            return tyre_age >= (self.cliff_lap - window)
        fallback = COMPOUND_BASE_LIFE.get(self.compound, 30)
        return tyre_age >= (fallback - window)


class TyreDegradationModel:
    """
    Fits and stores degradation curves per (compound, circuit).
    Built from OpenF1 stints + laps DataFrames.
    """

    def __init__(self):
        # Key: (compound, circuit_key) → DegradationCurve
        self.curves: dict[tuple[str, str], DegradationCurve] = {}

    # ── Data preparation ───────────────────────────────────────────────────

    def _merge_stint_laps(
        self,
        stints_df: pd.DataFrame,
        laps_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Join laps to their stint. Each lap gets: compound, tyre_age_at_lap.
        Excludes pit-out laps and lap 1 (fuel/traffic distort baseline).
        """
        if stints_df.empty or laps_df.empty:
            return pd.DataFrame()

        rows = []
        for _, stint in stints_df.iterrows():
            driver = stint["driver_number"]
            compound = stint.get("compound", "UNKNOWN")
            tyre_age_start = stint.get("tyre_age_at_start", 0) or 0
            lap_start = int(stint["lap_start"])
            lap_end = int(stint["lap_end"])

            stint_laps = laps_df[
                (laps_df["driver_number"] == driver) &
                (laps_df["lap_number"] >= lap_start) &
                (laps_df["lap_number"] <= lap_end) &
                (~laps_df.get("is_pit_out_lap", pd.Series([False]*len(laps_df))).values)
            ].copy()

            if stint_laps.empty:
                continue

            # Compute tyre age within stint
            stint_laps["tyre_age"] = (
                stint_laps["lap_number"] - lap_start + tyre_age_start
            )
            stint_laps["compound"] = compound
            rows.append(stint_laps)

        if not rows:
            return pd.DataFrame()

        merged = pd.concat(rows, ignore_index=True)
        # Only keep laps with valid numeric lap_duration
        merged = merged[merged["lap_duration"].notna()]
        merged = merged[merged["lap_duration"] > 60]   # sanity: >1 min
        merged = merged[merged["lap_duration"] < 200]  # sanity: <3:20

        # Drop lap 1 globally (often distorted by traffic)
        merged = merged[merged["lap_number"] > 1]
        return merged

    # ── Curve fitting ──────────────────────────────────────────────────────

    def fit(
        self,
        stints_df: pd.DataFrame,
        laps_df: pd.DataFrame,
        circuit_key: str,
        poly_degree: int = 2,
    ) -> dict[str, DegradationCurve]:
        """
        Fit degradation curves for each compound from one race's data.
        Returns dict of compound → DegradationCurve.
        """
        merged = self._merge_stint_laps(stints_df, laps_df)
        if merged.empty:
            log.warning(f"No merged data for circuit {circuit_key}")
            return {}

        results = {}

        for compound in merged["compound"].unique():
            if compound in ("UNKNOWN",):
                continue

            cdf = merged[merged["compound"] == compound].copy()

            # Need at least 10 data points to fit meaningfully
            if len(cdf) < 10:
                log.info(f"Skipping {compound} — only {len(cdf)} laps")
                continue

            # Normalise lap times: subtract median of first 5 laps on that compound
            # so we're fitting the *delta* from fresh-tyre pace
            fresh_mask = cdf["tyre_age"] <= 5
            if fresh_mask.sum() < 3:
                baseline = cdf["lap_duration"].quantile(0.1)
            else:
                baseline = cdf.loc[fresh_mask, "lap_duration"].median()

            cdf["lap_delta"] = cdf["lap_duration"] - baseline

            ages = cdf["tyre_age"].values.astype(float)
            deltas = cdf["lap_delta"].values.astype(float)

            # Fit polynomial
            try:
                coeffs = np.polyfit(ages, deltas, poly_degree)
                predicted = np.polyval(coeffs, ages)
                ss_res = np.sum((deltas - predicted) ** 2)
                ss_tot = np.sum((deltas - np.mean(deltas)) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            except np.linalg.LinAlgError:
                log.warning(f"Polyfit failed for {compound} on {circuit_key}")
                continue

            # Detect cliff: where predicted derivative exceeds 3× mean deg rate
            max_age = int(cdf["tyre_age"].max())
            cliff_lap = self._detect_cliff(coeffs, max_age, compound)

            curve = DegradationCurve(
                compound=compound,
                circuit=circuit_key,
                coefficients=coeffs,
                cliff_lap=cliff_lap,
                r_squared=round(r2, 3),
                sample_count=len(cdf),
                fitted=True,
            )

            self.curves[(compound, circuit_key)] = curve
            results[compound] = curve

            log.info(
                f"Fitted {compound} @ {circuit_key}: cliff_lap={cliff_lap}, "
                f"R²={r2:.3f}, n={len(cdf)}"
            )

        return results

    def _detect_cliff(
        self, coeffs: np.ndarray, max_age: int, compound: str
    ) -> Optional[int]:
        """
        Find the lap where the degradation rate (derivative) first exceeds
        the nominal rate by CLIFF_MULTIPLIER.
        """
        nominal_rate = COMPOUND_BASE_DEG.get(compound, 0.05)
        threshold = nominal_rate * CLIFF_MULTIPLIER

        # Derivative of polynomial
        deriv_coeffs = np.polyder(coeffs)
        for age in range(5, max_age + 1):
            rate = abs(np.polyval(deriv_coeffs, age))
            if rate >= threshold:
                return age

        return None

    # ── Predictions ────────────────────────────────────────────────────────

    def predict(
        self,
        compound: str,
        tyre_age: int,
        circuit_key: str,
    ) -> float:
        """
        Return expected lap time delta (seconds) for given compound + age.
        Falls back to baseline model if no fitted curve exists.
        """
        curve = self.curves.get((compound, circuit_key))
        if curve:
            return curve.predict_delta(tyre_age)
        # Unfitted fallback
        dummy = DegradationCurve(compound=compound, circuit=circuit_key)
        return dummy.predict_delta(tyre_age)

    def cliff_warning(self, compound: str, tyre_age: int, circuit_key: str) -> bool:
        """True if this compound is near or past its cliff at this circuit."""
        curve = self.curves.get((compound, circuit_key))
        if curve:
            return curve.is_near_cliff(tyre_age)
        cliff = COMPOUND_BASE_LIFE.get(compound, 30)
        return tyre_age >= cliff - 3

    def get_optimal_stint_length(self, compound: str, circuit_key: str) -> int:
        """Return the cliff lap (or default) as the optimal max stint length."""
        curve = self.curves.get((compound, circuit_key))
        if curve and curve.cliff_lap:
            return curve.cliff_lap
        return COMPOUND_BASE_LIFE.get(compound, 30)

    def summary(self) -> pd.DataFrame:
        """DataFrame of all fitted curves for inspection."""
        rows = []
        for (compound, circuit), curve in self.curves.items():
            rows.append({
                "compound": compound,
                "circuit": circuit,
                "cliff_lap": curve.cliff_lap,
                "r_squared": curve.r_squared,
                "sample_count": curve.sample_count,
                "fitted": curve.fitted,
            })
        return pd.DataFrame(rows)
