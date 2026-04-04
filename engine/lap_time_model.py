"""
Lap time model.

Predicts lap time for any lap given:
- Tyre compound + age + degradation curve
- Fuel load (decreases ~1.5 kg/lap → ~0.03s/lap faster over race)
- Weather (track temp, rainfall)
- Driver style profile
- DRS availability
- Safety car / VSC periods
- Track-specific base pace

All deltas are in seconds relative to a reference lap.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

from engine.tyre_model import TyreDegradationModel, COMPOUND_PACE_DELTA


# ── Driver style profiles (2025 grid) ────────────────────────────────────
# All values 0.0–1.0 unless noted.
# tyre_management: 1.0 = can nurse tyres very well
# aggression: 1.0 = pushes hard, higher peak pace + higher deg
# wet_pace: 1.0 = excellent wet performance
# overtake_ability: 1.0 = can pass in traffic (affects track position value)

DRIVER_PROFILES = {
    1:  {"name": "Max Verstappen",    "tyre_management": 0.88, "aggression": 0.92, "wet_pace": 0.82, "overtake_ability": 0.95},
    4:  {"name": "Lando Norris",      "tyre_management": 0.80, "aggression": 0.88, "wet_pace": 0.75, "overtake_ability": 0.85},
    16: {"name": "Charles Leclerc",   "tyre_management": 0.72, "aggression": 0.90, "wet_pace": 0.78, "overtake_ability": 0.82},
    44: {"name": "Lewis Hamilton",    "tyre_management": 0.91, "aggression": 0.80, "wet_pace": 0.92, "overtake_ability": 0.88},
    63: {"name": "George Russell",    "tyre_management": 0.83, "aggression": 0.78, "wet_pace": 0.80, "overtake_ability": 0.80},
    55: {"name": "Carlos Sainz",      "tyre_management": 0.85, "aggression": 0.75, "wet_pace": 0.74, "overtake_ability": 0.78},
    14: {"name": "Fernando Alonso",   "tyre_management": 0.93, "aggression": 0.82, "wet_pace": 0.88, "overtake_ability": 0.90},
    22: {"name": "Yuki Tsunoda",      "tyre_management": 0.65, "aggression": 0.85, "wet_pace": 0.70, "overtake_ability": 0.72},
    10: {"name": "Pierre Gasly",      "tyre_management": 0.74, "aggression": 0.73, "wet_pace": 0.72, "overtake_ability": 0.75},
    23: {"name": "Alex Albon",        "tyre_management": 0.78, "aggression": 0.72, "wet_pace": 0.73, "overtake_ability": 0.74},
    81: {"name": "Oscar Piastri",     "tyre_management": 0.82, "aggression": 0.80, "wet_pace": 0.76, "overtake_ability": 0.78},
    31: {"name": "Esteban Ocon",      "tyre_management": 0.76, "aggression": 0.70, "wet_pace": 0.71, "overtake_ability": 0.70},
    77: {"name": "Valtteri Bottas",   "tyre_management": 0.80, "aggression": 0.68, "wet_pace": 0.75, "overtake_ability": 0.68},
    24: {"name": "Zhou Guanyu",       "tyre_management": 0.70, "aggression": 0.65, "wet_pace": 0.67, "overtake_ability": 0.65},
    18: {"name": "Lance Stroll",      "tyre_management": 0.68, "aggression": 0.72, "wet_pace": 0.78, "overtake_ability": 0.65},
    27: {"name": "Nico Hulkenberg",   "tyre_management": 0.77, "aggression": 0.73, "wet_pace": 0.72, "overtake_ability": 0.76},
    20: {"name": "Kevin Magnussen",   "tyre_management": 0.71, "aggression": 0.80, "wet_pace": 0.70, "overtake_ability": 0.74},
    2:  {"name": "Logan Sargeant",    "tyre_management": 0.60, "aggression": 0.65, "wet_pace": 0.62, "overtake_ability": 0.60},
    3:  {"name": "Daniel Ricciardo",  "tyre_management": 0.76, "aggression": 0.78, "wet_pace": 0.74, "overtake_ability": 0.82},
    # 2025 newcomers — average profiles as placeholder
    50: {"name": "Oliver Bearman",    "tyre_management": 0.70, "aggression": 0.78, "wet_pace": 0.70, "overtake_ability": 0.72},
    87: {"name": "Isack Hadjar",      "tyre_management": 0.68, "aggression": 0.75, "wet_pace": 0.68, "overtake_ability": 0.70},
    30: {"name": "Liam Lawson",       "tyre_management": 0.72, "aggression": 0.78, "wet_pace": 0.72, "overtake_ability": 0.74},
    43: {"name": "Franco Colapinto",  "tyre_management": 0.66, "aggression": 0.76, "wet_pace": 0.68, "overtake_ability": 0.70},
    5:  {"name": "Gabriel Bortoleto", "tyre_management": 0.65, "aggression": 0.77, "wet_pace": 0.66, "overtake_ability": 0.68},
    6:  {"name": "Jack Doohan",       "tyre_management": 0.66, "aggression": 0.74, "wet_pace": 0.67, "overtake_ability": 0.68},
}

# Fuel load constants (2025 regs ~110 kg at race start)
FUEL_START_KG = 110.0
FUEL_BURN_PER_LAP_KG = 1.55       # average
FUEL_LAP_TIME_DELTA_PER_KG = 0.03  # seconds per kg of fuel

# DRS delta — average time gain when DRS open (track-dependent, this is global)
DRS_TIME_GAIN_S = 0.5   # ~0.4–0.6s across most circuits

# Clean air vs dirty air gap (seconds lost stuck in turbulent air)
DIRTY_AIR_DELTA_S = 0.3   # within 1 second of car ahead

# VSC and SC deltas
VSC_SPEED_DELTA_S = 8.0   # approximate time loss per lap under VSC
SC_SPEED_DELTA_S = 30.0   # approximate time loss per lap behind SC

# Weather model coefficients
TRACK_TEMP_OPTIMAL_C = 35.0          # compounds work best ~35°C track temp
TRACK_TEMP_COEFFICIENT = 0.01        # seconds per °C deviation from optimal
RAINFALL_WET_PENALTY_S = 15.0        # flat lap time loss in heavy rain on slicks

# Wind model
# Headwind on main straight kills top speed → slower lap
# Tailwind on main straight = free speed
# Crosswind on high-speed corners = aero instability → drivers lift
WIND_HEADWIND_COEFFICIENT  = 0.012   # s per km/h headwind
WIND_TAILWIND_COEFFICIENT  = 0.006   # s per km/h tailwind (less benefit than headwind hurts)
WIND_CROSSWIND_COEFFICIENT = 0.008   # s per km/h crosswind (stability penalty in fast corners)

def wind_lap_delta(wind_speed_kmh: float, wind_direction_deg: float,
                   circuit_orientation_deg: float = 0.0) -> float:
    """
    Compute lap time delta (seconds) from wind conditions.

    wind_direction_deg: meteorological convention (0=N, 90=E, 180=S, 270=W)
    circuit_orientation_deg: direction the main straight runs (from circuit DB)

    Returns positive = slower, negative = faster.
    """
    if wind_speed_kmh <= 0:
        return 0.0

    # Angle between wind and main straight direction
    angle_diff = abs((wind_direction_deg - circuit_orientation_deg + 180) % 360 - 180)

    # Component along straight (headwind/tailwind)
    straight_component = np.cos(np.radians(angle_diff)) * wind_speed_kmh
    # Component perpendicular (crosswind)
    cross_component = abs(np.sin(np.radians(angle_diff))) * wind_speed_kmh

    delta = 0.0
    if straight_component > 0:
        delta += straight_component * WIND_HEADWIND_COEFFICIENT   # headwind = slower
    else:
        delta += straight_component * WIND_TAILWIND_COEFFICIENT   # tailwind = faster (negative)

    delta += cross_component * WIND_CROSSWIND_COEFFICIENT
    return round(delta, 3)


@dataclass
class LapContext:
    """All inputs required to predict a single lap time."""
    lap_number: int
    total_laps: int
    compound: str
    tyre_age: int
    driver_number: int
    circuit_key: str
    # Optional enrichments
    fuel_start_kg: float = FUEL_START_KG
    drs_available: bool = True
    in_drs_train: bool = False
    clean_air: bool = True
    track_temp_c: float = TRACK_TEMP_OPTIMAL_C
    rainfall: float = 0.0
    safety_car: bool = False
    vsc: bool = False
    base_lap_time_s: float = 90.0
    # Wind
    wind_speed_kmh: float = 0.0
    wind_direction_deg: float = 0.0        # meteorological (0=N, 90=E…)
    circuit_orientation_deg: float = 0.0   # main straight heading


class LapTimeModel:
    """
    Combines all factors into a predicted lap time (seconds).
    """

    def __init__(self, deg_model: TyreDegradationModel):
        self.deg_model = deg_model

    def predict(self, ctx: LapContext) -> float:
        """Return predicted lap time in seconds for this lap."""
        t = ctx.base_lap_time_s

        # 1. Compound pace delta (fresh tyre advantage vs medium)
        t += COMPOUND_PACE_DELTA.get(ctx.compound, 0.0)

        # 2. Tyre degradation delta
        t += self.deg_model.predict(ctx.compound, ctx.tyre_age, ctx.circuit_key)

        # 3. Fuel load delta — lighter car is faster
        fuel_remaining = max(
            0.0,
            ctx.fuel_start_kg - (ctx.lap_number - 1) * FUEL_BURN_PER_LAP_KG
        )
        # Each kg = +FUEL_LAP_TIME_DELTA_PER_KG seconds vs empty car
        t += fuel_remaining * FUEL_LAP_TIME_DELTA_PER_KG

        # 4. DRS effect
        profile = DRIVER_PROFILES.get(ctx.driver_number, {})
        if ctx.drs_available and ctx.clean_air and not ctx.safety_car:
            # DRS benefit only if not in dirty air (in train = partial benefit)
            if ctx.in_drs_train:
                t -= DRS_TIME_GAIN_S * 0.5
            else:
                t -= DRS_TIME_GAIN_S

        # 5. Dirty air penalty
        if not ctx.clean_air:
            # Reduced by driver's overtake ability (aggressive drivers manage better)
            overtake_factor = profile.get("overtake_ability", 0.75)
            t += DIRTY_AIR_DELTA_S * (1.0 - overtake_factor * 0.4)

        # 6. Driver style — tyre management adjusts deg contribution
        # High tyre_management drivers effectively reduce deg by 10–15%
        tyre_mgmt = profile.get("tyre_management", 0.75)
        deg_reduction = (tyre_mgmt - 0.75) * 0.4   # +/- seconds
        t -= deg_reduction

        # 7. Weather — track temperature effect
        temp_delta = abs(ctx.track_temp_c - TRACK_TEMP_OPTIMAL_C)
        t += temp_delta * TRACK_TEMP_COEFFICIENT

        # 8. Rainfall on slicks — significant penalty
        if ctx.rainfall > 0.1 and ctx.compound in ("SOFT", "MEDIUM", "HARD"):
            t += RAINFALL_WET_PENALTY_S * ctx.rainfall

        # 9. Safety car / VSC — lap time forced up
        if ctx.safety_car:
            t += SC_SPEED_DELTA_S
        elif ctx.vsc:
            t += VSC_SPEED_DELTA_S

        # 10. Wind — headwind/tailwind/crosswind vs main straight orientation
        t += wind_lap_delta(ctx.wind_speed_kmh, ctx.wind_direction_deg,
                            ctx.circuit_orientation_deg)

        return round(t, 3)

    def predict_stint(
        self,
        compound: str,
        start_tyre_age: int,
        stint_laps: int,
        start_lap: int,
        total_laps: int,
        driver_number: int,
        circuit_key: str,
        base_lap_time_s: float = 90.0,
        weather_seq: Optional[list[dict]] = None,
        sc_laps: Optional[set[int]] = None,
        vsc_laps: Optional[set[int]] = None,
    ) -> list[float]:
        """
        Predict lap times for an entire stint.
        Returns list of lap times (seconds) for each lap in the stint.
        """
        sc_laps = sc_laps or set()
        vsc_laps = vsc_laps or set()
        lap_times = []

        for i in range(stint_laps):
            lap_num = start_lap + i
            tyre_age = start_tyre_age + i

            # Weather for this lap (use last known if sequence provided)
            weather = {}
            if weather_seq and i < len(weather_seq):
                weather = weather_seq[i]

            ctx = LapContext(
                lap_number=lap_num,
                total_laps=total_laps,
                compound=compound,
                tyre_age=tyre_age,
                driver_number=driver_number,
                circuit_key=circuit_key,
                base_lap_time_s=base_lap_time_s,
                track_temp_c=weather.get("track_temperature", TRACK_TEMP_OPTIMAL_C),
                rainfall=weather.get("rainfall", 0.0),
                wind_speed_kmh=weather.get("wind_speed", 0.0),
                wind_direction_deg=weather.get("wind_direction", 0.0),
                safety_car=(lap_num in sc_laps),
                vsc=(lap_num in vsc_laps),
            )
            lap_times.append(self.predict(ctx))

        return lap_times

    def get_driver_name(self, driver_number: int) -> str:
        return DRIVER_PROFILES.get(driver_number, {}).get("name", f"Driver #{driver_number}")

    def get_all_drivers(self) -> dict:
        return DRIVER_PROFILES
