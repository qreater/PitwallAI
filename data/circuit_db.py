"""
F1 2025 Circuit Database
========================
All 24 circuits on the 2025 calendar with:
  - DRS detection/activation zones (start/end as % of lap distance)
  - Turn-by-turn severity (1=easy, 5=extreme) → used for tyre load factor
  - Tyre degradation multiplier (calibrated from Pirelli allocations + historical data)
  - ERS harvest zones (heavy braking = best harvesting)
  - Pirelli compound allocation (which C# maps to Soft/Medium/Hard)
  - Circuit characteristics (abrasive, street, high-speed, etc.)
  - Full-throttle % (from 2025 F1DataAnalysis data)

Tyre deg multiplier scale:
  1.0 = baseline (Bahrain)
  <1.0 = lower deg than Bahrain
  >1.0 = higher deg than Bahrain

Sources: OpenF1, F1Technical, Pirelli press kits, Motor Sport Magazine 2025
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DRSZone:
    """A single DRS activation zone."""
    zone_id: int
    detection_point: str    # corner/straight name where detection loop is
    activation_point: str   # where DRS can be opened
    end_point: str          # braking zone where DRS must be closed
    straight_length_m: int  # approximate length of DRS-assisted straight
    avg_speed_gain_kmh: float = 12.5   # speed gain when DRS open (~12-15 km/h)


@dataclass
class Turn:
    """Individual corner data for tyre load modelling."""
    number: int
    name: str
    severity: int           # 1 (gentle) to 5 (extreme lateral G)
    direction: str          # L / R / chicane
    apex_speed_kmh: int
    tyre_load_factor: float # multiplier on tyre wear per lap (summed across all turns)
    ers_harvest: bool       # True = significant braking before this turn → ERS harvest


@dataclass
class Circuit:
    key: str                    # internal slug
    official_name: str
    country: str
    city: str
    lap_length_km: float
    total_laps_2025: int
    circuit_type: str           # permanent / street / semi-street
    surface_abrasiveness: str   # low / medium / high / very_high
    tyre_deg_multiplier: float  # vs Bahrain baseline = 1.0
    full_throttle_pct: float    # % of lap at full throttle (2025 data)
    base_lap_time_s: float      # reference lap time seconds
    pit_lane_loss_s: float      # total time lost in pit lane
    avg_track_temp_c: int       # typical race day track temp °C
    rain_probability: float     # 0.0 – 1.0
    safety_car_probability: float   # historical SC/VSC rate
    pirelli_soft_compound: str  # e.g. "C3"
    pirelli_medium_compound: str
    pirelli_hard_compound: str
    recommended_stops: list[int]    # typical 1-stop/2-stop options
    sprint_weekend: bool
    drs_zones: list[DRSZone]
    turns: list[Turn]
    notes: str = ""

    @property
    def tyre_load_index(self) -> float:
        """Aggregate tyre stress across all corners (used in deg model)."""
        if not self.turns:
            return 1.0
        total = sum(t.tyre_load_factor * (t.severity / 3.0) for t in self.turns)
        return round(total / len(self.turns), 3)

    @property
    def ers_harvest_corners(self) -> list[int]:
        """Turn numbers where ERS harvest is significant."""
        return [t.number for t in self.turns if t.ers_harvest]

    @property
    def drs_zone_count(self) -> int:
        return len(self.drs_zones)


# ── Circuit definitions ────────────────────────────────────────────────────
# Ordered by 2025 calendar sequence

CIRCUITS: dict[str, Circuit] = {}

def _reg(c: Circuit):
    CIRCUITS[c.key] = c
    return c


# ── 1. MELBOURNE — Albert Park ─────────────────────────────────────────────
_reg(Circuit(
    key="melbourne",
    official_name="Albert Park Circuit",
    country="Australia",
    city="Melbourne",
    lap_length_km=5.278,
    total_laps_2025=58,
    circuit_type="semi-street",
    surface_abrasiveness="low",
    tyre_deg_multiplier=0.78,
    full_throttle_pct=66.0,
    base_lap_time_s=81.0,
    pit_lane_loss_s=20.0,
    avg_track_temp_c=28,
    rain_probability=0.30,
    safety_car_probability=0.70,
    pirelli_soft_compound="C5",
    pirelli_medium_compound="C4",
    pirelli_hard_compound="C3",
    recommended_stops=[1, 2],
    sprint_weekend=False,
    drs_zones=[
        DRSZone(1, "Turn 2", "Straight after T2", "Turn 3",   450, 13.0),
        DRSZone(2, "Turn 6", "Lakeside straight",  "Turn 9",   600, 14.0),
        DRSZone(3, "Turn 13","Back straight",       "Turn 15",  500, 12.0),
        DRSZone(4, "Turn 16","Start/finish",        "Turn 1",   700, 15.0),
    ],
    turns=[
        Turn(1,  "Turn 1",            3, "R", 220, 1.3, True),
        Turn(3,  "Turn 3",            4, "L", 120, 1.6, True),
        Turn(6,  "Turn 6",            2, "R", 190, 1.1, False),
        Turn(9,  "Hairpin T9",        5, "R",  80, 1.8, True),
        Turn(11, "Turn 11",           3, "L", 165, 1.3, False),
        Turn(13, "Turn 13",           2, "R", 200, 1.1, False),
        Turn(15, "T15 chicane",       4, "chicane", 100, 1.5, True),
    ],
    notes="Resurfaced 2022. 4 DRS zones = high overtaking. Low deg, soft compounds."
))

# ── 2. SHANGHAI ────────────────────────────────────────────────────────────
_reg(Circuit(
    key="shanghai",
    official_name="Shanghai International Circuit",
    country="China",
    city="Shanghai",
    lap_length_km=5.451,
    total_laps_2025=56,
    circuit_type="permanent",
    surface_abrasiveness="medium",
    tyre_deg_multiplier=0.85,
    full_throttle_pct=64.0,
    base_lap_time_s=96.5,
    pit_lane_loss_s=23.0,
    avg_track_temp_c=22,
    rain_probability=0.25,
    safety_car_probability=0.50,
    pirelli_soft_compound="C4",
    pirelli_medium_compound="C3",
    pirelli_hard_compound="C2",
    recommended_stops=[1, 2],
    sprint_weekend=True,
    drs_zones=[
        DRSZone(1, "Turn 16", "Back straight",    "Turn 14",  1200, 16.0),
        DRSZone(2, "Turn 13", "Start/finish",     "Turn 1",    900, 14.0),
    ],
    turns=[
        Turn(1,  "T1 hairpin",      5, "R",  75, 1.9, True),
        Turn(3,  "T3",              4, "L", 110, 1.6, True),
        Turn(6,  "T6 long right",   3, "R", 170, 1.4, False),
        Turn(8,  "T7-8 esses",      4, "chicane", 140, 1.5, False),
        Turn(13, "T13 hairpin",     5, "L",  65, 2.0, True),
        Turn(16, "T16",             2, "R", 210, 1.1, False),
    ],
    notes="Ultra-long back straight (1.2km). 2-DRS zones. First sprint 2025."
))

# ── 3. BAHRAIN ─────────────────────────────────────────────────────────────
_reg(Circuit(
    key="bahrain",
    official_name="Bahrain International Circuit",
    country="Bahrain",
    city="Sakhir",
    lap_length_km=5.412,
    total_laps_2025=57,
    circuit_type="permanent",
    surface_abrasiveness="very_high",
    tyre_deg_multiplier=1.0,   # baseline
    full_throttle_pct=65.0,
    base_lap_time_s=93.5,
    pit_lane_loss_s=22.5,
    avg_track_temp_c=38,
    rain_probability=0.02,
    safety_car_probability=0.45,
    pirelli_soft_compound="C3",
    pirelli_medium_compound="C2",
    pirelli_hard_compound="C1",
    recommended_stops=[2],
    sprint_weekend=False,
    drs_zones=[
        DRSZone(1, "Turn 4",  "Main straight",  "Turn 1",  1050, 14.0),
        DRSZone(2, "Turn 11", "Straight T11-12","Turn 14",  500, 11.0),
    ],
    turns=[
        Turn(1,  "T1 right",     3, "R", 185, 1.3, True),
        Turn(4,  "T4 hairpin",   5, "R",  70, 1.9, True),
        Turn(8,  "T8 hairpin",   5, "L",  75, 1.9, True),
        Turn(10, "T10",          3, "R", 155, 1.3, False),
        Turn(11, "T11",          4, "L", 120, 1.6, True),
        Turn(13, "T13 chicane",  4, "chicane", 105, 1.5, True),
        Turn(15, "T15",          2, "L", 195, 1.1, False),
    ],
    notes="Very high abrasiveness. Primarily thermal rear deg. C1-C2-C3 allocation. Typically 2-stop."
))

# ── 4. JEDDAH ──────────────────────────────────────────────────────────────
_reg(Circuit(
    key="jeddah",
    official_name="Jeddah Corniche Circuit",
    country="Saudi Arabia",
    city="Jeddah",
    lap_length_km=6.174,
    total_laps_2025=50,
    circuit_type="street",
    surface_abrasiveness="low",
    tyre_deg_multiplier=0.82,
    full_throttle_pct=76.1,   # 2nd highest on calendar
    base_lap_time_s=90.5,
    pit_lane_loss_s=23.0,
    avg_track_temp_c=32,
    rain_probability=0.03,
    safety_car_probability=0.75,
    pirelli_soft_compound="C4",
    pirelli_medium_compound="C3",
    pirelli_hard_compound="C2",
    recommended_stops=[1, 2],
    sprint_weekend=False,
    drs_zones=[
        DRSZone(1, "Turn 27", "Start/finish",    "Turn 1",  1100, 16.0),
        DRSZone(2, "Turn 22", "Straight T22-24", "Turn 24",  800, 14.0),
        DRSZone(3, "Turn 12", "Straight T12-13", "Turn 13",  600, 12.0),
    ],
    turns=[
        Turn(2,  "T2 fast right",  2, "R", 250, 1.0, False),
        Turn(4,  "T4",             3, "L", 210, 1.2, False),
        Turn(13, "T13 hairpin",    5, "R",  80, 1.8, True),
        Turn(22, "T22",            4, "L", 120, 1.5, True),
        Turn(27, "T27 final",      3, "R", 200, 1.2, True),
    ],
    notes="Fastest street circuit. 27 corners. 3 DRS zones. Primarily wall risks → high SC probability."
))

# ── 5. SUZUKA ──────────────────────────────────────────────────────────────
_reg(Circuit(
    key="suzuka",
    official_name="Suzuka International Racing Course",
    country="Japan",
    city="Suzuka",
    lap_length_km=5.807,
    total_laps_2025=53,
    circuit_type="permanent",
    surface_abrasiveness="high",
    tyre_deg_multiplier=0.95,
    full_throttle_pct=62.0,
    base_lap_time_s=93.0,
    pit_lane_loss_s=22.0,
    avg_track_temp_c=24,
    rain_probability=0.40,
    safety_car_probability=0.55,
    pirelli_soft_compound="C3",
    pirelli_medium_compound="C2",
    pirelli_hard_compound="C1",
    recommended_stops=[1, 2],
    sprint_weekend=False,
    drs_zones=[
        DRSZone(1, "Turn 16", "Main straight", "Turn 1", 700, 12.0),
    ],
    turns=[
        Turn(1,  "T1 right",         3, "R", 200, 1.3, True),
        Turn(3,  "S-curves start",   5, "chicane", 190, 1.7, False),
        Turn(7,  "Dunlop",           3, "R", 175, 1.3, False),
        Turn(11, "Hairpin",          5, "L",  70, 1.9, True),
        Turn(13, "Spoon curve",      4, "L", 170, 1.5, False),
        Turn(16, "130R",             5, "L", 290, 2.0, False),
        Turn(17, "Chicane",          4, "chicane", 130, 1.6, True),
    ],
    notes="Only 1 DRS zone — overtaking very difficult. 130R is one of F1's most demanding corners."
))

# ── 6. MIAMI ───────────────────────────────────────────────────────────────
_reg(Circuit(
    key="miami",
    official_name="Miami International Autodrome",
    country="USA",
    city="Miami",
    lap_length_km=5.412,
    total_laps_2025=57,
    circuit_type="street",
    surface_abrasiveness="medium",
    tyre_deg_multiplier=0.88,
    full_throttle_pct=60.0,
    base_lap_time_s=91.0,
    pit_lane_loss_s=21.0,
    avg_track_temp_c=42,
    rain_probability=0.20,
    safety_car_probability=0.60,
    pirelli_soft_compound="C4",
    pirelli_medium_compound="C3",
    pirelli_hard_compound="C2",
    recommended_stops=[1, 2],
    sprint_weekend=True,
    drs_zones=[
        DRSZone(1, "Turn 11", "Long straight",    "Turn 1",   900, 14.0),
        DRSZone(2, "Turn 16", "Back straight",    "Turn 17",  600, 12.0),
        DRSZone(3, "Turn 3",  "Start/finish",     "Turn 4",   700, 13.0),
    ],
    turns=[
        Turn(1,  "T1",         4, "R", 130, 1.6, True),
        Turn(6,  "T6",         3, "L", 185, 1.3, False),
        Turn(11, "T11",        3, "R", 220, 1.2, False),
        Turn(14, "T14-15",     4, "chicane", 110, 1.5, True),
        Turn(17, "T17",        5, "L",  80, 1.8, True),
    ],
    notes="Very high track temps → significant tyre overheating risk. 3 DRS zones."
))

# ── 7. IMOLA ───────────────────────────────────────────────────────────────
_reg(Circuit(
    key="imola",
    official_name="Autodromo Enzo e Dino Ferrari",
    country="Italy",
    city="Imola",
    lap_length_km=4.909,
    total_laps_2025=63,
    circuit_type="permanent",
    surface_abrasiveness="medium",
    tyre_deg_multiplier=0.87,
    full_throttle_pct=65.0,
    base_lap_time_s=77.0,
    pit_lane_loss_s=22.0,
    avg_track_temp_c=30,
    rain_probability=0.30,
    safety_car_probability=0.55,
    pirelli_soft_compound="C4",
    pirelli_medium_compound="C3",
    pirelli_hard_compound="C2",
    recommended_stops=[1, 2],
    sprint_weekend=False,
    drs_zones=[
        DRSZone(1, "Tamburello", "Main straight",   "Tosa",    900, 14.0),
        DRSZone(2, "Rivazza",    "Back straight",   "Variante",700, 12.0),
    ],
    turns=[
        Turn(1,  "Tamburello",  2, "R", 270, 1.0, False),
        Turn(2,  "Tosa",        5, "L",  80, 1.9, True),
        Turn(5,  "Piratella",   4, "L", 175, 1.5, False),
        Turn(9,  "Acque Minerali", 4, "chicane", 115, 1.5, True),
        Turn(14, "Rivazza",     4, "L", 145, 1.5, True),
    ],
    notes="Narrow track limits overtaking. Good gravel traps. Medium deg."
))

# ── 8. MONACO ─────────────────────────────────────────────────────────────
_reg(Circuit(
    key="monaco",
    official_name="Circuit de Monaco",
    country="Monaco",
    city="Monte Carlo",
    lap_length_km=3.337,
    total_laps_2025=78,
    circuit_type="street",
    surface_abrasiveness="low",
    tyre_deg_multiplier=0.45,   # lowest on calendar — slow corners
    full_throttle_pct=38.0,     # lowest full-throttle %
    base_lap_time_s=74.5,
    pit_lane_loss_s=24.0,
    avg_track_temp_c=27,
    rain_probability=0.25,
    safety_car_probability=0.80,
    pirelli_soft_compound="C6",   # new 2025 ultra-soft
    pirelli_medium_compound="C5",
    pirelli_hard_compound="C4",
    recommended_stops=[2],   # 2025 regs mandate minimum 2 stops at Monaco
    sprint_weekend=False,
    drs_zones=[
        DRSZone(1, "Anthony Noghes", "Start/finish", "Sainte Devote", 600, 10.0),
    ],
    turns=[
        Turn(1,  "Sainte Devote",  4, "R", 105, 1.5, True),
        Turn(5,  "Massenet",       3, "R", 220, 1.1, False),
        Turn(6,  "Casino",         3, "L", 160, 1.2, False),
        Turn(10, "Mirabeau",       5, "R",  75, 1.7, True),
        Turn(12, "Fairmont hairpin",5,"R",  50, 1.8, True),
        Turn(15, "Tabac",          4, "R", 165, 1.4, False),
        Turn(19, "La Rascasse",    5, "L",  60, 1.7, True),
    ],
    notes="Mandatory 2-stop 2025. C6 introduced for street circuits. Lowest deg on calendar."
))

# ── 9. MONTREAL ────────────────────────────────────────────────────────────
_reg(Circuit(
    key="montreal",
    official_name="Circuit Gilles-Villeneuve",
    country="Canada",
    city="Montréal",
    lap_length_km=4.361,
    total_laps_2025=70,
    circuit_type="semi-street",
    surface_abrasiveness="low",
    tyre_deg_multiplier=0.60,
    full_throttle_pct=68.0,
    base_lap_time_s=75.0,
    pit_lane_loss_s=23.5,
    avg_track_temp_c=26,
    rain_probability=0.40,
    safety_car_probability=0.70,
    pirelli_soft_compound="C5",
    pirelli_medium_compound="C4",
    pirelli_hard_compound="C3",
    recommended_stops=[1, 2],
    sprint_weekend=False,
    drs_zones=[
        DRSZone(1, "Chicane du Casino","Main straight",  "Turn 1",   950, 14.0),
        DRSZone(2, "Turn 9",          "Back straight",   "Hairpin",  600, 11.0),
    ],
    turns=[
        Turn(1,  "T1",            4, "R", 110, 1.6, True),
        Turn(3,  "L'Epingle",     5, "L",  70, 1.8, True),
        Turn(6,  "Pont de la Concorde", 3, "L", 195, 1.2, False),
        Turn(8,  "Chicane",       4, "chicane", 100, 1.5, True),
        Turn(13, "Wall of Champions",   5, "R", 180, 1.5, True),
    ],
    notes="Low deg, heavy braking. Wall of Champions = high SC risk. Soft compounds."
))

# ── 10. BARCELONA ─────────────────────────────────────────────────────────
_reg(Circuit(
    key="barcelona",
    official_name="Circuit de Barcelona-Catalunya",
    country="Spain",
    city="Barcelona",
    lap_length_km=4.675,
    total_laps_2025=66,
    circuit_type="permanent",
    surface_abrasiveness="medium",
    tyre_deg_multiplier=0.93,
    full_throttle_pct=58.0,
    base_lap_time_s=83.0,
    pit_lane_loss_s=22.0,
    avg_track_temp_c=36,
    rain_probability=0.15,
    safety_car_probability=0.40,
    pirelli_soft_compound="C4",
    pirelli_medium_compound="C3",
    pirelli_hard_compound="C2",
    recommended_stops=[1, 2],
    sprint_weekend=False,
    drs_zones=[
        DRSZone(1, "Turn 16", "Main straight", "Turn 1",  950, 13.0),
        DRSZone(2, "Turn 5",  "Back straight", "Turn 7",  600, 11.0),
    ],
    turns=[
        Turn(1,  "Turn 1",       4, "R", 120, 1.6, True),
        Turn(3,  "Turn 3",       4, "L", 170, 1.4, False),
        Turn(5,  "Turn 5",       5, "R", 215, 1.6, False),
        Turn(9,  "Turn 9",       5, "L", 130, 1.7, False),
        Turn(13, "Turn 13",      3, "R", 175, 1.2, False),
        Turn(16, "Final corner", 4, "L", 200, 1.4, False),
    ],
    notes="Sector 3 revamped. Long back-to-back corners load same side of tyre."
))

# ── 11. RED BULL RING ─────────────────────────────────────────────────────
_reg(Circuit(
    key="red_bull_ring",
    official_name="Red Bull Ring",
    country="Austria",
    city="Spielberg",
    lap_length_km=4.318,
    total_laps_2025=71,
    circuit_type="permanent",
    surface_abrasiveness="medium",
    tyre_deg_multiplier=0.80,
    full_throttle_pct=71.0,
    base_lap_time_s=67.0,
    pit_lane_loss_s=19.0,   # shortest pit lane
    avg_track_temp_c=30,
    rain_probability=0.35,
    safety_car_probability=0.50,
    pirelli_soft_compound="C4",
    pirelli_medium_compound="C3",
    pirelli_hard_compound="C2",
    recommended_stops=[1, 2],
    sprint_weekend=False,
    drs_zones=[
        DRSZone(1, "Turn 9", "Main straight",  "Turn 1",  900, 14.0),
        DRSZone(2, "Turn 3", "Straight T3-4",  "Turn 4",  500, 12.0),
    ],
    turns=[
        Turn(1,  "Turn 1 uphill",   4, "R", 130, 1.5, True),
        Turn(2,  "Turn 2",          3, "R", 175, 1.2, False),
        Turn(3,  "Turn 3",          4, "L", 195, 1.4, False),
        Turn(4,  "Turn 4 hairpin",  5, "L",  85, 1.8, True),
        Turn(9,  "Turn 9 final",    4, "R", 200, 1.4, True),
    ],
    notes="Short lap = many laps. Low pit loss. Elevation changes affect tyre temps."
))

# ── 12. SILVERSTONE ────────────────────────────────────────────────────────
_reg(Circuit(
    key="silverstone",
    official_name="Silverstone Circuit",
    country="Great Britain",
    city="Silverstone",
    lap_length_km=5.891,
    total_laps_2025=52,
    circuit_type="permanent",
    surface_abrasiveness="high",
    tyre_deg_multiplier=1.05,   # 2nd highest deg on calendar
    full_throttle_pct=63.0,
    base_lap_time_s=90.0,
    pit_lane_loss_s=21.5,
    avg_track_temp_c=28,
    rain_probability=0.40,
    safety_car_probability=0.45,
    pirelli_soft_compound="C4",
    pirelli_medium_compound="C3",
    pirelli_hard_compound="C2",
    recommended_stops=[2],
    sprint_weekend=False,
    drs_zones=[
        DRSZone(1, "Abbey",     "Wellington straight", "Brooklands", 800, 13.0),
        DRSZone(2, "Loop",      "Hanger straight",     "Stowe",      900, 14.0),
    ],
    turns=[
        Turn(1,  "Copse",       5, "R", 285, 1.8, False),
        Turn(3,  "Maggotts",    5, "chicane", 270, 1.9, False),
        Turn(6,  "Becketts",    5, "chicane", 250, 2.0, False),
        Turn(9,  "Stowe",       4, "R", 195, 1.5, True),
        Turn(13, "Club",        4, "L", 185, 1.4, False),
        Turn(15, "Abbey",       3, "L", 230, 1.2, False),
        Turn(17, "Woodcote",    3, "R", 240, 1.2, False),
    ],
    notes="Maggotts-Becketts complex is highest lateral-G sequence on calendar → extreme tyre stress."
))

# ── 13. HUNGARORING ────────────────────────────────────────────────────────
_reg(Circuit(
    key="hungaroring",
    official_name="Hungaroring",
    country="Hungary",
    city="Budapest",
    lap_length_km=4.381,
    total_laps_2025=70,
    circuit_type="permanent",
    surface_abrasiveness="medium",
    tyre_deg_multiplier=0.90,
    full_throttle_pct=52.0,
    base_lap_time_s=80.0,
    pit_lane_loss_s=22.5,
    avg_track_temp_c=42,
    rain_probability=0.25,
    safety_car_probability=0.45,
    pirelli_soft_compound="C4",
    pirelli_medium_compound="C3",
    pirelli_hard_compound="C2",
    recommended_stops=[1, 2],
    sprint_weekend=False,
    drs_zones=[
        DRSZone(1, "Turn 14", "Main straight", "Turn 1", 950, 12.0),
    ],
    turns=[
        Turn(1,  "T1",    4, "R", 105, 1.6, True),
        Turn(4,  "T4",    5, "L",  90, 1.7, True),
        Turn(6,  "T6",    4, "R", 160, 1.4, False),
        Turn(11, "T11",   5, "L", 125, 1.6, True),
        Turn(14, "T14",   3, "R", 195, 1.2, False),
    ],
    notes="Monaco of permanent tracks. Very hard to overtake. High temps stress tyres."
))

# ── 14. SPA ────────────────────────────────────────────────────────────────
_reg(Circuit(
    key="spa",
    official_name="Circuit de Spa-Francorchamps",
    country="Belgium",
    city="Stavelot",
    lap_length_km=7.004,
    total_laps_2025=44,
    circuit_type="permanent",
    surface_abrasiveness="medium",
    tyre_deg_multiplier=0.85,
    full_throttle_pct=69.0,
    base_lap_time_s=106.0,
    pit_lane_loss_s=19.0,
    avg_track_temp_c=22,
    rain_probability=0.55,
    safety_car_probability=0.65,
    pirelli_soft_compound="C4",
    pirelli_medium_compound="C3",
    pirelli_hard_compound="C2",
    recommended_stops=[1, 2],
    sprint_weekend=True,
    drs_zones=[
        DRSZone(1, "Bus Stop",   "Kemmel straight",  "Les Combes",  1100, 16.0),
        DRSZone(2, "Blanchimont","Straight",          "Bus Stop",     700, 13.0),
    ],
    turns=[
        Turn(1,  "La Source",   5, "R",  85, 1.7, True),
        Turn(2,  "Eau Rouge",   5, "L", 290, 1.9, False),
        Turn(3,  "Raidillon",   5, "R", 310, 2.0, False),
        Turn(9,  "Pouhon",      5, "L", 235, 1.8, False),
        Turn(15, "Blanchimont", 4, "L", 295, 1.6, False),
        Turn(18, "Bus Stop",    4, "chicane", 95, 1.5, True),
    ],
    notes="Longest circuit. Eau Rouge/Raidillon peak lateral G. Wet race common."
))

# ── 15. ZANDVOORT ─────────────────────────────────────────────────────────
_reg(Circuit(
    key="zandvoort",
    official_name="Circuit Zandvoort",
    country="Netherlands",
    city="Zandvoort",
    lap_length_km=4.259,
    total_laps_2025=72,
    circuit_type="permanent",
    surface_abrasiveness="medium",
    tyre_deg_multiplier=0.91,
    full_throttle_pct=56.0,
    base_lap_time_s=72.5,
    pit_lane_loss_s=21.0,
    avg_track_temp_c=24,
    rain_probability=0.35,
    safety_car_probability=0.50,
    pirelli_soft_compound="C4",
    pirelli_medium_compound="C3",
    pirelli_hard_compound="C2",
    recommended_stops=[1, 2],
    sprint_weekend=False,
    drs_zones=[
        DRSZone(1, "Turn 12", "Main straight", "Turn 1",  700, 11.0),
        DRSZone(2, "Turn 9",  "Straight",      "Turn 10", 400, 10.0),
    ],
    turns=[
        Turn(1,  "Tarzanbocht",  5, "R",  90, 1.7, True),
        Turn(3,  "Hugenholtzbocht",4,"R", 195, 1.4, False),
        Turn(7,  "Renaultbocht", 5, "L", 160, 1.6, False),
        Turn(9,  "Arie Luyendyk",5, "banked", 270, 1.8, False),
        Turn(14, "Audi S",       5, "banked", 250, 1.8, False),
    ],
    notes="Two banked corners (Hugenholtz + Arie Luyendyk). Narrow = hard to overtake."
))

# ── 16. MONZA ─────────────────────────────────────────────────────────────
_reg(Circuit(
    key="monza",
    official_name="Autodromo Nazionale Monza",
    country="Italy",
    city="Monza",
    lap_length_km=5.793,
    total_laps_2025=53,
    circuit_type="permanent",
    surface_abrasiveness="low",
    tyre_deg_multiplier=0.65,   # low deg, high pit-lane loss → 1-stop preferred
    full_throttle_pct=77.6,     # highest on 2025 calendar
    base_lap_time_s=81.5,
    pit_lane_loss_s=16.5,       # shortest overall pit loss due to layout
    avg_track_temp_c=32,
    rain_probability=0.25,
    safety_car_probability=0.45,
    pirelli_soft_compound="C4",
    pirelli_medium_compound="C3",
    pirelli_hard_compound="C2",
    recommended_stops=[1],
    sprint_weekend=False,
    drs_zones=[
        DRSZone(1, "Parabolica",    "Main straight",   "Turn 1",   1100, 16.0),
        DRSZone(2, "Turn 4",        "Curva Grande",    "Turn 6",    900, 15.0),
    ],
    turns=[
        Turn(1,  "Prima Variante",   5, "chicane",  90, 1.7, True),
        Turn(4,  "Seconda Variante", 4, "chicane", 110, 1.5, True),
        Turn(7,  "Lesmo 1",          4, "R",       180, 1.4, False),
        Turn(8,  "Lesmo 2",          4, "R",       195, 1.4, False),
        Turn(10, "Ascari",           5, "chicane", 155, 1.6, True),
        Turn(11, "Parabolica",       4, "R",       255, 1.3, False),
    ],
    notes="Temple of Speed. 77.6% full throttle. Low downforce. Graining risk on fresh surface (2024)."
))

# ── 17. BAKU ──────────────────────────────────────────────────────────────
_reg(Circuit(
    key="baku",
    official_name="Baku City Circuit",
    country="Azerbaijan",
    city="Baku",
    lap_length_km=6.003,
    total_laps_2025=51,
    circuit_type="street",
    surface_abrasiveness="low",
    tyre_deg_multiplier=0.60,
    full_throttle_pct=70.0,
    base_lap_time_s=105.0,
    pit_lane_loss_s=21.5,
    avg_track_temp_c=30,
    rain_probability=0.10,
    safety_car_probability=0.80,
    pirelli_soft_compound="C4",
    pirelli_medium_compound="C3",
    pirelli_hard_compound="C2",
    recommended_stops=[1, 2],
    sprint_weekend=False,
    drs_zones=[
        DRSZone(1, "Turn 20", "Start/finish",  "Turn 1",  2200, 18.0),  # longest DRS straight
        DRSZone(2, "Turn 8",  "Straight T8",   "Turn 9",   500, 11.0),
    ],
    turns=[
        Turn(1,  "T1",          4, "R", 110, 1.5, True),
        Turn(3,  "T3 castle",   5, "L", 115, 1.5, True),
        Turn(8,  "T8",          3, "R", 225, 1.1, False),
        Turn(15, "T15-16",      4, "chicane", 100, 1.5, True),
        Turn(20, "T20 final",   3, "L", 270, 1.1, True),
    ],
    notes="Longest DRS straight (2.2km). High SC probability from crashes. Castle section very narrow."
))

# ── 18. SINGAPORE ─────────────────────────────────────────────────────────
_reg(Circuit(
    key="singapore",
    official_name="Marina Bay Street Circuit",
    country="Singapore",
    city="Singapore",
    lap_length_km=4.940,
    total_laps_2025=62,
    circuit_type="street",
    surface_abrasiveness="low",
    tyre_deg_multiplier=0.70,
    full_throttle_pct=45.0,
    base_lap_time_s=101.5,
    pit_lane_loss_s=24.5,
    avg_track_temp_c=36,
    rain_probability=0.45,
    safety_car_probability=0.85,
    pirelli_soft_compound="C5",
    pirelli_medium_compound="C4",
    pirelli_hard_compound="C3",
    recommended_stops=[1, 2],
    sprint_weekend=False,
    drs_zones=[
        DRSZone(1, "Turn 29", "Start/finish",  "Turn 1",   700, 11.0),
        DRSZone(2, "Turn 8",  "Straight",       "Turn 9",   500, 10.0),
        DRSZone(3, "Turn 22", "Straight",       "Turn 23",  400,  9.0),
    ],
    turns=[
        Turn(1,  "T1",           4, "R", 105, 1.5, True),
        Turn(3,  "T3 Anderson",  5, "L",  80, 1.7, True),
        Turn(10, "T10",          5, "R",  75, 1.7, True),
        Turn(18, "T18",          4, "L", 115, 1.4, True),
        Turn(23, "T23",          3, "R", 170, 1.2, False),
    ],
    notes="Hottest race on calendar at night. High SC probability. 3-stop possible."
))

# ── 19. AMERICAS (COTA) ────────────────────────────────────────────────────
_reg(Circuit(
    key="americas",
    official_name="Circuit of the Americas",
    country="USA",
    city="Austin",
    lap_length_km=5.513,
    total_laps_2025=56,
    circuit_type="permanent",
    surface_abrasiveness="high",
    tyre_deg_multiplier=0.92,
    full_throttle_pct=62.0,
    base_lap_time_s=96.0,
    pit_lane_loss_s=22.0,
    avg_track_temp_c=35,
    rain_probability=0.30,
    safety_car_probability=0.55,
    pirelli_soft_compound="C4",
    pirelli_medium_compound="C3",
    pirelli_hard_compound="C2",
    recommended_stops=[1, 2],
    sprint_weekend=True,
    drs_zones=[
        DRSZone(1, "Turn 11", "Main straight",  "Turn 1",  1100, 14.0),
        DRSZone(2, "Turn 8",  "Back straight",  "Turn 9",   600, 12.0),
    ],
    turns=[
        Turn(1,  "Turn 1 uphill",  4, "L", 125, 1.5, True),
        Turn(3,  "S-curves",       5, "chicane", 225, 1.7, False),
        Turn(9,  "T9 hairpin",     5, "L",  80, 1.8, True),
        Turn(12, "T12-13",         4, "chicane", 140, 1.5, True),
        Turn(19, "T19 final",      4, "L", 195, 1.4, False),
    ],
    notes="S-curves and T9 hairpin = high deg. DRS zones allow good overtaking."
))

# ── 20. MEXICO ─────────────────────────────────────────────────────────────
_reg(Circuit(
    key="mexico",
    official_name="Autodromo Hermanos Rodriguez",
    country="Mexico",
    city="Mexico City",
    lap_length_km=4.304,
    total_laps_2025=71,
    circuit_type="permanent",
    surface_abrasiveness="low",
    tyre_deg_multiplier=0.75,
    full_throttle_pct=61.0,
    base_lap_time_s=79.5,
    pit_lane_loss_s=22.0,
    avg_track_temp_c=28,
    rain_probability=0.20,
    safety_car_probability=0.50,
    pirelli_soft_compound="C4",
    pirelli_medium_compound="C3",
    pirelli_hard_compound="C2",
    recommended_stops=[1, 2],
    sprint_weekend=False,
    drs_zones=[
        DRSZone(1, "Turn 16", "Main straight",   "Turn 1",  1200, 14.0),
        DRSZone(2, "Turn 4",  "Straight T4-6",   "Turn 6",   600, 12.0),
    ],
    turns=[
        Turn(1,  "T1 Peraltada",  5, "L", 250, 1.5, False),
        Turn(4,  "T4",            4, "R", 105, 1.5, True),
        Turn(7,  "Esses",         5, "chicane", 180, 1.6, False),
        Turn(13, "T13",           4, "L", 115, 1.4, True),
        Turn(16, "Stadium final", 4, "R", 185, 1.3, True),
    ],
    notes="2000m altitude = ~20% less engine power. Reduced aero efficiency. Lower deg."
))

# ── 21. INTERLAGOS ─────────────────────────────────────────────────────────
_reg(Circuit(
    key="interlagos",
    official_name="Autodromo Jose Carlos Pace",
    country="Brazil",
    city="São Paulo",
    lap_length_km=4.309,
    total_laps_2025=71,
    circuit_type="permanent",
    surface_abrasiveness="high",
    tyre_deg_multiplier=0.88,
    full_throttle_pct=65.0,
    base_lap_time_s=72.0,
    pit_lane_loss_s=21.0,
    avg_track_temp_c=34,
    rain_probability=0.50,
    safety_car_probability=0.65,
    pirelli_soft_compound="C4",
    pirelli_medium_compound="C3",
    pirelli_hard_compound="C2",
    recommended_stops=[1, 2],
    sprint_weekend=True,
    drs_zones=[
        DRSZone(1, "Juncao",     "Main straight",    "Senna S",    1100, 15.0),
        DRSZone(2, "Turn 4",     "Back straight",    "Turn 6",      500, 11.0),
    ],
    turns=[
        Turn(1,  "Senna S",      5, "chicane",  80, 1.8, True),
        Turn(4,  "Descida do Lago",4,"R", 195, 1.4, False),
        Turn(8,  "Ferradura",    5, "L",  85, 1.7, True),
        Turn(11, "Laranja",      4, "R", 170, 1.4, False),
        Turn(13, "Juncao",       4, "L", 210, 1.4, True),
    ],
    notes="Anti-clockwise. Frequent rain + SC. Best overtaking in F1 per historical data."
))

# ── 22. LAS VEGAS ──────────────────────────────────────────────────────────
_reg(Circuit(
    key="las_vegas",
    official_name="Las Vegas Strip Circuit",
    country="USA",
    city="Las Vegas",
    lap_length_km=6.201,
    total_laps_2025=50,
    circuit_type="street",
    surface_abrasiveness="low",
    tyre_deg_multiplier=0.58,
    full_throttle_pct=72.0,
    base_lap_time_s=96.0,
    pit_lane_loss_s=22.5,
    avg_track_temp_c=12,   # night race, cold temperatures
    rain_probability=0.05,
    safety_car_probability=0.55,
    pirelli_soft_compound="C4",
    pirelli_medium_compound="C3",
    pirelli_hard_compound="C2",
    recommended_stops=[1],
    sprint_weekend=False,
    drs_zones=[
        DRSZone(1, "Turn 12", "Straight (Strip)",  "Turn 14",  1900, 17.0),
        DRSZone(2, "Turn 17", "Start/finish",       "Turn 1",    900, 14.0),
    ],
    turns=[
        Turn(1,  "T1",          4, "R", 105, 1.5, True),
        Turn(5,  "T5",          3, "L", 225, 1.2, False),
        Turn(12, "T12",         3, "R", 270, 1.1, False),
        Turn(14, "T14",         4, "L",  90, 1.6, True),
        Turn(17, "T17",         3, "R", 265, 1.1, True),
    ],
    notes="Cold night race → cold tyres → graining risk. Longest straight (1.9km). Low overall deg."
))

# ── 23. LUSAIL ─────────────────────────────────────────────────────────────
_reg(Circuit(
    key="lusail",
    official_name="Lusail International Circuit",
    country="Qatar",
    city="Lusail",
    lap_length_km=5.380,
    total_laps_2025=57,
    circuit_type="permanent",
    surface_abrasiveness="high",
    tyre_deg_multiplier=1.08,   # 3rd highest deg on calendar — extreme heat + abrasion
    full_throttle_pct=67.0,
    base_lap_time_s=84.0,
    pit_lane_loss_s=21.5,
    avg_track_temp_c=48,   # extreme track temps, especially in October
    rain_probability=0.02,
    safety_car_probability=0.40,
    pirelli_soft_compound="C3",
    pirelli_medium_compound="C2",
    pirelli_hard_compound="C1",
    recommended_stops=[2, 3],
    sprint_weekend=True,
    drs_zones=[
        DRSZone(1, "Turn 16", "Main straight",   "Turn 1",  1068, 14.0),
        DRSZone(2, "Turn 10", "Back straight",   "Turn 12",  600, 11.0),
    ],
    turns=[
        Turn(1,  "T1",     4, "R", 110, 1.6, True),
        Turn(3,  "T3",     5, "L", 185, 1.7, False),
        Turn(6,  "T6",     5, "R", 215, 1.7, False),
        Turn(10, "T10",    4, "L", 145, 1.5, True),
        Turn(14, "T14",    4, "R", 160, 1.5, False),
        Turn(16, "T16",    3, "L", 235, 1.2, True),
    ],
    notes="2023: extreme 3-stop race. Track temps up to 50°C. Hardest compounds."
))

# ── 24. YAS MARINA ────────────────────────────────────────────────────────
_reg(Circuit(
    key="yas_marina",
    official_name="Yas Marina Circuit",
    country="UAE",
    city="Abu Dhabi",
    lap_length_km=5.281,
    total_laps_2025=58,
    circuit_type="permanent",
    surface_abrasiveness="medium",
    tyre_deg_multiplier=0.78,
    full_throttle_pct=64.0,
    base_lap_time_s=84.5,
    pit_lane_loss_s=21.0,
    avg_track_temp_c=32,
    rain_probability=0.02,
    safety_car_probability=0.40,
    pirelli_soft_compound="C4",
    pirelli_medium_compound="C3",
    pirelli_hard_compound="C2",
    recommended_stops=[1, 2],
    sprint_weekend=False,
    drs_zones=[
        DRSZone(1, "Turn 21", "Start/finish",    "Turn 1",   900, 13.0),
        DRSZone(2, "Turn 9",  "Back straight",   "Turn 11",  600, 12.0),
        DRSZone(3, "Turn 5",  "Middle straight", "Turn 6",   500, 11.0),
    ],
    turns=[
        Turn(1,  "T1",     3, "R", 200, 1.2, True),
        Turn(5,  "T5",     4, "L", 120, 1.5, True),
        Turn(7,  "T7",     3, "R", 185, 1.2, False),
        Turn(9,  "T9",     4, "L", 145, 1.4, True),
        Turn(13, "T13",    4, "R", 125, 1.4, True),
        Turn(21, "T21",    3, "L", 215, 1.2, True),
    ],
    notes="Season finale. 3 DRS zones. Night race. Medium deg, reliable race."
))


# ── Lookup helpers ─────────────────────────────────────────────────────────

def get_circuit(key: str) -> Optional[Circuit]:
    return CIRCUITS.get(key)

def list_circuits() -> list[str]:
    return list(CIRCUITS.keys())

def circuits_by_deg_level(threshold_low: float = 0.70, threshold_high: float = 0.95):
    """Group circuits into low/medium/high tyre degradation categories."""
    low  = [k for k, c in CIRCUITS.items() if c.tyre_deg_multiplier < threshold_low]
    med  = [k for k, c in CIRCUITS.items() if threshold_low <= c.tyre_deg_multiplier < threshold_high]
    high = [k for k, c in CIRCUITS.items() if c.tyre_deg_multiplier >= threshold_high]
    return {"low": sorted(low), "medium": sorted(med), "high": sorted(high)}

def get_drs_zones(circuit_key: str) -> list[DRSZone]:
    c = CIRCUITS.get(circuit_key)
    return c.drs_zones if c else []

def get_ers_harvest_turns(circuit_key: str) -> list[int]:
    c = CIRCUITS.get(circuit_key)
    return c.ers_harvest_corners if c else []
