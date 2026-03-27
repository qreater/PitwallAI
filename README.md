# PitwallAI

A full-stack Formula 1 race strategy simulation engine built on 2025 regulations,
real OpenF1 telemetry data, and a reinforcement learning pit decision agent.

---

## Architecture

```
f1_strategist/
├── data/
│   ├── openf1_client.py     OpenF1 API wrapper (stints, laps, weather, SC, pit)
│   └── circuit_db.py        All 24 circuits — DRS zones, turns, deg multipliers
├── engine/
│   ├── tyre_model.py        Per-compound polynomial deg curves + cliff detection
│   ├── lap_time_model.py    Full lap time predictor (9 factors)
│   ├── race_simulator.py    Lap-by-lap race simulation engine
│   ├── optimizer.py         Monte Carlo strategy optimizer
│   └── weather.py           Live weather API + rain crossover calculator
├── ml/
│   └── rl_agent.py          PPO reinforcement learning pit agent (Gymnasium)
├── server.py                FastAPI REST + WebSocket race brain
├── main.py                  CLI entrypoint
└── requirements.txt
```

---

## Features

### Tyre Degradation Model (`engine/tyre_model.py`)
- Fetches real stint + lap data from OpenF1 API
- Fits **degree-2 polynomial** degradation curves per compound per circuit
- **Cliff detection**: identifies lap where degradation rate exceeds 3.5× nominal
- Falls back to calibrated Pirelli baseline priors when data is sparse
- Compounds: SOFT, MEDIUM, HARD, INTERMEDIATE, WET
- Per-circuit deg multipliers calibrated from Pirelli compound allocation choices

### Lap Time Model (`engine/lap_time_model.py`)
9 factors stacked per lap:
1. **Compound pace delta** — fresh tyre speed vs MEDIUM baseline
2. **Tyre degradation** — polynomial deg curve output
3. **Fuel load delta** — 0.03s/lap per kg, ~1.55 kg/lap burn rate
4. **DRS effect** — 0.5s gain in clean air, 0.25s in DRS train
5. **Dirty air penalty** — 0.3s loss within 1s of car ahead
6. **Driver style** — tyre management score adjusts effective deg per lap
7. **Track temperature** — 0.01s per °C deviation from 35°C optimal
8. **Rainfall** — up to 15s/lap penalty for slicks in rain
9. **Safety car / VSC** — full SC adds ~30s/lap, VSC ~8s/lap

### Driver Profiles (`engine/lap_time_model.py`)
25 drivers with individual ratings (0–1):
- `tyre_management` — ability to nurse tyres, reduces effective degradation
- `aggression` — peak pace but higher deg rate
- `wet_pace` — relative performance in wet conditions
- `overtake_ability` — reduces dirty air penalty (can defend track position better)

### Race Simulator (`engine/race_simulator.py`)
- Full lap-by-lap simulation for any `Strategy` object
- Pit stop time: circuit-specific pit lane loss (16.5s Monza → 24.5s Singapore)
- Pit time variance: ±0.2s random wheelgun/traffic noise
- Stochastic SC generation: Poisson-like model, ~SC every 55 laps
- **Undercut window**: `undercut_window(gap, pit_loss, compound_delta, laps_remaining)`
- **Overcut window**: `overcut_window(gap_behind, pit_loss, pace_advantage, laps_remaining)`
- 2025 regulation check: must use ≥2 different dry compounds in dry race

### Strategy Optimizer (`engine/optimizer.py`)
- Generates all 1-stop, 2-stop, and 3-stop compound permutations
- Pit windows centered on fitted optimal stint lengths ± 8 laps
- Scores each strategy via **Monte Carlo** (default 200 runs, stochastic SC)
- Filters regulation non-compliant strategies
- `live_reoptimize()`: re-runs from current lap/compound/gap mid-race
- Returns ranked `StrategyScore` objects with mean/std/best/worst time + win rate

### Track Database (`data/circuit_db.py`)
All 24 circuits on the 2025 calendar:
- **DRS zones**: detection point, activation point, end point, straight length (m)
- **Turns**: number, name, severity (1–5), direction, apex speed, tyre load factor, ERS harvest flag
- **Pirelli compound allocations**: actual C1–C6 designation per circuit
- **Circuit characteristics**: abrasiveness, type (permanent/street/semi-street)
- Full-throttle percentage (from 2025 F1DataAnalysis data)
- Tyre deg multiplier (Lusail 1.08× → Monaco 0.45×)
- Historical rain probability and SC probability per circuit

### Weather Module (`engine/weather.py`)
- **Live weather**: Open-Meteo API (free, no API key required)
  - Track temperature, air temperature, humidity, wind, rainfall rate
  - Rain probability for next 30 minutes
- **Crossover calculator**: finds lap at which intermediate tyres become faster than slicks
  - Pirelli crossover model: slick penalty + inter improvement curves
  - Hysteresis: different thresholds for slick→inter vs inter→slick (0.30 vs 0.08 mm/h)
  - Temperature correction: cold track shifts crossover earlier
  - Handles all transitions: slick↔inter↔wet

### ML Agent (`ml/rl_agent.py`)
PPO reinforcement learning agent via Stable-Baselines3:
- **18-dimensional observation space**: lap fraction, tyre age, compound, fuel, gaps,
  deg delta, cliff proximity, weather, SC state, driver profile, circuit deg multiplier
- **6 discrete actions**: stay out / pit for S / pit for M / pit for H / pit for inter / pit for wet
- **Reward function**: −lap_time (scaled) + cliff penalty + reg penalty + excessive stop penalty
- Trained with γ=0.995 (future laps weighted heavily), entropy bonus for exploration
- Network: MLP 256→256→128

### FastAPI Server (`server.py`)
```
GET  /health
GET  /circuits                    All 24 circuits summary
GET  /circuits/by-deg             Grouped by degradation level
GET  /circuits/{key}              Full detail: turns, DRS zones, ERS harvest turns
GET  /drivers                     25 driver profiles
GET  /compounds/{circuit_key}     Compound data adjusted for circuit
POST /optimize                    Full MC strategy optimization (pulls OpenF1 data)
POST /simulate                    Single strategy lap-by-lap simulation
WS   /ws/live/{circuit}/{driver}  Live race brain — per-lap strategy updates
```

WebSocket input (per lap):
```json
{
  "lap": 35, "total_laps": 57,
  "compound": "SOFT", "tyre_age": 17,
  "gap_ahead_s": 2.1, "gap_behind_s": 5.4,
  "track_temp_c": 38, "rainfall": 0.0,
  "base_lap_time_s": 93.5
}
```

WebSocket output:
```json
{
  "type": "strategy_update",
  "lap": 35,
  "cliff_warning": true,
  "cliff_in_laps": 2,
  "undercut_window": false,
  "weather_note": null,
  "ers_harvest_turns": [4, 8, 11],
  "drs_zones": [...],
  "recommendations": [...]
}
```

---

## Installation

```bash
pip install -r requirements.txt
```

For ML training (optional):
```bash
pip install stable-baselines3[extra] gymnasium
```

---

## Quick Start

### CLI — optimize a race strategy
```bash
# Fetch real 2024 Bahrain data, fit deg model, optimize for Verstappen
python main.py --year 2024 --country Bahrain --driver 1

# Monaco 2-stop strategies for Leclerc
python main.py --year 2024 --country Monaco --driver 16

# Qatar 3-stop strategies (high degradation)
python main.py --year 2024 --country Qatar --driver 1 --three-stop
```

### Server
```bash
uvicorn server:app --reload --port 8000
```

### API examples
```bash
# All circuits
curl http://localhost:8000/circuits

# Monaco detail
curl http://localhost:8000/circuits/monaco

# Optimize Bahrain 2024 for Norris
curl -X POST http://localhost:8000/optimize \
  -H "Content-Type: application/json" \
  -d '{"session_key": 9472, "circuit_key": "bahrain", "driver_number": 4,
       "total_laps": 57, "base_lap_time_s": 93.5, "mc_runs": 200}'

# Single strategy simulation
curl -X POST http://localhost:8000/simulate \
  -H "Content-Type: application/json" \
  -d '{"circuit_key": "bahrain", "driver_number": 1, "total_laps": 57,
       "starting_compound": "SOFT", "pit_stops": [{"lap": 20, "new_compound": "HARD"}],
       "base_lap_time_s": 93.5}'
```

### ML training
```python
from engine.tyre_model import TyreDegradationModel
from engine.lap_time_model import LapTimeModel
from ml.rl_agent import train

deg = TyreDegradationModel()
lap = LapTimeModel(deg)
model = train(deg, lap, circuit_key='bahrain', driver_number=1, total_timesteps=500_000)
```

---

## 2025 Regulation Notes
- **Mandatory 2 dry compounds** in dry races (enforced in simulator + optimizer)
- **Monaco mandatory 2 stops** (hardcoded in circuit DB `recommended_stops=[2]`)
- **Sprint weekends**: Shanghai, Miami, Austin, Interlagos, Baku, Lusail
- **C6 ultra-soft** introduced for street circuits (Monaco: C6/C5/C4)
- Fuel load: ~110 kg at race start, ~1.55 kg/lap burn

---

## Data Sources
- **OpenF1 API** (`api.openf1.org/v1`) — lap times, stints, weather, pit stops, race control
- **Open-Meteo** (`api.open-meteo.com`) — live weather, no API key
- **Pirelli press kits** — compound allocations per circuit
- **F1DataAnalysis** — full-throttle percentages, circuit characteristics

---

## What's Next (Frontend)
The React + D3 frontend dashboard is the next build phase:
- Live track map with DRS zones highlighted and car positions
- Tyre stint panel with real-time degradation curves and cliff warnings
- Race timeline showing all cars' gaps and pit windows
- Strategy advisor panel with natural-language pit call rationale
- Weather overlay with crossover lap indicator
