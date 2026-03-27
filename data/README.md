Summary of the contents in the `backend/data` folder:

### **Data Layer Overview**
The `backend/data` directory handles both static circuit intelligence and dynamic race data fetching.

#### **1. `circuit_db.py` (2025 Circuit Intelligence)**
This is a comprehensive database containing all 24 circuits on the 2025 F1 calendar. It uses structured dataclasses (`Circuit`, `Turn`, `DRSZone`) to model track-specific variables critical for strategy simulations:
*   **DRS Zones:** Exact detection and activation points, straight lengths, and estimated speed gains.
*   **Turn-by-Turn Data:** Each corner is mapped with severity (1–5), lateral load factors, and apex speeds.
*   **Tyre Modeling:** Includes surface abrasiveness, tyre degradation multipliers (calibrated against Bahrain), and Pirelli compound allocations (e.g., C3/C4/C5).
*   **Strategy Factors:** Pit lane time loss, historical Safety Car probability, and rain probability.
*   **ERS Management:** Identifies specific heavy-braking zones for optimal energy harvesting.

#### **2. `openf1_client.py` (API Integration)**
A robust Python client for the **OpenF1 API** that facilitates historical race analysis without requiring authentication.
*   **Core Endpoints:** Fetches `sessions`, `meetings`, `laps`, `stints`, `pit stops`, `weather`, and `race control messages`.
*   **Key Features:**
    *   **Stint Analysis:** Tracks tyre age, compound usage, and lap ranges (backbone of degradation modeling).
    *   **Telemetry Access:** High-frequency car data (speed, RPM, throttle, brake, DRS status).
    *   **Data Structure:** Automatically converts API JSON responses into **Pandas DataFrames** for easy integration with ML models or strategy calculators.
    *   **Rate Limiting:** Built-in delays to ensure respectful usage of the Free OpenF1 tier.
