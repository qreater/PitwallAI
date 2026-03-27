"""
OpenF1 API client — historical data (no auth required).
Base URL: https://api.openf1.org/v1
Free endpoints: sessions, meetings, laps, stints, pit, weather,
                intervals, race_control, drivers, car_data
"""

import requests
import pandas as pd
import time
import logging
from typing import Optional
from functools import lru_cache

log = logging.getLogger(__name__)

BASE_URL = "https://api.openf1.org/v1"
RATE_LIMIT_DELAY = 0.3  # seconds between requests (be a good citizen)


class OpenF1Client:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})

    def _get(self, endpoint: str, params: dict = None) -> list[dict]:
        url = f"{BASE_URL}/{endpoint}"
        try:
            time.sleep(RATE_LIMIT_DELAY)
            resp = self.session.get(url, params=params, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            log.error(f"OpenF1 fetch failed [{endpoint}]: {e}")
            return []

    def _to_df(self, endpoint: str, params: dict = None) -> pd.DataFrame:
        data = self._get(endpoint, params)
        return pd.DataFrame(data) if data else pd.DataFrame()

    # ── Session discovery ──────────────────────────────────────────────────

    def get_sessions(self, year: int, session_type: str = "Race") -> pd.DataFrame:
        """All race sessions for a given year."""
        return self._to_df("sessions", {"year": year, "session_type": session_type})

    def get_session_key(self, year: int, country: str, session_type: str = "Race") -> Optional[int]:
        """Resolve a human-readable race to its session_key integer."""
        df = self._to_df("sessions", {
            "year": year,
            "country_name": country,
            "session_type": session_type,
        })
        if df.empty:
            log.warning(f"No session found: {year} {country} {session_type}")
            return None
        return int(df.iloc[0]["session_key"])

    def get_drivers(self, session_key: int) -> pd.DataFrame:
        """Driver list for a session (number, code, name, team)."""
        return self._to_df("drivers", {"session_key": session_key})

    # ── Core strategy data ─────────────────────────────────────────────────

    def get_stints(self, session_key: int, driver_number: Optional[int] = None) -> pd.DataFrame:
        """
        Stint data: compound, lap_start, lap_end, tyre_age_at_start.
        This is the backbone of tyre degradation analysis.
        """
        params = {"session_key": session_key}
        if driver_number:
            params["driver_number"] = driver_number
        df = self._to_df("stints", params)
        if not df.empty:
            df["stint_length"] = df["lap_end"] - df["lap_start"] + 1
        return df

    def get_laps(self, session_key: int, driver_number: Optional[int] = None) -> pd.DataFrame:
        """
        Per-lap timing: lap_duration, sector times, is_pit_out_lap,
        segments (mini-sector colours), speed trap.
        """
        params = {"session_key": session_key}
        if driver_number:
            params["driver_number"] = driver_number
        df = self._to_df("laps", params)
        if not df.empty:
            # Convert lap_duration from seconds (float) — already numeric in OpenF1
            df["lap_duration"] = pd.to_numeric(df.get("lap_duration", pd.Series()), errors="coerce")
            # Flag in/out laps for exclusion from degradation fitting
            if "is_pit_out_lap" in df.columns:
                df["is_pit_out_lap"] = df["is_pit_out_lap"].fillna(False).astype(bool)
        return df

    def get_pit_stops(self, session_key: int) -> pd.DataFrame:
        """
        Pit stop data: driver_number, lap_number, stop_duration, lane_duration.
        stop_duration = stationary time only (wheelgun etc.)
        lane_duration = full pit lane traversal
        """
        return self._to_df("pit", {"session_key": session_key})

    def get_intervals(self, session_key: int) -> pd.DataFrame:
        """
        Gap to leader + interval to car ahead, updated ~every 4s during race.
        Useful for undercut/overcut gap modelling.
        """
        return self._to_df("intervals", {"session_key": session_key})

    def get_weather(self, session_key: int) -> pd.DataFrame:
        """
        Track temp, air temp, humidity, wind speed, rainfall flag — per minute.
        """
        return self._to_df("weather", {"session_key": session_key})

    def get_race_control(self, session_key: int) -> pd.DataFrame:
        """
        Race control messages: Safety Car, VSC, flags, incidents.
        flag field: GREEN/YELLOW/RED/CHEQUERED, message for SC/VSC text.
        """
        df = self._to_df("race_control", {"session_key": session_key})
        if not df.empty and "message" in df.columns:
            df["is_sc"] = df["message"].str.contains("SAFETY CAR", na=False, case=False)
            df["is_vsc"] = df["message"].str.contains("VIRTUAL", na=False, case=False)
        return df

    def get_car_data(self, session_key: int, driver_number: int) -> pd.DataFrame:
        """
        High-frequency telemetry (~3.7 Hz): speed, RPM, gear, throttle, brake, DRS.
        DRS: 0/1=off, 8=eligible, 10/12/14=active.
        Note: large payload — use sparingly, filter by lap if possible.
        """
        return self._to_df("car_data", {
            "session_key": session_key,
            "driver_number": driver_number,
        })

    # ── Convenience bundled fetch ──────────────────────────────────────────

    def get_full_race_data(self, session_key: int) -> dict[str, pd.DataFrame]:
        """
        Fetch all strategy-relevant data for one race session.
        Returns dict of DataFrames keyed by endpoint name.
        """
        log.info(f"Fetching full race data for session_key={session_key}")
        return {
            "drivers":      self.get_drivers(session_key),
            "stints":       self.get_stints(session_key),
            "laps":         self.get_laps(session_key),
            "pit_stops":    self.get_pit_stops(session_key),
            "intervals":    self.get_intervals(session_key),
            "weather":      self.get_weather(session_key),
            "race_control": self.get_race_control(session_key),
        }
