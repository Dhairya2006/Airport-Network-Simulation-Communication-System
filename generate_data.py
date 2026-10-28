"""Generate transparent, reproducible SYNTHETIC network sessions.

Status is the hidden scenario sampled BEFORE the observations. Distributions
intentionally overlap, so the classifier is not just memorizing a threshold.
No rows here are measurements from an actual airport or Packet Tracer.
"""
import argparse
import numpy as np
import pandas as pd
from settings import DATA_PATH, DEPARTMENTS, LABELS


def make_dataset(sessions=600, observations_per_session=12, seed=42):
    if sessions < 30 or observations_per_session < 1:
        raise ValueError("Use at least 30 sessions and at least 1 observation per session")
    rng = np.random.default_rng(seed)
    department_effect = {
        "Check-In": (2.0, 6.0), "Security": (-2.0, 10.0),
        "Operations": (8.0, 12.0), "Baggage": (4.0, 7.0),
        "Servers": (11.0, -2.0), "Wireless": (3.0, 14.0),
        "IoT": (-12.0, -4.0),
    }
    # The means + standard deviations overlap by design.
    params = {
        "Normal": (37, 16, 0.7, 0.65, 44, 27, 8, 7),
        "Congested": (72, 15, 3.0, 2.1, 125, 64, 32, 22),
        "Critical": (88, 13, 8.3, 4.9, 255, 115, 76, 50),
    }
    rows = []
    for session in range(sessions):
        dept = str(rng.choice(DEPARTMENTS))
        status = str(rng.choice(LABELS, p=[0.57, 0.29, 0.14]))
        u_mean, u_std, loss_mean, loss_std, delay_mean, delay_std, jit_mean, jit_std = params[status]
        dept_util, dept_delay = department_effect[dept]
        # One hidden environmental offset per session introduces correlation.
        load_shift = float(rng.normal(0, 7))
        delay_shift = float(rng.normal(0, 18))
        devices = int(rng.integers(4, 30 if dept == "IoT" else 55))
        event_smoke = int(dept == "IoT" and rng.random() < 0.04)
        event_motion = int(dept in ("IoT", "Security") and rng.random() < 0.12)
        for step in range(observations_per_session):
            load_noise = rng.normal(0, u_std)
            utilization = np.clip(u_mean + dept_util + load_shift + load_noise, 0, 100)
            loss = np.clip(loss_mean + rng.normal(0, loss_std) + max(0, utilization - 75) * 0.035, 0, 100)
            delay = np.clip(delay_mean + dept_delay + delay_shift + rng.normal(0, delay_std) + loss * 3.5, 1, 1500)
            jitter = np.clip(jit_mean + rng.normal(0, jit_std) + loss * 1.5, 0, 600)
            # A fixed five-minute observation window is assumed.
            packet_count = max(0, round((utilization + 5) * devices * rng.uniform(3, 12)))
            rows.append({
                "session_id": f"S{session:04d}",
                "timestamp": (pd.Timestamp("2026-10-01 08:00:00") + pd.Timedelta(minutes=5 * (session * observations_per_session + step))).isoformat(),
                "department": dept,
                "packet_count": int(packet_count),
                "utilization_pct": round(float(utilization), 2),
                "packet_loss_pct": round(float(loss), 2),
                "delay_ms": round(float(delay), 2),
                "jitter_ms": round(float(jitter), 2),
                "connected_devices": devices,
                "smoke_alert": event_smoke,
                "motion_alert": event_motion,
                "status": status,
            })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic airport traffic sessions")
    parser.add_argument("--sessions", type=int, default=600)
    parser.add_argument("--observations", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    df = make_dataset(args.sessions, args.observations, args.seed)
    df.to_csv(DATA_PATH, index=False)
    print(f"Saved {len(df):,} SYNTHETIC rows from {args.sessions} sessions to {DATA_PATH}")
    print(df.status.value_counts().to_string())


if __name__ == "__main__":
    main()
