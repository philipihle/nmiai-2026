"""Auto-run: cache completed rounds + submit on active rounds.

Designed to run hourly via cron. Logs to auto_run.log.
"""
import json
import logging
import os
import sys
import time

import numpy as np
import requests

TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI3ZjdkNzAxZC0xN2I4LTQxYTgtODhlYi05NzYzYjU0MWQ1NTQiLCJlbWFpbCI6ImVybWVzam9hbmRyZWFzQGdtYWlsLmNvbSIsImlzX2FkbWluIjpmYWxzZSwiZXhwIjoxNzc0NjI1NTY0fQ.A-0e_Ga8GZkUbb4NoYJM2Ng-HyLn_B0GeW9Y05KO5ic"
BASE = "https://api.ainm.no"
HEADERS = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}
CACHE_DIR = "replay_cache"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler("auto_run.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("auto_run")


def get_rounds():
    return requests.get(f"{BASE}/astar-island/rounds", headers=HEADERS, timeout=30).json()


def cache_round(round_id, rn, n_seeds):
    """Cache replay + analysis for a completed round. Returns True if new data cached."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    new_data = False

    # Detail
    detail_path = os.path.join(CACHE_DIR, f"round{rn}_detail.json")
    if not os.path.exists(detail_path):
        detail = requests.get(f"{BASE}/astar-island/rounds/{round_id}", headers=HEADERS, timeout=30).json()
        with open(detail_path, "w") as f:
            json.dump(detail, f)
        log.info("R%d: detail cached", rn)
        new_data = True
        time.sleep(0.5)

    for si in range(n_seeds):
        # Replay
        replay_path = os.path.join(CACHE_DIR, f"round{rn}_seed{si}_replay.json")
        if not os.path.exists(replay_path):
            for attempt in range(3):
                try:
                    resp = requests.post(
                        f"{BASE}/astar-island/replay",
                        json={"round_id": round_id, "seed_index": si},
                        headers=HEADERS, timeout=60,
                    ).json()
                    if "frames" in resp:
                        with open(replay_path, "w") as f:
                            json.dump(resp, f)
                        log.info("R%d seed %d: replay cached", rn, si)
                        new_data = True
                        break
                    else:
                        time.sleep(3)
                except Exception as e:
                    log.warning("R%d seed %d replay attempt %d: %s", rn, si, attempt, e)
                    time.sleep(3)
            time.sleep(0.5)

        # Analysis
        analysis_path = os.path.join(CACHE_DIR, f"round{rn}_seed{si}_analysis.json")
        if not os.path.exists(analysis_path):
            try:
                analysis = requests.get(
                    f"{BASE}/astar-island/analysis/{round_id}/{si}",
                    headers=HEADERS, timeout=30,
                ).json()
                with open(analysis_path, "w") as f:
                    json.dump(analysis, f)
                log.info("R%d seed %d: analysis cached", rn, si)
                new_data = True
            except Exception as e:
                log.warning("R%d seed %d analysis: %s", rn, si, e)
            time.sleep(0.5)

    return new_data


def solve_active(round_id, rn):
    """Run solver on an active round and submit."""
    from solver import AstarSolver

    log.info("Solving R%d (%s)...", rn, round_id[:8])
    solver = AstarSolver(TOKEN, use_mc=True)

    try:
        result = solver.solve_round(round_id)
        log.info("R%d result: %s", rn, result.get("status", "unknown"))
        log.info("R%d round_type=%s, queries=%s",
                 rn, result.get("round_type", "?"), result.get("queries_executed", "?"))
        write_report(rn, round_id, result, solver)
        return result
    except Exception as e:
        log.error("R%d solve failed: %s", rn, e)
        return {"error": str(e)}


def write_report(rn, round_id, result, solver):
    """Write a per-round report after solving."""
    import datetime
    report_path = f"reports/round{rn}_report.md"
    os.makedirs("reports", exist_ok=True)

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    round_type = result.get("round_type", "unknown")
    status = result.get("status", "unknown")
    queries = result.get("queries_executed", "?")
    scout_q = result.get("scout_queries", "?")
    deep_q = result.get("deep_queries", "?")
    seeds_results = result.get("results", [])

    lines = [
        f"# Round {rn} — Auto-solve Report",
        f"",
        f"- **Dato:** {now}",
        f"- **Round ID:** `{round_id}`",
        f"- **Status:** {status}",
        f"- **Round type:** {round_type}",
        f"- **Queries:** {queries} (scout={scout_q}, deep={deep_q})",
        f"- **Model:** ensemble (MC + prior + ML)",
        f"",
        f"## Seed Results",
        f"",
    ]

    for sr in seeds_results:
        si = sr.get("seed", "?")
        s_status = sr.get("status", sr.get("error", "?"))
        lines.append(f"- Seed {si}: {s_status}")

    # Try to get scout observations info
    obs_path = os.path.join("obs_cache", f"{round_id}.json")
    if os.path.exists(obs_path):
        with open(obs_path) as f:
            obs = json.load(f)
        total_cells = 0
        total_settle = 0
        total_ruin = 0
        for o in obs:
            grid = o["grid"]
            for row in grid:
                for v in row:
                    total_cells += 1
                    if v in (1, 2):
                        total_settle += 1
                    if v == 3:
                        total_ruin += 1
        if total_cells > 0:
            settle_rate = total_settle / total_cells
            ruin_rate = total_ruin / total_cells
            lines.extend([
                f"",
                f"## Observations Summary",
                f"",
                f"- Total observed cells: {total_cells}",
                f"- Settlement rate: {settle_rate:.4f} ({total_settle} cells)",
                f"- Ruin rate: {ruin_rate:.4f} ({total_ruin} cells)",
                f"- Detection: {'dieoff' if settle_rate < 0.03 else 'normal'} (threshold: 0.03)",
            ])

    lines.extend([
        f"",
        f"## Notes",
        f"",
        f"Auto-solved by cron job. Ensemble model with tuned SimParams.",
        f"Normal: spawn=0.15, death=0.04 | Dieoff: spawn=0.15, death=0.15",
        f"",
    ])

    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    log.info("R%d: report written to %s", rn, report_path)


def main():
    log.info("=" * 50)
    log.info("Auto-run started")

    try:
        rounds = get_rounds()
    except Exception as e:
        log.error("Failed to fetch rounds: %s", e)
        return

    # 1. Cache all completed rounds
    completed = [r for r in rounds if r["status"] == "completed"]
    new_cached = 0
    for r in completed:
        rn = r["round_number"]
        n_seeds = r.get("seeds_count", 5)
        if cache_round(r["id"], rn, n_seeds):
            new_cached += 1

    if new_cached:
        log.info("Cached %d new rounds", new_cached)
    else:
        log.info("No new rounds to cache")

    # 2. Solve active rounds
    active = [r for r in rounds if r["status"] == "active"]
    if not active:
        log.info("No active rounds")
    else:
        for r in active:
            rn = r["round_number"]
            log.info("Active round: R%d (%s)", rn, r["id"][:8])
            solve_active(r["id"], rn)

    log.info("Auto-run done")
    log.info("=" * 50)


if __name__ == "__main__":
    main()
