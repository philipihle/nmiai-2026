"""Wait for R18 to complete, then cache its data.
Polls the API every 5 minutes. Exits once R18 GT is cached.
"""
import json
import os
import sys
import time

import numpy as np
import requests

TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI3ZjdkNzAxZC0xN2I4LTQxYTgtODhlYi05NzYzYjU0MWQ1NTQiLCJlbWFpbCI6ImVybWVzam9hbmRyZWFzQGdtYWlsLmNvbSIsImlzX2FkbWluIjpmYWxzZSwiZXhwIjoxNzc0NjI1NTY0fQ.A-0e_Ga8GZkUbb4NoYJM2Ng-HyLn_B0GeW9Y05KO5ic"
BASE = "https://api.ainm.no"
HEADERS = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}
CACHE_DIR = "replay_cache"


def cache_round(round_id, rn, n_seeds):
    """Cache replay + analysis for a completed round."""
    os.makedirs(CACHE_DIR, exist_ok=True)

    detail_path = os.path.join(CACHE_DIR, f"round{rn}_detail.json")
    if not os.path.exists(detail_path):
        detail = requests.get(f"{BASE}/astar-island/rounds/{round_id}", headers=HEADERS, timeout=30).json()
        with open(detail_path, "w") as f:
            json.dump(detail, f)
        print(f"R{rn}: detail cached")
        time.sleep(0.5)

    for si in range(n_seeds):
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
                        print(f"R{rn} seed {si}: replay cached")
                        break
                    else:
                        time.sleep(3)
                except Exception as e:
                    print(f"R{rn} seed {si} replay attempt {attempt}: {e}")
                    time.sleep(3)
            time.sleep(0.5)

        analysis_path = os.path.join(CACHE_DIR, f"round{rn}_seed{si}_analysis.json")
        if not os.path.exists(analysis_path):
            try:
                analysis = requests.get(
                    f"{BASE}/astar-island/analysis/{round_id}/{si}",
                    headers=HEADERS, timeout=30,
                ).json()
                with open(analysis_path, "w") as f:
                    json.dump(analysis, f)
                print(f"R{rn} seed {si}: analysis cached")
            except Exception as e:
                print(f"R{rn} seed {si} analysis: {e}")
            time.sleep(0.5)


def get_round_info(round_number):
    """Get round ID and status for a specific round number."""
    rounds = requests.get(f"{BASE}/astar-island/rounds", headers=HEADERS, timeout=30).json()
    for rd in rounds:
        if rd["round_number"] == round_number:
            return rd
    return None


def main():
    target_rounds = [int(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else [18]
    max_wait = 7200  # 2 hours max

    for rn in target_rounds:
        check_path = os.path.join(CACHE_DIR, f"round{rn}_seed0_analysis.json")
        if os.path.exists(check_path):
            print(f"R{rn}: already cached")
            continue

        print(f"Waiting for R{rn} to complete...")
        t0 = time.time()
        while time.time() - t0 < max_wait:
            try:
                rd = get_round_info(rn)
                if rd and rd["status"] == "completed":
                    n_seeds = rd.get("seeds_count", 5)
                    print(f"R{rn} completed! Caching {n_seeds} seeds...")
                    cache_round(rd["id"], rn, n_seeds)
                    # Also cache any other newly completed rounds
                    rounds = requests.get(f"{BASE}/astar-island/rounds", headers=HEADERS, timeout=30).json()
                    for other in rounds:
                        orn = other["round_number"]
                        if other["status"] == "completed" and not os.path.exists(
                            os.path.join(CACHE_DIR, f"round{orn}_seed0_analysis.json")
                        ):
                            print(f"Also caching R{orn}...")
                            cache_round(other["id"], orn, other.get("seeds_count", 5))
                    break
                else:
                    status = rd["status"] if rd else "not found"
                    print(f"  R{rn} status: {status}, waiting 5 min...")
                    time.sleep(300)
            except Exception as e:
                print(f"  Error checking R{rn}: {e}, retrying in 5 min...")
                time.sleep(300)
        else:
            print(f"R{rn}: timeout after {max_wait}s")


if __name__ == "__main__":
    main()
