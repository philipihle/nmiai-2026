"""Generate morning summary comparing production vs experimental results.

Reads logs and results from overnight runs.
"""
import glob
import json
import os
import re
import sys
from datetime import datetime

CACHE_DIR = "replay_cache"


def find_latest_logdir():
    dirs = sorted(glob.glob("logs/night_*"))
    return dirs[-1] if dirs else None


def parse_backtest_log(log_path):
    """Extract per-round scores from backtest log."""
    scores = {}
    if not os.path.exists(log_path):
        return scores
    with open(log_path) as f:
        for line in f:
            m = re.search(r'R(\d+)\s.*?score[=:]\s*(\d+\.?\d*)', line, re.IGNORECASE)
            if m:
                scores[int(m.group(1))] = float(m.group(2))
    return scores


def parse_hypothesis_results():
    path = "hypothesis_results.json"
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def parse_r7_r12_results():
    path = os.path.join(CACHE_DIR, "r7_r12_deep_results.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def parse_auto_run_log():
    """Find rounds solved overnight from auto_run or cron logs."""
    solved = []
    for log_path in ["cron_auto.log", "auto_run.log"]:
        if not os.path.exists(log_path):
            continue
        with open(log_path) as f:
            for line in f:
                m = re.search(r'R(\d+) result: completed', line)
                if m:
                    rn = int(m.group(1))
                    if rn not in [s[0] for s in solved]:
                        solved.append((rn, line.strip()))
    return solved


def main():
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    logdir = find_latest_logdir()

    print("=" * 60)
    print(f"MORNING SUMMARY — Astar Island Night Run")
    print(f"{now}")
    print("=" * 60)

    # Production backtest
    prod_scores = {}
    if logdir:
        prod_scores = parse_backtest_log(os.path.join(logdir, "backtest_production.log"))
    if not prod_scores:
        prod_scores = parse_backtest_log("backtest_production.log")

    if prod_scores:
        avg = sum(prod_scores.values()) / len(prod_scores) if prod_scores else 0
        print(f"\nPRODUCTION PIPELINE:")
        print(f"  Backtest avg (R1-R{max(prod_scores.keys())}): {avg:.1f}")
        for rn in sorted(prod_scores):
            print(f"    R{rn}: {prod_scores[rn]:.1f}")
    else:
        print("\nPRODUCTION PIPELINE: No backtest results found")

    # Experimental backtest
    exp_scores = {}
    if logdir:
        exp_scores = parse_backtest_log(os.path.join(logdir, "backtest_exp.log"))
    if not exp_scores:
        exp_scores = parse_backtest_log("backtest_exp.log")

    if exp_scores:
        avg_exp = sum(exp_scores.values()) / len(exp_scores)
        print(f"\nEXPERIMENTAL PIPELINE:")
        print(f"  Backtest avg: {avg_exp:.1f}")
        if prod_scores:
            common = set(prod_scores) & set(exp_scores)
            if common:
                delta = sum(exp_scores[r] - prod_scores[r] for r in common) / len(common)
                print(f"  Delta vs production: {delta:+.1f}")
                improved = [r for r in common if exp_scores[r] > prod_scores[r] + 0.5]
                regressed = [r for r in common if exp_scores[r] < prod_scores[r] - 0.5]
                if improved:
                    print(f"  Improved: {', '.join(f'R{r}({exp_scores[r]-prod_scores[r]:+.1f})' for r in improved)}")
                if regressed:
                    print(f"  Regressed: {', '.join(f'R{r}({exp_scores[r]-prod_scores[r]:+.1f})' for r in regressed)}")
    else:
        print("\nEXPERIMENTAL PIPELINE: No results found")

    # R7/R12 recovery
    r7r12 = parse_r7_r12_results()
    if r7r12:
        print(f"\nR7/R12 RECOVERY:")
        for rn_str, data in r7r12.items():
            rn = int(rn_str)
            old = {7: 74.1, 12: 69.0}.get(rn, 0)
            print(f"  R{rn}: {data['score']:.1f} (was {old:.1f}, delta={data['score']-old:+.1f})")

    # Hypothesis results
    hyp = parse_hypothesis_results()
    if hyp:
        print(f"\nHYPOTHESIS TESTING:")
        for key, result in hyp.items():
            status = "CONFIRMED" if result.get("confirmed") else "not confirmed"
            desc = result.get("description", key)
            print(f"  {key}: {status} — {desc}")
            if result.get("confirmed"):
                for k in ["r", "r_pop", "r_food", "r_allied", "bias", "ks_stat", "r2"]:
                    if k in result:
                        print(f"    {k}={result[k]:.3f}")

    # Rounds solved overnight
    solved = parse_auto_run_log()
    if solved:
        print(f"\nROUNDS SOLVED OVERNIGHT:")
        for rn, line in solved:
            print(f"  R{rn}: {line[-60:]}")

    # Calibration update status
    params_path = os.path.join(CACHE_DIR, "per_round_params.json")
    if os.path.exists(params_path):
        with open(params_path) as f:
            params = json.load(f)
        print(f"\nCALIBRATION POINTS: {len(params)} rounds")
        for p in sorted(params, key=lambda x: x["settle_rate"]):
            print(f"  R{p['round']:>2}: rate={p['settle_rate']:.3f}, score={p.get('score', 0):.1f}")

    # Errors
    if logdir:
        error_file = os.path.join(logdir, "errors.txt")
        if os.path.exists(error_file):
            with open(error_file) as f:
                errors = f.read().strip()
            if errors:
                print(f"\nERRORS:")
                print(f"  {errors}")

    # Recommendation
    print(f"\n{'='*60}")
    print("RECOMMENDATION:")
    if exp_scores and prod_scores:
        common = set(prod_scores) & set(exp_scores)
        if common:
            delta = sum(exp_scores[r] - prod_scores[r] for r in common) / len(common)
            if delta > 1.0:
                print(f"  SWAP to experimental (+{delta:.1f} avg)")
                print(f"  Run: bash swap_to_exp.sh")
            elif delta > 0:
                print(f"  Marginal improvement (+{delta:.1f}). Review per-round before swapping.")
            else:
                print(f"  Keep production (experimental is {delta:.1f} worse)")
    else:
        print("  Insufficient data for comparison")
    print("=" * 60)


if __name__ == "__main__":
    main()
