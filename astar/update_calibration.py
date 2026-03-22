"""Update _CALIBRATION_POINTS in solver.py from per_round_params.json.

Reads tuning results, sorts by settle_rate, and rewrites the calibration
block in solver.py. Creates a backup first.
"""
import json
import os
import re
import shutil
import time

CACHE_DIR = "replay_cache"
SOLVER_PATH = "solver.py"
PARAMS_PATH = os.path.join(CACHE_DIR, "per_round_params.json")

# Rounds to exclude from calibration (known outliers)
EXCLUDE_ROUNDS = {7, 12}

# Params that are fixed and shouldn't be in calibration points
FIXED_PARAMS = {"spawn_max_per_step", "port_wealth_threshold", "wealth_decay", "wealth_coastal_rate"}


def load_results():
    with open(PARAMS_PATH) as f:
        results = json.load(f)
    # Filter out excluded rounds
    results = [r for r in results if r["round"] not in EXCLUDE_ROUNDS]
    # Sort by settle_rate
    results.sort(key=lambda r: r["settle_rate"])
    return results


def format_calibration_block(results):
    """Generate the _CALIBRATION_POINTS Python code block."""
    lines = []
    lines.append("_CALIBRATION_POINTS = [")
    for r in results:
        rn = r["round"]
        rate = r["settle_rate"]
        params = {k: v for k, v in r["params"].items() if k not in FIXED_PARAMS}
        lines.append(f'    {{"rate": {rate:.3f}, "round": {rn}, "params": {{')
        for k, v in sorted(params.items()):
            if isinstance(v, int):
                lines.append(f'        "{k}": {v},')
            else:
                lines.append(f'        "{k}": {v:.3f},')
        lines.append("    }},")
    lines.append("]")
    return "\n".join(lines)


def update_solver(new_block):
    """Replace _CALIBRATION_POINTS block in solver.py."""
    # Backup
    backup = f"{SOLVER_PATH}.bak.{int(time.time())}"
    shutil.copy2(SOLVER_PATH, backup)
    print(f"Backup: {backup}")

    with open(SOLVER_PATH) as f:
        content = f.read()

    # Find and replace the calibration block
    # Pattern: _CALIBRATION_POINTS = [ ... ]  (multiline)
    pattern = r"_CALIBRATION_POINTS\s*=\s*\[.*?\n\]"
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        print("ERROR: Could not find _CALIBRATION_POINTS in solver.py")
        return False

    old_block = match.group()
    old_count = old_block.count('"round":')
    new_count = new_block.count('"round":')
    print(f"Replacing {old_count} calibration points with {new_count}")

    content = content[:match.start()] + new_block + content[match.end():]

    with open(SOLVER_PATH, "w") as f:
        f.write(content)

    # Verify syntax
    try:
        compile(content, SOLVER_PATH, "exec")
        print("Syntax check: OK")
    except SyntaxError as e:
        print(f"SYNTAX ERROR: {e}")
        print("Restoring backup...")
        shutil.copy2(backup, SOLVER_PATH)
        return False

    return True


def main():
    if not os.path.exists(PARAMS_PATH):
        print(f"No results at {PARAMS_PATH}")
        return

    results = load_results()
    print(f"Loaded {len(results)} calibration points")
    for r in results:
        print(f"  R{r['round']:>2}: rate={r['settle_rate']:.3f}, score={r.get('score', 0):.1f}")

    new_block = format_calibration_block(results)
    if update_solver(new_block):
        print("solver.py updated successfully")
    else:
        print("FAILED to update solver.py")


if __name__ == "__main__":
    main()
