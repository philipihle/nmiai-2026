#!/bin/bash
# Priority tuning: R18, R7, R12 → calibration → backtest
set -uo pipefail
cd /home/devstar18472/astar-island

echo "=========================================="
echo "Priority Tune Started: $(date)"
echo "=========================================="

# R18 is already running via tune_per_round_3x.py
# Wait for it to finish
echo "[$(date)] Waiting for R18 tune_per_round_3x.py (PID check)..."
while pgrep -f "tune_per_round_3x.py.*18" > /dev/null 2>&1; do
    sleep 30
done
echo "[$(date)] R18 3x tune finished"

# R7 and R12 deep tune
echo "[$(date)] Starting R7/R12 deep tune"
python3 tune_r7_r12_deep.py --mc-deep 500 2>&1 | tee logs/tune_r7_r12_priority.log

# Merge R7/R12 results into per_round_params.json
python3 << 'PYEOF'
import json, os
params_path = "replay_cache/per_round_params.json"
deep_path = "replay_cache/r7_r12_deep_results.json"
if os.path.exists(deep_path):
    with open(params_path) as f:
        params = json.load(f)
    with open(deep_path) as f:
        deep = json.load(f)
    params_map = {r['round']: r for r in params}
    for rn_str, result in deep.items():
        rn = int(rn_str)
        old_score = params_map.get(rn, {}).get('score', 0)
        if result['score'] > old_score:
            # Clean up extra keys
            clean = {
                'round': rn,
                'settle_rate': result['settle_rate'],
                'score': result['score'],
                'params': {k: v for k, v in result['params'].items()
                          if k not in ('__mc_runs',)},
            }
            params_map[rn] = clean
            print(f"R{rn}: {old_score:.1f} → {result['score']:.1f} (IMPROVED)")
        else:
            print(f"R{rn}: keeping {old_score:.1f} (new {result['score']:.1f} not better)")
    with open(params_path, 'w') as f:
        json.dump(list(params_map.values()), f, indent=2)
    print("Updated per_round_params.json")
PYEOF

# Update calibration
echo "[$(date)] Updating calibration"
python3 update_calibration.py 2>&1 | tee logs/update_calib_priority.log

# Run backtest
echo "[$(date)] Running backtest (ensemble mode)"
python3 backtest.py --mode ensemble 2>&1 | tee logs/backtest_priority.log

echo "=========================================="
echo "Priority Tune Complete: $(date)"
echo "=========================================="
