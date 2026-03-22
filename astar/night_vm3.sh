#!/bin/bash
# VM3 (112 CPU): Experimental pipeline — NEVER touches production files
# Estimated: 6-7 hours
set -uo pipefail
cd /home/devstar18472/astar-island
LOGDIR="logs/night_$(date +%Y%m%d_%H%M)"
mkdir -p "$LOGDIR"
exec > >(tee -a "$LOGDIR/vm3_master.log") 2>&1

run_step() {
    local name="$1"
    local cmd="$2"
    local logfile="$3"
    echo "[$(date)] Starting: $name"
    if eval "$cmd" 2>&1 | tee "$LOGDIR/$logfile"; then
        echo "[$(date)] Completed: $name"
    else
        echo "[$(date)] FAILED: $name" | tee -a "$LOGDIR/errors.txt"
    fi
}

echo "=========================================="
echo "VM3 Night Run Started: $(date)"
echo "=========================================="

# ---- STEP 1: Create experimental copies ----
echo "[$(date)] Creating experimental copies..."
cp simulator.py simulator_exp.py
cp solver.py solver_exp.py
cp backtest.py backtest_exp.py
# Patch imports in copies
sed -i 's/from simulator import/from simulator_exp import/g' solver_exp.py
sed -i 's/from solver import/from solver_exp import/g' backtest_exp.py

# Create tune_per_round_exp.py
cp tune_per_round.py tune_per_round_exp.py
sed -i 's/from simulator import/from simulator_exp import/g' tune_per_round_exp.py

echo "[$(date)] Experimental copies created"

# ---- STEP 2: Hypothesis testing ----
run_step "Hypothesis testing" \
    "python3 test_hypotheses.py" \
    "hypotheses.log"

# ---- STEP 3: Implement confirmed hypotheses ----
run_step "Implement hypotheses" \
    "python3 implement_hypotheses.py" \
    "implement.log"

# ---- STEP 4: Tune experimental simulator ----
# Tune a representative set of rounds with experimental simulator
for RN in 1 2 3 4 5 6 8 9 10 11; do
    run_step "Exp tune R$RN" \
        "python3 tune_per_round_exp.py --rounds $RN" \
        "tune_exp_r${RN}.log"
done

# ---- STEP 5: R7 + R12 deep search ----
run_step "R7+R12 deep search" \
    "python3 tune_r7_r12_deep.py --rounds 7 12 --mc-deep 500" \
    "tune_r7_r12.log"

# ---- STEP 6: Experimental backtest ----
# Update experimental solver calibration
run_step "Update exp calibration" \
    "python3 -c \"
import json, os
# Use tune_per_round_exp results if available
exp_params = os.path.join('replay_cache', 'per_round_params.json')
if os.path.exists(exp_params):
    print('Exp calibration from:', exp_params)
else:
    print('No experimental params found')
\"" \
    "exp_calib.log"

run_step "Experimental backtest" \
    "python3 backtest_exp.py --mode ensemble" \
    "backtest_exp.log"

echo "=========================================="
echo "VM3 Night Run Complete: $(date)"
echo "=========================================="
