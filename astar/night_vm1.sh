#!/bin/bash
# VM1 (192 CPU): Production tuning + auto-solve
# Estimated: 5-6 hours
set -uo pipefail
cd /home/devstar18472/astar-island
LOGDIR="logs/night_$(date +%Y%m%d_%H%M)"
mkdir -p "$LOGDIR"
exec > >(tee -a "$LOGDIR/vm1_master.log") 2>&1

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
echo "VM1 Night Run Started: $(date)"
echo "=========================================="

# ---- STEP 1: Wait for R18 GT + cache ----
run_step "Wait for R18 GT" \
    "python3 wait_and_cache_r18.py 18" \
    "wait_r18.log"

# ---- STEP 2: Tune R17 + R18 ----
run_step "Tune R17" \
    "python3 tune_per_round.py --rounds 17" \
    "tune_r17.log"

run_step "Tune R18" \
    "python3 tune_per_round.py --rounds 18" \
    "tune_r18.log"

# ---- STEP 3: Update calibration points ----
run_step "Update calibration" \
    "python3 update_calibration.py" \
    "update_calib.log"

# ---- STEP 4: Deep re-tune R1-R16 (excl R7, R12) ----
for RN in 1 2 3 4 5 6 8 9 10 11 13 14 15 16; do
    run_step "Deep tune R$RN" \
        "python3 tune_per_round.py --rounds $RN" \
        "tune_deep_r${RN}.log"
done

# ---- STEP 4b: Update calibration with deep results ----
run_step "Update calibration (final)" \
    "python3 update_calibration.py" \
    "update_calib_final.log"

# ---- STEP 5: Full backtest ----
run_step "Full backtest" \
    "python3 backtest.py --mode ensemble" \
    "backtest_production.log"

# ---- STEP 6: Sync experimental results from VM3 ----
echo "[$(date)] Syncing results from VM3..."
rsync -avz --timeout=30 \
    devstar18472@34.29.146.13:/home/devstar18472/astar-island/hypothesis_results.json \
    ./ 2>/dev/null || echo "VM3 sync failed (may not be done yet)"

rsync -avz --timeout=30 \
    devstar18472@34.29.146.13:/home/devstar18472/astar-island/logs/ \
    ./logs_vm3/ 2>/dev/null || true

# ---- STEP 7: Morning summary ----
run_step "Morning summary" \
    "python3 morning_summary.py" \
    "morning_summary.log"

echo "=========================================="
echo "VM1 Night Run Complete: $(date)"
echo "=========================================="
