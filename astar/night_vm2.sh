#!/bin/bash
# VM2 (16 CPU): ML retraining + prior learning
# Estimated: 3 hours
set -uo pipefail
cd /home/devstar18472/astar-island
LOGDIR="logs/night_$(date +%Y%m%d_%H%M)"
mkdir -p "$LOGDIR"
exec > >(tee -a "$LOGDIR/vm2_master.log") 2>&1

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
echo "VM2 Night Run Started: $(date)"
echo "=========================================="

# ---- STEP 1: Re-learn priors from R1-R17 ----
run_step "Learn priors (R1-R17)" \
    "python3 learn_priors.py" \
    "learn_priors_v1.log"

# ---- STEP 2: Retrain ML with R1-R17 ----
run_step "Train ML (R1-R17)" \
    "python3 train_model.py" \
    "train_model_v1.log"

# Backup
cp ml_models.pkl ml_models_v2.pkl 2>/dev/null || true

# ---- STEP 3: Wait for R18 data ----
echo "[$(date)] Waiting for R18 data from VM1..."
MAX_WAIT=7200
WAITED=0
while [ ! -f replay_cache/round18_seed0_analysis.json ]; do
    if [ $WAITED -ge $MAX_WAIT ]; then
        echo "[$(date)] Timeout waiting for R18 data"
        break
    fi
    # Try to sync from VM1
    rsync -avz --timeout=30 \
        devstar18472@34.29.189.131:/home/devstar18472/astar-island/replay_cache/round18* \
        replay_cache/ 2>/dev/null || true
    if [ -f replay_cache/round18_seed0_analysis.json ]; then
        echo "[$(date)] R18 data received"
        break
    fi
    echo "[$(date)] R18 data not available yet, waiting 10 min... ($WAITED/$MAX_WAIT)"
    sleep 600
    WAITED=$((WAITED + 600))
done

# ---- STEP 4: Re-learn priors with R18 ----
if [ -f replay_cache/round18_seed0_analysis.json ]; then
    run_step "Learn priors (R1-R18)" \
        "python3 learn_priors.py" \
        "learn_priors_v2.log"

    # ---- STEP 5: Retrain ML with R18 ----
    run_step "Train ML (R1-R18)" \
        "python3 train_model.py" \
        "train_model_v2.log"

    cp ml_models.pkl ml_models_v3.pkl 2>/dev/null || true
fi

# ---- STEP 6: Sync ML models to VM1 ----
echo "[$(date)] Syncing ML models to VM1..."
rsync -avz --timeout=30 \
    ml_models.pkl ml_models_v3.pkl \
    devstar18472@34.29.189.131:/home/devstar18472/astar-island/ \
    2>/dev/null || echo "Sync to VM1 failed"

echo "=========================================="
echo "VM2 Night Run Complete: $(date)"
echo "=========================================="
