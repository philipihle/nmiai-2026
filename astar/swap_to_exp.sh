#!/bin/bash
# Swap experimental simulator/solver to production
# MANUAL ONLY — review morning_summary.py output first!
set -euo pipefail
cd /home/devstar18472/astar-island

TS=$(date +%s)

echo "Backing up production files..."
cp simulator.py "simulator.py.bak.$TS"
cp solver.py "solver.py.bak.$TS"

echo "Swapping experimental -> production..."
cp simulator_exp.py simulator.py
cp solver_exp.py solver.py

# Verify syntax
python3 -c "import simulator; import solver; print('Syntax OK')"

echo "Done! Production now uses experimental pipeline."
echo "Backups: simulator.py.bak.$TS, solver.py.bak.$TS"
echo "To revert: cp simulator.py.bak.$TS simulator.py && cp solver.py.bak.$TS solver.py"
