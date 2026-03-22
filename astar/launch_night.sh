#!/bin/bash
cd ~/astar-island
PROMPT=$(cat night_prompt.txt)
exec claude --dangerously-skip-permissions -p "$PROMPT"
