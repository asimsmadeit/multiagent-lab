#!/bin/bash
# Quick experiment runner
# Usage: ./run_exp.sh [POD_NUM] [TOTAL_PODS] [SCENARIO] [TRIALS]

POD_NUM=${1:-1}
TOTAL_PODS=${2:-1}
SCENARIO=${3:-ultimatum_bluff}
TRIALS=${4:-25}

cd ~/multiagent-lab
export PYTHONPATH=$PWD

echo "========================================"
echo "  Running Experiment"
echo "========================================"
echo "  Pod:      $POD_NUM/$TOTAL_PODS"
echo "  Scenario: $SCENARIO"
echo "  Trials:   $TRIALS"
echo "========================================"

# Workaround for flash_attn issues on some GPUs
python3 -c "
import sys
sys.modules['flash_attn'] = None

import runpy
sys.argv = [
    'run',
    '--model', 'google/gemma-7b-it',
    '--scenario-name', '$SCENARIO',
    '--trials', '$TRIALS',
    '--max-rounds', '5',
    '--device', 'cuda',
    '--dtype', 'bfloat16',
    '--parallel-pod', '$POD_NUM/$TOTAL_PODS'
]
runpy.run_path('interpretability/run_deception_experiment.py', run_name='__main__')
"

echo ""
echo "========================================"
echo "  Experiment Complete!"
echo "========================================"
echo "  Results in: ~/multiagent-lab/experiment_output/"
ls -la ~/multiagent-lab/experiment_output/*.pt 2>/dev/null || echo "  (no .pt files yet)"
