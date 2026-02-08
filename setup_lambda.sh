#!/bin/bash
# Lambda Labs One-Line Setup
# Usage: curl -s https://raw.githubusercontent.com/YOUR_REPO/main/setup_lambda.sh | bash
# Or: bash setup_lambda.sh [POD_NUMBER] [TOTAL_PODS] [SCENARIO] [TRIALS]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Lambda Labs Quick Setup${NC}"
echo -e "${GREEN}========================================${NC}"

# Parse args or use defaults
POD_NUM=${1:-1}
TOTAL_PODS=${2:-1}
SCENARIO=${3:-ultimatum_bluff}
TRIALS=${4:-25}

# ============================================================
# STEP 1: System packages (fast, already cached on Lambda)
# ============================================================
echo -e "\n${YELLOW}[1/5] Installing system packages...${NC}"
sudo apt-get update -qq && sudo apt-get install -y -qq python3.11-dev git > /dev/null 2>&1
echo -e "${GREEN}✓ System packages ready${NC}"

# ============================================================
# STEP 2: Clone/update repo
# ============================================================
echo -e "\n${YELLOW}[2/5] Setting up repository...${NC}"
cd ~
if [ -d "multiagent-lab" ]; then
    cd multiagent-lab && git pull --quiet
    echo -e "${GREEN}✓ Repository updated${NC}"
else
    git clone --quiet https://github.com/asimsmadeit/multiagent-lab.git
    cd multiagent-lab
    echo -e "${GREEN}✓ Repository cloned${NC}"
fi

# ============================================================
# STEP 3: Python environment (use system python, skip venv for speed)
# ============================================================
echo -e "\n${YELLOW}[3/5] Installing Python packages...${NC}"
pip install --quiet --upgrade pip

# Install PyTorch for CUDA 12.8 (GH200/B200)
pip install --quiet torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install requirements
pip install --quiet -r requirements.txt
pip install --quiet -e .

echo -e "${GREEN}✓ Python packages installed${NC}"

# ============================================================
# STEP 4: Check for API keys
# ============================================================
echo -e "\n${YELLOW}[4/5] Checking API keys...${NC}"

if [ -z "$OPENAI_API_KEY" ]; then
    echo -e "${RED}⚠ OPENAI_API_KEY not set!${NC}"
    echo "  Run: export OPENAI_API_KEY='sk-your-key'"
else
    echo -e "${GREEN}✓ OPENAI_API_KEY is set${NC}"
fi

if [ -z "$HF_TOKEN" ]; then
    echo -e "${RED}⚠ HF_TOKEN not set (needed for Gemma model)${NC}"
    echo "  Run: export HF_TOKEN='hf_your_token'"
    echo "  Get token: https://huggingface.co/settings/tokens"
else
    echo -e "${GREEN}✓ HF_TOKEN is set${NC}"
    huggingface-cli login --token $HF_TOKEN --add-to-git-credential 2>/dev/null || true
fi

# ============================================================
# STEP 5: Verify GPU
# ============================================================
echo -e "\n${YELLOW}[5/5] Verifying GPU...${NC}"
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# ============================================================
# READY
# ============================================================
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}  SETUP COMPLETE!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "Configuration:"
echo -e "  Pod:      ${POD_NUM}/${TOTAL_PODS}"
echo -e "  Scenario: ${SCENARIO}"
echo -e "  Trials:   ${TRIALS}"
echo ""
echo -e "${YELLOW}To run experiment:${NC}"
echo ""
if [ "$TOTAL_PODS" -gt 1 ]; then
    echo "  ./run_exp.sh ${POD_NUM} ${TOTAL_PODS} ${SCENARIO} ${TRIALS}"
else
    echo "  ./run_exp.sh"
fi
echo ""
echo -e "${YELLOW}Or manually:${NC}"
echo ""
echo "  cd ~/multiagent-lab"
echo "  export PYTHONPATH=\$PWD"
echo "  python interpretability/run_deception_experiment.py \\"
echo "      --model google/gemma-7b-it \\"
echo "      --scenario-name ${SCENARIO} \\"
echo "      --trials ${TRIALS} \\"
echo "      --parallel-pod ${POD_NUM}/${TOTAL_PODS}"
