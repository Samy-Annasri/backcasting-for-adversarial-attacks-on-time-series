#!/usr/bin/env bash
set -e

# Create Python venv
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install common deps from requirements.txt (if present)
if [ -f requirements.txt ]; then
  echo "Installing requirements.txt"
  # Install everything but avoid forcing an incompatible torch; pip will install torch as listed but user may prefer to manage it explicitly
  pip install -r requirements.txt
else
  echo "requirements.txt not found. Installing minimal deps..."
  pip install numpy pandas matplotlib scikit-learn tqdm openpyxl jupyter nbconvert nbstripout
fi
echo "Environment ready. Activate with: source .venv/bin/activate"
