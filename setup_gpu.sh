#!/usr/bin/env bash
# One-shot GPU Docker setup for AnomalyNetFlow.
# Run as: sudo bash setup_gpu.sh
set -euo pipefail

echo "=== [1/4] Installing nvidia-container-toolkit ==="
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  > /etc/apt/sources.list.d/nvidia-container-toolkit.list

apt-get update -qq
apt-get install -y nvidia-container-toolkit

echo "=== [2/4] Configuring Docker nvidia runtime ==="
nvidia-ctk runtime configure --runtime=docker

echo "=== [3/4] Restarting Docker ==="
systemctl restart docker

echo "=== [4/4] Verifying nvidia runtime in Docker ==="
docker info | grep -i "runtime\|nvidia" || true

echo ""
echo "GPU setup complete. Signal Claude Code to rebuild."
touch /tmp/gpu_setup_done
