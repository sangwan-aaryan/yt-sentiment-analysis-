#!/usr/bin/env bash
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive

sudo apt-get update -y
sudo apt-get install --yes --no-install-recommends awscli curl docker.io
sudo systemctl enable --now docker