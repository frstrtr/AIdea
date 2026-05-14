#!/usr/bin/env bash
# Idempotent installer for AIdea on a fresh Debian 12 / Ubuntu 24.04 VM.
# Run as a user with sudo. Repeatable: rerun to update.
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/frstrtr/AIdea.git}"
INSTALL_DIR="${INSTALL_DIR:-/opt/aidea}"
SERVICE_USER="${SERVICE_USER:-aidea}"

log() { printf '\033[1;36m[install]\033[0m %s\n' "$*"; }

log "1/7 system packages"
sudo apt-get update -qq
sudo apt-get install -y --no-install-recommends \
    python3 python3-venv python3-full git curl ca-certificates

log "2/7 Node.js + agent CLI (claude)"
if ! command -v claude >/dev/null 2>&1; then
    curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
    sudo apt-get install -y nodejs
    sudo npm install -g @anthropic-ai/claude-code
fi

log "3/7 service user: $SERVICE_USER"
if ! id "$SERVICE_USER" >/dev/null 2>&1; then
    sudo useradd --system --create-home \
        --home-dir "/home/$SERVICE_USER" --shell /bin/bash "$SERVICE_USER"
fi

log "4/7 clone / update repo at $INSTALL_DIR"
if [ ! -d "$INSTALL_DIR/.git" ]; then
    sudo mkdir -p "$INSTALL_DIR"
    sudo chown "$SERVICE_USER:$SERVICE_USER" "$INSTALL_DIR"
    sudo -u "$SERVICE_USER" git clone "$REPO_URL" "$INSTALL_DIR"
else
    sudo -u "$SERVICE_USER" git -C "$INSTALL_DIR" pull --ff-only
fi

log "5/7 Python venv + requirements"
sudo -u "$SERVICE_USER" python3 -m venv "$INSTALL_DIR/.venv"
sudo -u "$SERVICE_USER" "$INSTALL_DIR/.venv/bin/pip" install -q -U pip
sudo -u "$SERVICE_USER" "$INSTALL_DIR/.venv/bin/pip" install -q -r "$INSTALL_DIR/requirements.txt"

log "6/7 .env scaffold"
if [ ! -f "$INSTALL_DIR/.env" ]; then
    sudo -u "$SERVICE_USER" cp "$INSTALL_DIR/.env.example" "$INSTALL_DIR/.env"
    sudo chmod 600 "$INSTALL_DIR/.env"
    sudo chown "$SERVICE_USER:$SERVICE_USER" "$INSTALL_DIR/.env"
    echo
    echo "    >> Edit $INSTALL_DIR/.env now: set TELEGRAM_BOT_TOKEN, "
    echo "       AIDEA_HOST (use 0.0.0.0 to expose the web app), etc."
    echo
fi

log "7/7 systemd units"
sudo install -m 0644 "$INSTALL_DIR/deploy/aidea-web.service" /etc/systemd/system/aidea-web.service
sudo install -m 0644 "$INSTALL_DIR/deploy/aidea-bot.service" /etc/systemd/system/aidea-bot.service
sudo systemctl daemon-reload

cat <<EOF

Install complete.

NEXT STEPS (manual, one-time):
  1) Log the agent CLI into your account on this VM:
       sudo -iu $SERVICE_USER
       claude            # follow the prompts to log in via the web flow
       exit
     The bot and web app inherit auth from this user's CLI session.

  2) Edit secrets:
       sudo -u $SERVICE_USER nano $INSTALL_DIR/.env
     (set TELEGRAM_BOT_TOKEN at minimum)

  3) Enable + start services:
       sudo systemctl enable --now aidea-web
       sudo systemctl enable --now aidea-bot

  4) Check status:
       sudo systemctl status aidea-web aidea-bot --no-pager
       sudo journalctl -u aidea-web -f
       sudo journalctl -u aidea-bot -f

  5) If you set AIDEA_HOST=0.0.0.0, the web UI is reachable on
     http://<vm-ip>:\${AIDEA_PORT:-8000}/  — front it with a reverse proxy
     (nginx / caddy) before exposing to the public internet.
EOF
