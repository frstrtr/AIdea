# AIdea — deployment

This directory has everything needed to run AIdea on a Linux VM
(Debian 12 or Ubuntu 22.04+). It assumes you have:

- An existing Proxmox VE node (or any cloud-init-capable host).
- A public SSH key to put in the VM.
- A Telegram bot token from @BotFather.
- An interactive session to log the agent CLI into your account
  (one-time, can't be scripted).

## What you get

Two systemd-managed services on one VM:

- `aidea-web.service` — FastAPI + inline UI on `:${AIDEA_PORT:-8000}`.
- `aidea-bot.service` — Telegram long-polling bot.

Both share a single working directory (`/opt/aidea`), one `.env` file,
and one `usage.jsonl` log.

## Files

| File | Role |
|---|---|
| `install.sh` | Idempotent installer to run on the VM (`bash install.sh`). |
| `aidea-web.service` | systemd unit for the FastAPI app. |
| `aidea-bot.service` | systemd unit for the Telegram bot. |
| `cloud-init.yaml` | Optional cloud-init user-data that runs `install.sh` automatically on first boot. |

## Path A — Proxmox via cloud-init (fully automated install)

1. **Create a VM** on PVE using a cloud image template (e.g.
   `debian-12-genericcloud-amd64.qcow2` imported into your storage as
   a template). 2 vCPU / 2 GB RAM / 20 GB disk is enough.

2. **Attach cloud-init**. In the PVE UI: VM → Hardware → Add → CloudInit
   Drive. Then VM → Cloud-Init → set User, IP config, and paste the
   contents of `cloud-init.yaml` into the "User Data" field (after
   replacing the SSH public-key placeholder with your real key).

3. **Boot the VM.** Cloud-init runs the installer on first boot. After
   it finishes the install is complete; what remains is the two manual
   steps below.

4. **SSH in and log the agent CLI in:**
   ```bash
   ssh ops@<vm-ip>
   sudo -iu aidea
   claude                    # follow the web-flow prompts
   exit
   ```

5. **Set the bot token + bind address:**
   ```bash
   sudo -u aidea nano /opt/aidea/.env
   #   TELEGRAM_BOT_TOKEN=<token from BotFather>
   #   AIDEA_HOST=0.0.0.0      # if you want the web UI reachable from outside
   #   AIDEA_PORT=8000
   ```

6. **Enable the services:**
   ```bash
   sudo systemctl enable --now aidea-web aidea-bot
   sudo systemctl status aidea-web aidea-bot --no-pager
   sudo journalctl -u aidea-bot -f
   ```

## Path B — manual install (any Linux VM)

If you already have a VM and just want to install on it:

```bash
git clone https://github.com/frstrtr/AIdea.git /tmp/aidea-bootstrap
bash /tmp/aidea-bootstrap/deploy/install.sh
```

Then follow steps 4–6 from Path A.

## What the installer actually does

In order:

1. Installs `python3`, `python3-venv`, `git`, `curl` from apt.
2. Installs Node.js 22 and the agent CLI (`@anthropic-ai/claude-code`)
   from npm.
3. Creates a system user `aidea` with a home directory.
4. Clones (or `git pull`s) the repo to `/opt/aidea`.
5. Creates a Python venv at `/opt/aidea/.venv` and installs
   `requirements.txt`.
6. Copies `.env.example` → `.env` (mode 0600, owned by `aidea`) if no
   `.env` exists yet.
7. Drops `aidea-web.service` and `aidea-bot.service` into
   `/etc/systemd/system/` and reloads systemd.

It is safe to rerun — it pulls the latest commit and reinstalls
requirements without overwriting your `.env`.

## Exposing the web UI publicly

The systemd unit reads `AIDEA_HOST` and `AIDEA_PORT` from `.env`. Set
`AIDEA_HOST=0.0.0.0` to bind on all interfaces. Then either:

- Put a reverse proxy (nginx / caddy) in front with HTTPS, or
- Restrict to your own IPs via firewall, or
- Keep `AIDEA_HOST=127.0.0.1` and SSH-tunnel from your laptop:
  `ssh -L 8000:127.0.0.1:8000 ops@<vm-ip>`.

The `uvicorn` line in `aidea-web.service` already includes
`--proxy-headers --forwarded-allow-ips=*` for proxy use.

## Logs and state

- `journalctl -u aidea-web` / `-u aidea-bot` — service stdout.
- `/opt/aidea/usage.jsonl` — per-LLM-call tokens / duration / cost
  / rate-limit window.
- `/opt/aidea/decks/` — cached topic-aware decks.
- The `decks/` dir and `usage.jsonl` are gitignored; they grow with
  use and are safe to back up or rotate.

## Updating

```bash
sudo -u aidea git -C /opt/aidea pull
sudo -u aidea /opt/aidea/.venv/bin/pip install -r /opt/aidea/requirements.txt
sudo systemctl restart aidea-web aidea-bot
```

Or just re-run `install.sh` (idempotent).
