"""PortoAPI payment integration for AIdea subscriptions.

Model A (Invoice API) with the per-customer-address customization: each
Telegram user gets ONE stable USDT-TRC20 receiving address (PortoAPI binds it
to our ``customer_ref`` = ``uid=<telegram_id>``). A signed ``invoice.paid``
webhook flips the user to a subscriber via ``quota.set_subscriber``. Fully
non-custodial — we never hold the seed; PortoAPI derives addresses from our
public xPub.

Config (env / .env), all optional until you go live:
  AIDEA_PORTO_WEBHOOK_SECRET  HMAC-SHA256 key to verify inbound webhooks
                              (presence enables the webhook receiver)
  AIDEA_PORTO_API_KEY         bearer token for outbound invoice creation
  AIDEA_PORTO_BASE_URL        API base (default Nile testnet 192.168.86.205:8000)
  AIDEA_PORTO_PRICE_USDT      price of one subscription period (enables buying)
  AIDEA_PORTO_PERIOD_DAYS     days granted per period (default 30)
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import quota

_PROCESSED = Path(__file__).parent / "porto_processed.json"
_LOCK = threading.Lock()


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------
def webhook_secret() -> str:
    return os.environ.get("AIDEA_PORTO_WEBHOOK_SECRET", "").strip()


def api_key() -> str:
    return os.environ.get("AIDEA_PORTO_API_KEY", "").strip()


def base_url() -> str:
    return os.environ.get(
        "AIDEA_PORTO_BASE_URL", "http://192.168.86.205:8000",
    ).rstrip("/")


def period_days() -> int:
    try:
        return max(1, int(os.environ.get("AIDEA_PORTO_PERIOD_DAYS", "30") or 30))
    except ValueError:
        return 30


def price_usdt() -> float | None:
    raw = os.environ.get("AIDEA_PORTO_PRICE_USDT", "").strip()
    try:
        return float(raw) if raw else None
    except ValueError:
        return None


def webhook_configured() -> bool:
    """True once the HMAC secret is set — the receiver rejects calls until then."""
    return bool(webhook_secret())


def buying_enabled() -> bool:
    """True once we can actually create invoices (api key + a price)."""
    return bool(api_key()) and price_usdt() is not None


# --------------------------------------------------------------------------
# signature verification — matches PortoAPI's HMAC-SHA256 over the raw body
# --------------------------------------------------------------------------
def verify(raw_body: bytes, signature: str) -> bool:
    secret = webhook_secret()
    if not secret:
        return False
    expected = hmac.new(secret.encode(), raw_body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature or "")


# --------------------------------------------------------------------------
# idempotency — dedupe on tx_hash (fallback invoice_id+event)
# --------------------------------------------------------------------------
def _load_processed() -> dict:
    try:
        with _PROCESSED.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_processed(data: dict) -> None:
    tmp = _PROCESSED.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f)
    tmp.replace(_PROCESSED)


def _mark(key: str, info: dict) -> None:
    data = _load_processed()
    data[key] = info
    _save_processed(data)


def _unmark(key: str) -> None:
    data = _load_processed()
    if data.pop(key, None) is not None:
        _save_processed(data)


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------
_UID_RE = re.compile(r"uid=(\d+)")
_DAYS_RE = re.compile(r"days=(\d+)")


def parse_uid_days(invoice: dict) -> tuple[int | None, int | None]:
    """Resolve (uid, days). uid from customer_ref/description; days from
    description, else derived from the amount paid and our price (the
    running-tab model for direct/renewal payments with no explicit days)."""
    blob = f"{invoice.get('customer_ref', '')} {invoice.get('description', '')}"
    uid = int(m.group(1)) if (m := _UID_RE.search(blob)) else None
    days = int(m.group(1)) if (m := _DAYS_RE.search(blob)) else None
    if days is None:
        price = price_usdt()
        try:
            paid = float(invoice.get("paid_amount") or invoice.get("amount") or 0)
        except (TypeError, ValueError):
            paid = 0.0
        if price and price > 0 and paid > 0:
            days = max(1, round(paid / price * period_days()))
    return uid, days


# --------------------------------------------------------------------------
# webhook handling — idempotent; activates on paid/overpaid
# --------------------------------------------------------------------------
_PAID_EVENTS = {"invoice.paid", "invoice.overpaid"}


def handle_event(payload: dict) -> dict:
    """Process a *verified* webhook payload. Idempotent (dedupe on tx_hash).
    On a paid/overpaid event with a resolvable uid+days, extend the
    subscription via quota.set_subscriber and persist before returning.
    Returns a small result dict; never raises for ordinary outcomes."""
    event = payload.get("event", "")
    invoice = payload.get("invoice") or {}
    tx = payload.get("transaction") or {}
    inv_id = invoice.get("invoice_id")
    tx_hash = tx.get("tx_hash")
    key = tx_hash or f"inv:{inv_id}:{event}"

    with _LOCK:
        if key in _load_processed():
            return {"ok": True, "deduped": True, "key": key}

        if event in _PAID_EVENTS:
            uid, days = parse_uid_days(invoice)
            if not (uid and days):
                # Don't mark — leave it for reconciliation / manual handling.
                return {"ok": True, "activated": False, "event": event,
                        "reason": "unresolved uid/days", "uid": uid, "days": days,
                        "invoice_id": inv_id}
            # Reserve the key first so a duplicate delivery can't double-credit;
            # roll back the mark if activation itself fails.
            _mark(key, {"event": event, "ts": time.time(),
                        "invoice_id": inv_id, "uid": uid, "days": days})
            try:
                until = quota.set_subscriber(uid, days=days)
            except Exception:
                _unmark(key)
                raise
            return {"ok": True, "activated": True, "event": event, "key": key,
                    "uid": uid, "days": days, "subscribed_until": until}

        # Non-activating terminal/info events — record so we don't re-handle.
        _mark(key, {"event": event, "ts": time.time(), "invoice_id": inv_id})
        return {"ok": True, "activated": False, "event": event, "key": key}


# --------------------------------------------------------------------------
# outbound: create an invoice (the buy flow)
# --------------------------------------------------------------------------
def create_invoice(uid: int, days: int | None = None,
                   amount: float | None = None) -> dict:
    """Create a subscription invoice for a Telegram user. Returns PortoAPI's
    invoice object (incl. the customer's stable ``address`` + ``payment_uri``).
    Synchronous urllib — call via asyncio.to_thread from async code."""
    if not buying_enabled():
        raise RuntimeError(
            "PortoAPI buying not configured (need AIDEA_PORTO_API_KEY + "
            "AIDEA_PORTO_PRICE_USDT)"
        )
    days = days or period_days()
    amount = amount if amount is not None else price_usdt()
    body = {
        "amount": amount,
        "customer_ref": f"uid={uid}",
        "description": f"aidea-sub:uid={uid};days={days}",
    }
    req = urllib.request.Request(
        f"{base_url()}/v1/invoices",
        data=json.dumps(body).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key()}",
        },
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.load(resp)
