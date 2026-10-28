"""Send Telegram messages via the Bot API (no SDK — plain HTTPS POST).

Env: TELEGRAM_BOT_TOKEN (from @BotFather), TELEGRAM_CHAT_ID (your chat id).
"""
import httpx

from agent.config import Config

API = "https://api.telegram.org/bot{token}/sendMessage"


class TelegramError(Exception):
    pass


async def send_message(text: str) -> str:
    """Send `text` to the configured chat. Returns a confirmation or raises."""
    text = (text or "").strip()[:4000]
    if not text:
        raise TelegramError("message text is required")
    token = Config.TELEGRAM_BOT_TOKEN
    chat = Config.TELEGRAM_CHAT_ID
    if not token or token.startswith("PASTE_"):
        raise TelegramError("TELEGRAM_BOT_TOKEN not configured in .env (get one from @BotFather)")
    if not chat:
        raise TelegramError("TELEGRAM_CHAT_ID not configured in .env")
    try:
        async with httpx.AsyncClient(timeout=15) as c:
            r = await c.post(API.format(token=token),
                             json={"chat_id": chat, "text": text})
    except httpx.HTTPError as e:
        raise TelegramError(f"network error: {e}") from e
    d = r.json()
    if not d.get("ok"):
        desc = d.get("description", r.text[:120])
        raise TelegramError(f"telegram api error: {desc}")
    return f"sent to chat {chat}"