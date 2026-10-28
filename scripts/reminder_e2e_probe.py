"""Live E2E: set a reminder by voice via /say, then wait for it to fire and
be spoken (announcement event). Also exercises send_telegram error path."""
import asyncio
import json
import sys
import urllib.request

sys.path.insert(0, ".")


async def main():
    await asyncio.sleep(1)
    req = urllib.request.Request(
        "http://127.0.0.1:8900/say", method="POST",
        data=json.dumps({"text": "Remind me in one minute to check the oven, please."}).encode(),
        headers={"Content-Type": "application/json"})
    urllib.request.urlopen(req, timeout=10)
    print("say: sent")

    fired = False
    for _ in range(110):  # up to ~110s (reminder fires at ~60s)
        await asyncio.sleep(1)
        try:
            hist = json.loads(urllib.request.urlopen(
                "http://127.0.0.1:8900/api/events", timeout=5).read())
        except Exception:
            continue
        results = [e for e in hist.get("events", []) if e.get("type") == "tool_result"]
        if results and not fired:
            for e in results:
                print(f"TOOL {e.get('name')}: {str(e.get('result'))[:100]}")
            fired = True
        anns = [e for e in hist.get("events", []) if e.get("type") == "announcement"]
        if anns:
            print("ANNOUNCEMENT:", anns[-1].get("text"))
            return
    print("no announcement within timeout")


asyncio.run(main())