"""Live E2E: ask for another city's time via /say and verify get_time handles tz."""
import asyncio
import json
import sys
import urllib.request

sys.path.insert(0, ".")


async def main():
    await asyncio.sleep(1)
    req = urllib.request.Request(
        "http://127.0.0.1:8900/say", method="POST",
        data=json.dumps({"text": "What time is it in New York right now?"}).encode(),
        headers={"Content-Type": "application/json"})
    urllib.request.urlopen(req, timeout=10)
    print("say: sent")

    for _ in range(60):  # up to ~60s
        await asyncio.sleep(1)
        try:
            hist = json.loads(urllib.request.urlopen("http://127.0.0.1:8900/api/events", timeout=5).read())
        except Exception:
            continue
        results = [e for e in hist.get("events", []) if e.get("type") == "tool_result" and e.get("name") == "get_time"]
        if results:
            print("GET_TIME RESULT:", results[-1].get("result"))
            break
    else:
        print("no get_time result yet; last events:",
              [(e.get("type"), e.get("name")) for e in hist.get("events", [])[-8:]])


asyncio.run(main())