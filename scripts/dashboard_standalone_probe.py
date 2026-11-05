import asyncio
import sys
import threading
import time

sys.path.insert(0, "/home/sonny-rpi5/AI231-MEX3")

from agent.config import Config
from agent.dashboard import DashboardServer
from agent.observability import EventBus

print("DASHBOARD_HOST =", repr(Config.DASHBOARD_HOST))
print("DASHBOARD_PORT =", Config.DASHBOARD_PORT)

bus = EventBus()
srv = DashboardServer(bus, say_handler=None)
srv.start()
time.sleep(4)
print("threads:", [t.name for t in threading.enumerate()])
import urllib.request
try:
    r = urllib.request.urlopen("http://127.0.0.1:8900/", timeout=3)
    print("local http:", r.status)
except Exception as e:
    print("local probe failed:", e)
time.sleep(1)