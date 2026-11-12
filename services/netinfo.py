"""Network and dashboard location info for the dashboard_info voice tool.

Answers questions like "how do I connect to the dashboard?", "what network
and IP are the dashboard on?" — returns the dashboard URL, the Pi's current
LAN IP, and the Wi-Fi network it is connected to.
"""
import socket
import subprocess

from agent.config import Config


def _local_ip() -> str:
    """The Pi's LAN IP (via a UDP probe — no packets are actually sent)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def _ssid() -> str | None:
    """Current Wi-Fi network name (None when on ethernet or unavailable)."""
    # NetworkManager (Debian 12/13 default)
    try:
        out = subprocess.run(
            ["nmcli", "-t", "-f", "active,ssid", "dev", "wifi"],
            capture_output=True, text=True, timeout=5).stdout
        for line in out.splitlines():
            if line.startswith("yes:"):
                ssid = line.split(":", 1)[1].strip()
                if ssid:
                    return ssid
    except (OSError, subprocess.SubprocessError):
        pass
    # wpa_supplicant / iw fallback
    try:
        out = subprocess.run(
            ["iwgetid", "-r"], capture_output=True, text=True, timeout=5).stdout
        ssid = out.strip()
        if ssid:
            return ssid
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def dashboard_info() -> dict:
    """Everything the agent needs to answer dashboard-connection questions."""
    ip = _local_ip()
    port = Config.DASHBOARD_PORT
    exposed = Config.DASHBOARD_HOST in ("0.0.0.0", "::")
    return {
        "dashboard_url": f"http://{ip}:{port}",
        "ip": ip,
        "port": port,
        "open_to_network": exposed,
        "same_network_required": True,
        "ssid": _ssid(),
        "hostname": socket.gethostname(),
    }
