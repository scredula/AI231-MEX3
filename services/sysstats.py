"""System stats for the Pi/PC — CPU load, temperature, memory, disk, uptime."""
import datetime
import shutil

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None


def system_stats() -> str:
    """One-paragraph system summary, Pi-friendly, no psutil hard dependency."""
    if psutil is None:
        return "psutil not installed — run: pip install psutil"
    vm = psutil.virtual_memory()
    disk = shutil.disk_usage("/")
    cpu = psutil.cpu_percent(interval=0.3)
    up = datetime.datetime.now() - datetime.datetime.fromtimestamp(psutil.boot_time())
    lines = [
        f"CPU: {cpu:.0f}% across {psutil.cpu_count()} cores",
        f"Memory: {vm.percent:.0f}% used ({vm.available // (1024**2)} MB free of {vm.total // (1024**2)} MB)",
        f"Disk (/): {(disk.used / disk.total) * 100:.0f}% used ({disk.free // (1024**3)} GB free of {disk.total // (1024**3)} GB)",
        f"Uptime: {str(up).split('.')[0]}",
    ]
    temp = None
    try:
        temps = (psutil.sensors_temperatures() if hasattr(psutil, 'sensors_temperatures') else {}) or {}
        for key in ("cpu_thermal", "cpu-thermal", "k10temp", "coretemp", "soc_thermal"):
            if key in temps and temps[key]:
                temp = temps[key][0].current
                break
        if temp is None and temps:
            first = next(iter(temps.values()))
            if first:
                temp = first[0].current
    except (AttributeError, OSError):
        pass
    if temp is not None:
        lines.insert(1, f"CPU temp: {temp:.0f}°C")
    return "\n".join(lines)