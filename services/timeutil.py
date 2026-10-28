"""Time helper with IANA timezone support (used by text + voice get_time tools)."""
import datetime
import zoneinfo


def now_str(tz: str = "") -> str:
    """Current date/time/weekday, optionally in an IANA timezone like
    'America/New_York'. Falls back to local time on bad input and says so."""
    try:
        if tz:
            zone = zoneinfo.ZoneInfo(tz)
            now = datetime.datetime.now(zone)
            return now.strftime("%A, %d %B %Y, %H:%M:%S (%Z, UTC%z)")
    except (zoneinfo.ZoneInfoNotFoundError, ValueError, KeyError):
        pass
    local = datetime.datetime.now()
    note = f" (requested timezone '{tz}' unknown; showing local time)" if tz else ""
    return local.strftime("%A, %d %B %Y, %H:%M:%S") + note
