"""Spotify control via the Web API (spotipy) — shared by the voice tools and the
spotify MCP server.

Auth: one-time Authorization Code flow (scripts/spotify_auth.py) caches tokens
in .spotify_cache (repo root, gitignored); spotipy auto-refreshes afterwards.

NOTE on audio: the Web API CONTROLS playback but carries no audio. Play the
sound on a Spotify Connect device — e.g. the Pi itself via librespot/spotifyd,
or any device logged into the account (phone, laptop, speakers).
"""
import logging
import os
import time
from pathlib import Path

from agent.config import Config  # noqa: E402,F401  (loads .env into os.environ)

log = logging.getLogger("spotify")

CACHE_PATH = Path(os.environ.get("SPOTIFY_CACHE",
                                 Path(__file__).resolve().parent.parent / ".spotify_cache"))

SCOPES = ("user-read-playback-state user-modify-playback-state "
          "user-read-currently-playing streaming user-read-private")

_client = None


def _sp():
    """Lazy spotipy client; raises SpotifyError with a helpful message if not authed."""
    global _client
    if _client is not None:
        return _client
    try:
        import spotipy
        from spotipy.oauth2 import SpotifyOAuth
    except ImportError as e:
        raise SpotifyError(f"spotipy not installed: {e}")
    cid = os.environ.get("SPOTIFY_CLIENT_ID")
    sec = os.environ.get("SPOTIFY_CLIENT_SECRET")
    redirect = os.environ.get("SPOTIFY_REDIRECT_URI", "http://127.0.0.1:8888/callback")
    if not cid or not sec:
        raise SpotifyError("SPOTIFY_CLIENT_ID / SPOTIFY_CLIENT_SECRET missing in .env")
    if not CACHE_PATH.exists():
        raise SpotifyError("not authorized yet — run scripts/spotify_auth.py once (browser login)")
    _client = spotipy.Spotify(auth_manager=SpotifyOAuth(
        client_id=cid, client_secret=sec, redirect_uri=redirect,
        scope=SCOPES, cache_path=str(CACHE_PATH),
        open_browser=False, requests_timeout=10))
    return _client


class SpotifyError(Exception):
    pass


def _handle(fn):
    """Run a spotipy call; translate common failures into friendly text."""
    try:
        return fn()
    except SpotifyError:
        raise
    except Exception as e:
        msg = str(e)
        if "NO_ACTIVE_DEVICE" in msg or "device not found" in msg.lower():
            raise SpotifyError("no active Spotify device — say 'play it on the Pi' or open Spotify somewhere first")
        if "RESTRICTED" in msg or "403" in msg:
            raise SpotifyError("Spotify rejected the request (check premium account and app settings)")
        raise SpotifyError(msg[:200])


# ---- playback state helpers ----

def _find_device(name_hint: str) -> dict | None:
    """Find a Connect device by fuzzy name (e.g. 'pi', 'phone')."""
    sp = _sp()
    devices = (_handle(lambda: sp.devices()) or {}).get("devices", [])
    hint = (name_hint or "").strip().lower()
    if not hint:
        return None
    for d in devices:
        if hint in d["name"].lower():
            return d
    for d in devices:                      # also match type words
        if hint in (d.get("type") or "").lower():
            return d
    return None


def _ensure_device(name_hint: str | None) -> dict | None:
    """Transfer playback to the hinted device if one is given; otherwise return
    the current active device (or None)."""
    sp = _sp()
    if name_hint:
        dev = _find_device(name_hint)
        if dev is None:
            names = ", ".join(d["name"] for d in (_handle(lambda: sp.devices()).get("devices", [])))
            raise SpotifyError(f"no device matching {name_hint!r} — available: {names or 'none'}")
        _handle(lambda: sp.transfer_playback(dev["id"], force_play=True))
        time.sleep(1.0)                    # Connect handover settles
        return dev
    cur = _handle(lambda: sp.current_playback())
    if cur and cur.get("device"):
        return cur["device"]
    # nothing active: pick the Pi (AI231) if present, else the first device
    dev = _find_device("ai231") or _find_device("pi")
    if dev is None:
        devices = _handle(lambda: sp.devices()).get("devices", [])
        dev = devices[0] if devices else None
    if dev:
        _handle(lambda: sp.transfer_playback(dev["id"], force_play=False))
        time.sleep(1.0)
    return dev


# ---- public API (shared by voice tools + MCP server) ----

def search_and_play(query: str, device_hint: str | None = None) -> str:
    """Search tracks for `query` and start playback of the best match."""
    sp = _sp()
    res = _handle(lambda: sp.search(q=query, type="track", limit=5))
    tracks = res.get("tracks", {}).get("items", [])
    if not tracks:
        # fall back to artist search and play top track
        res = _handle(lambda: sp.search(q=query, type="artist", limit=1))
        artists = res.get("artists", {}).get("items", [])
        if not artists:
            return f"No Spotify results for {query!r}."
        artist = artists[0]
        top = _handle(lambda: sp.artist_top_tracks(artist["id"]))["tracks"]
        if not top:
            return f"No playable tracks for {query!r}."
        tracks = top
    track = tracks[0]
    dev = _ensure_device(device_hint)
    _handle(lambda: sp.start_playback(device_id=dev["id"] if dev else None,
                                      uris=[track["uri"]]))
    # Single-track contexts have no queue: 'next' would just restart the same
    # track and the device pauses. Queue the rest of the album so next/previous
    # walk the collection naturally.
    try:
        album_id = track.get("album", {}).get("id")
        if album_id:
            album = _handle(lambda: sp.album_tracks(album_id, limit=50))
            album_uris = [t["uri"] for t in album.get("items", []) if t["uri"] != track["uri"]]
            for uri in album_uris:
                _handle(lambda: sp.add_to_queue(uri, device_id=dev["id"] if dev else None))
    except Exception as e:
        log.warning(f"could not queue album context after single-track play: {e}")
    artists = ", ".join(a["name"] for a in track["artists"])
    return (f"Playing {track['name']} by {artists} on "
            f"{(dev or {}).get('name', 'Spotify')}.")


def play_context(query: str, kind: str = "album", device_hint: str | None = None) -> str:
    """Play an album/artist/playlist by name."""
    sp = _sp()
    kind = kind if kind in ("album", "artist", "playlist") else "album"
    res = _handle(lambda: sp.search(q=query, type=kind, limit=1))
    items = res.get(kind + "s", {}).get("items", [])
    if not items:
        return f"No {kind} results for {query!r}."
    item = items[0]
    dev = _ensure_device(device_hint)
    uri = item["uri"] if kind != "artist" else None
    if kind == "artist":
        _handle(lambda: sp.start_playback(device_id=dev["id"] if dev else None,
                                          context_uri=item["uri"]))
    else:
        _handle(lambda: sp.start_playback(device_id=dev["id"] if dev else None,
                                          context_uri=item["uri"]))
    return f"Playing {kind} {item['name']} on {(dev or {}).get('name', 'Spotify')}."


def pause() -> str:
    _handle(lambda: _sp().pause_playback())
    return "Paused."


def resume(device_hint: str | None = None) -> str:
    _ensure_device(device_hint)
    _handle(lambda: _sp().start_playback())
    return "Resumed."


def _resume_if_paused() -> None:
    """After a skip, Spotify Connect devices can land paused (empty queue)."""
    cp = _handle(lambda: _sp().current_playback())
    if cp and not cp.get("is_playing"):
        _handle(lambda: _sp().start_playback())


def next_track() -> str:
    _handle(lambda: _sp().next_track())
    time.sleep(0.6)
    _resume_if_paused()
    time.sleep(0.4)
    cp = _handle(lambda: _sp().current_playback())
    name = ((cp or {}).get("item") or {}).get("name", "next track")
    return f"Next: {name}."


def previous_track() -> str:
    _handle(lambda: _sp().previous_track())
    time.sleep(0.6)
    _resume_if_paused()
    return "Previous track."


def set_volume(level: int) -> str:
    level = max(0, min(100, int(level)))
    _handle(lambda: _sp().volume(level))
    return f"Volume {level}%."


def now_playing() -> str:
    cp = _handle(lambda: _sp().current_playback())
    if not cp or not cp.get("item"):
        return "Nothing is playing right now."
    item = cp["item"]
    artists = ", ".join(a["name"] for a in item.get("artists", []))
    state = "paused" if cp.get("is_playing") is False else "playing"
    return (f"{state}: {item['name']} by {artists} on "
            f"{cp['device']['name']} at {cp.get('progress_ms', 0) // 1000} seconds.")


def list_devices() -> str:
    devices = _handle(lambda: _sp().devices()).get("devices", [])
    if not devices:
        return "No Spotify Connect devices visible — open Spotify somewhere first."
    return "; ".join(f"{d['name']} ({d['type']}){' ACTIVE' if d.get('is_active') else ''}"
                     for d in devices)


def add_to_queue(query: str) -> str:
    sp = _sp()
    res = _handle(lambda: sp.search(q=query, type="track", limit=1))
    tracks = res.get("tracks", {}).get("items", [])
    if not tracks:
        return f"No track found for {query!r}."
    _handle(lambda: sp.add_to_queue(tracks[0]["uri"]))
    return f"Queued {tracks[0]['name']}."


def auth_status() -> str:
    try:
        _sp()
        return "Spotify authorized."
    except SpotifyError as e:
        return f"Spotify NOT authorized: {e}"


# ---- structured data (web-UI player; returns dicts, not sentences) ----

def now_playing_raw() -> dict:
    """Full playback state for the web-UI player ({} when nothing is playing)."""
    cp = _handle(lambda: _sp().current_playback())
    if not cp or not cp.get("item"):
        return {"playing": False, "active": bool(cp)}
    item = cp["item"]
    return {
        "playing": bool(cp.get("is_playing")),
        "title": item.get("name"),
        "artists": ", ".join(a.get("name", "") for a in item.get("artists", [])),
        "album": item.get("album", {}).get("name"),
        "art": (item.get("album", {}).get("images") or [{}])[0].get("url"),
        "progress_ms": cp.get("progress_ms") or 0,
        "duration_ms": item.get("duration_ms") or 0,
        "device": cp.get("device", {}).get("name"),
        "device_id": cp.get("device", {}).get("id"),
        "volume": cp.get("device", {}).get("volume_percent"),
        "shuffle": cp.get("shuffle_state"),
    }


def seek(position_ms: int, device_hint: str | None = None) -> str:
    """Move playback to an absolute position (milliseconds)."""
    dev = _ensure_device(device_hint)
    _handle(lambda: _sp().seek_track(int(position_ms),
                                     device_id=dev["id"] if dev else None))
    return f"Seeked to {int(position_ms) // 1000}s."


def list_devices_raw() -> list:
    """Connect devices as dicts: id, name, type, active."""
    res = _handle(lambda: _sp().devices()).get("devices", [])
    return [{"id": d["id"], "name": d["name"], "type": d.get("type"),
             "active": d.get("is_active", False)} for d in res]


def transfer(device_id: str) -> str:
    """Move playback to a specific Connect device (no forced play)."""
    _handle(lambda: _sp().transfer_playback(device_id, force_play=False))
    return "Playback moved."


def search_raw(query: str, limit: int = 5) -> dict:
    """Top matches across tracks, albums and playlists for the web-UI dropdown."""
    sp = _sp()
    out = {"tracks": [], "albums": [], "playlists": []}
    try:
        res = _handle(lambda: sp.search(q=query, type="track", limit=limit))
        for t in res.get("tracks", {}).get("items", []):
            out["tracks"].append({
                "uri": t["uri"],
                "name": t.get("name"),
                "artists": ", ".join(a.get("name", "") for a in t.get("artists", [])),
                "album": t.get("album", {}).get("name"),
                "art": (t.get("album", {}).get("images") or [{}])[0].get("url"),
                "duration_ms": t.get("duration_ms") or 0,
            })
    except Exception as e:
        log.warning(f"track search failed: {e}")
    try:
        res = _handle(lambda: sp.search(q=query, type="album", limit=limit))
        for a in res.get("albums", {}).get("items", []):
            out["albums"].append({
                "uri": a.get("uri"),
                "name": a.get("name"),
                "artists": ", ".join(x.get("name", "") for x in a.get("artists", [])),
                "art": (a.get("images") or [{}])[0].get("url"),
            })
    except Exception as e:
        log.warning(f"album search failed: {e}")
    try:
        res = _handle(lambda: sp.search(q=query, type="playlist", limit=limit))
        for p in res.get("playlists", {}).get("items", []):
            if not p:
                continue
            out["playlists"].append({
                "uri": p.get("uri"),
                "name": p.get("name"),
                "owner": (p.get("owner") or {}).get("display_name"),
                "art": (p.get("images") or [{}])[0].get("url"),
            })
    except Exception as e:
        log.warning(f"playlist search failed: {e}")
    return out


def play_uri(uri: str, device_hint: str | None = None) -> str:
    """Start playback of a specific URI (track, album or playlist)."""
    sp = _sp()
    dev = _ensure_device(device_hint)
    _handle(lambda: sp.start_playback(device_id=dev["id"] if dev else None,
                                      uris=[uri] if uri.startswith("spotify:track") else None,
                                      context_uri=uri if not uri.startswith("spotify:track") else None))
    return "Playing selection."
