"""Custom MCP server: Spotify playback control (Web API + Connect).

Voice control of the user's Spotify: search+play, transport, volume, queue,
devices. The audio itself plays on the target Spotify Connect device (the Pi
itself via librespot/spotifyd, or any device on the account).

Run:  python -m mcp_servers.spotify_server            (stdio)
      python -m mcp_servers.spotify_server --http 8935
"""
import argparse

from mcp.server.fastmcp import FastMCP

from services import spotify_client as spc

mcp = FastMCP("spotify")


@mcp.tool()
def spotify_search_play(query: str, device: str = "") -> str:
    """Search Spotify and play the best track match. device: optional Spotify
    Connect device name hint, e.g. 'pi' or 'phone' (empty = current/first)."""
    try:
        return spc.search_and_play(query, device or None)
    except spc.SpotifyError as e:
        raise ValueError(str(e))


@mcp.tool()
def spotify_pause() -> str:
    """Pause Spotify playback."""
    try:
        return spc.pause()
    except spc.SpotifyError as e:
        raise ValueError(str(e))


@mcp.tool()
def spotify_resume(device: str = "") -> str:
    """Resume Spotify playback (optionally on a named device)."""
    try:
        return spc.resume(device or None)
    except spc.SpotifyError as e:
        raise ValueError(str(e))


@mcp.tool()
def spotify_next() -> str:
    """Skip to the next track."""
    try:
        return spc.next_track()
    except spc.SpotifyError as e:
        raise ValueError(str(e))


@mcp.tool()
def spotify_previous() -> str:
    """Go back to the previous track."""
    try:
        return spc.previous_track()
    except spc.SpotifyError as e:
        raise ValueError(str(e))


@mcp.tool()
def spotify_volume(level: int) -> str:
    """Set Spotify volume 0-100."""
    try:
        return spc.set_volume(level)
    except spc.SpotifyError as e:
        raise ValueError(str(e))


@mcp.tool()
def spotify_now_playing() -> str:
    """What track is playing right now, and where."""
    try:
        return spc.now_playing()
    except spc.SpotifyError as e:
        raise ValueError(str(e))


@mcp.tool()
def spotify_devices() -> str:
    """List visible Spotify Connect devices."""
    try:
        return spc.list_devices()
    except spc.SpotifyError as e:
        raise ValueError(str(e))


@mcp.tool()
def spotify_queue(query: str) -> str:
    """Find a track and add it to the play queue."""
    try:
        return spc.add_to_queue(query)
    except spc.SpotifyError as e:
        raise ValueError(str(e))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--http", type=int, metavar="PORT")
    args = parser.parse_args()
    if args.http:
        mcp.settings.port = args.http
        mcp.settings.host = "127.0.0.1"
        mcp.run(transport="streamable-http")
    else:
        mcp.run()