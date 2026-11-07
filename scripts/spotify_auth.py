"""One-time Spotify authorization (Authorization Code flow).

Opens the browser for the Spotify login, catches the redirect on
http://127.0.0.1:8888/callback, exchanges the code, and caches tokens in
.spotify_cache (repo root). spotipy then auto-refreshes forever.

Run on the machine that will USE the tokens (the auth cache is portable —
copy it to the Pi after running here on the PC):
    .venv/Scripts/python.exe scripts/spotify_auth.py
"""
import http.server
import sys
import threading
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
from agent.config import Config  # noqa: E402  (loads .env)

CID = os.environ.get("SPOTIFY_CLIENT_ID", Config.__dict__.get("SPOTIFY_CLIENT_ID", ""))
CSEC = os.environ.get("SPOTIFY_CLIENT_SECRET", "")
REDIRECT = "http://127.0.0.1:8888/callback"
CACHE = str(Path(__file__).resolve().parent.parent / ".spotify_cache")
SCOPES = ("user-read-playback-state user-modify-playback-state "
          "user-read-currently-playing streaming user-read-private")

code_holder = {}


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if "code=" in self.path:
            code_holder["code"] = self.path.split("code=")[1].split("&")[0]
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(b"<h2>Spotify authorized - you can close this tab.</h2>")
        else:
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(b"<h2>Waiting...</h2>")

    def log_message(self, *a):
        pass


def main() -> None:
    import spotipy
    from spotipy.oauth2 import SpotifyOAuth

    url = ("https://accounts.spotify.com/authorize"
           f"?client_id={CID}&response_type=code&redirect_uri={REDIRECT}"
           f"&scope={SCOPES.replace(' ', '%20')}")
    server = http.server.HTTPServer(("127.0.0.1", 8888), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print("Opening browser for Spotify login…")
    webbrowser.open(url)
    print(f"If it doesn't open, paste this into a browser:\n  {url}")
    print("Waiting for the callback on http://127.0.0.1:8888/callback …")

    t0 = time.monotonic()
    while "code" not in code_holder and time.monotonic() - t0 < 300:
        time.sleep(0.3)
    server.shutdown()
    if "code" not in code_holder:
        sys.exit("timed out waiting for the Spotify callback")
    print("code received — exchanging for tokens…")

    oauth = SpotifyOAuth(client_id=CID, client_secret=CSEC, redirect_uri=REDIRECT,
                         scope=SCOPES, cache_path=CACHE, open_browser=False)
    oauth.get_access_token(code_holder["code"])   # exchange + write cache
    sp = spotipy.Spotify(auth_manager=oauth)
    me = sp.current_user()
    print(f"AUTHORIZED as {me.get('display_name')} ({me.get('id')})")
    print(f"token cache: {CACHE}")
    print("copy this file to the Pi: ~/AI231-MEX3/.spotify_cache")


if __name__ == "__main__":
    import time
    main()