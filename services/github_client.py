"""GitHub publishing via REST API + local git push. Separate from website generation.

publish_website() creates a PRIVATE repo (never changes visibility of existing repos),
pushes the site, configures GitHub Pages and verifies deployment before reporting success.
Requires GITHUB_TOKEN with repo scope. Verification: repo exists + Pages build status.
"""
import base64
import json
import time
import logging

import httpx

from agent.config import Config

log = logging.getLogger("github")

API = "https://api.github.com"


class GitHubError(Exception):
    pass


def _headers(token: str) -> dict:
    if not token:
        raise GitHubError("GITHUB_TOKEN not configured — add it to .env to enable publishing")
    return {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}


def _user_login(token: str) -> str:
    r = httpx.get(f"{API}/user", headers=_headers(token), timeout=30)
    if r.status_code != 200:
        raise GitHubError(f"cannot authenticate GitHub token: {r.status_code}")
    return r.json()["login"]


def create_repo(token: str, name: str) -> dict:
    r = httpx.post(f"{API}/user/repos", headers=_headers(token), timeout=30,
                   json={"name": name, "private": True, "auto_init": False})
    if r.status_code == 201:
        return r.json()
    if r.status_code == 422:  # already exists — reuse, do not change visibility
        login = _user_login(token)
        return {"full_name": f"{login}/{name}", "html_url": f"https://github.com/{login}/{name}"}
    raise GitHubError(f"create_repo failed: {r.status_code} {r.text[:200]}")


def _put_file(token: str, full_name: str, path: str, content: bytes, message: str) -> None:
    r = httpx.put(f"{API}/repos/{full_name}/contents/{path}", headers=_headers(token), timeout=30,
                  json={"message": message, "content": base64.b64encode(content).decode()})
    if r.status_code not in (200, 201):
        raise GitHubError(f"put {path} failed: {r.status_code} {r.text[:200]}")


def enable_pages(token: str, full_name: str, branch: str = "main") -> None:
    r = httpx.post(f"{API}/repos/{full_name}/pages", headers=_headers(token), timeout=30,
                   json={"source": {"branch": branch, "path": "/"}})
    if r.status_code not in (201, 409):  # 409 = already enabled
        raise GitHubError(f"enable_pages failed: {r.status_code} {r.text[:200]}")


def wait_for_pages(token: str, full_name: str, timeout: int = 120) -> str:
    """Poll the Pages endpoint until the site status is built/live. Verified before reporting success."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = httpx.get(f"{API}/repos/{full_name}/pages", headers=_headers(token), timeout=30)
        if r.status_code == 200:
            info = r.json()
            status = info.get("status", "")
            url = info.get("html_url", "")
            if status in ("built", "live"):
                return url
        time.sleep(5)
    raise GitHubError(f"pages did not finish building within {timeout}s (repo pushed, check Actions tab)")


def publish_website(project: str = "site", repo_name: str = "") -> dict:
    """Push workspace/<project> to a private repo + GitHub Pages. Returns verified URLs."""
    token = Config.GITHUB_TOKEN
    src = Config.WORKSPACE_DIR / project
    if not (src / "index.html").exists():
        raise GitHubError(f"{src}/index.html not found — build the website first")

    login = _user_login(token)
    repo_name = repo_name or f"{src.name}-site"
    repo = create_repo(token, repo_name)
    full_name = repo["full_name"]

    for f in sorted(src.rglob("*")):
        if f.is_file() and ".git" not in f.parts:
            _put_file(token, full_name, str(f.relative_to(src)), f.read_bytes(), f"add {f.name}")

    enable_pages(token, full_name)
    pages_url = wait_for_pages(token, full_name)
    return {"repo_url": repo.get("html_url", f"https://github.com/{full_name}"), "pages_url": pages_url}
