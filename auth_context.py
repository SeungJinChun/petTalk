from __future__ import annotations

import base64
import json
import logging
import os
import time
from typing import Any
from urllib.parse import unquote
from urllib.request import Request as UrlRequest, urlopen
from urllib.error import HTTPError, URLError

from fastapi import Request

logger = logging.getLogger(__name__)
AUTH_TOKEN_COOKIE = "supabase_access_token"


def _bearer_token(request: Request) -> str | None:
    auth_header = request.headers.get("authorization") or request.headers.get("Authorization")
    if not auth_header:
        return None
    prefix = "Bearer "
    if not auth_header.startswith(prefix):
        return None
    token = auth_header[len(prefix):].strip()
    return token or None


def _request_access_token(request: Request) -> str | None:
    bearer_token = _bearer_token(request)
    if bearer_token:
        return bearer_token

    cookie_token = request.cookies.get(AUTH_TOKEN_COOKIE)
    if not cookie_token:
        return None

    token = unquote(cookie_token).strip()
    return token or None


def _decode_token_payload(token: str) -> dict[str, Any] | None:
    parts = token.split(".")
    if len(parts) != 3:
        return None

    payload = parts[1]
    payload += "=" * (-len(payload) % 4)

    try:
        decoded = base64.urlsafe_b64decode(payload.encode("ascii"))
        data = json.loads(decoded.decode("utf-8"))
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError):
        return None

    if not isinstance(data, dict):
        return None

    return data


def _user_from_token_payload(token: str, supabase_url: str) -> dict[str, Any] | None:
    payload = _decode_token_payload(token)
    if not payload:
        return None

    user_id = payload.get("sub")
    email = payload.get("email")
    expires_at = payload.get("exp")
    issuer = str(payload.get("iss") or "")
    expected_issuer = f"{supabase_url}/auth/v1" if supabase_url else ""

    if not user_id:
        return None
    if isinstance(expires_at, (int, float)) and expires_at <= time.time():
        return None
    if expected_issuer and issuer.rstrip("/") != expected_issuer.rstrip("/"):
        return None

    return {
        "user_id": str(user_id),
        "email": str(email) if email else None,
        "auth_provider": "supabase-jwt",
        "authenticated": True,
        "verified": False,
    }


def resolve_auth_user(request: Request) -> dict[str, Any]:
    """Resolve the current user.

    Production:
      Frontend sends Supabase access token:
        Authorization: Bearer <access_token>

      This function asks Supabase Auth `/auth/v1/user` who the token belongs to.

    Development fallback:
      If no token or Supabase env vars are missing, returns local-user.
      This keeps your current local workflow working.
    """
    token = _request_access_token(request)
    supabase_url = os.environ.get("SUPABASE_URL", "").rstrip("/")
    anon_key = os.environ.get("SUPABASE_ANON_KEY", "")

    if token and supabase_url and anon_key:
        try:
            req = UrlRequest(
                f"{supabase_url}/auth/v1/user",
                headers={
                    "apikey": anon_key,
                    "Authorization": f"Bearer {token}",
                },
                method="GET",
            )
            with urlopen(req, timeout=8) as response:
                payload = json.loads(response.read().decode("utf-8"))
            user_id = payload.get("id") or payload.get("sub")
            email = payload.get("email")
            if user_id:
                return {
                    "user_id": str(user_id),
                    "email": email,
                    "auth_provider": "supabase",
                    "authenticated": True,
                    "verified": True,
                }
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            logger.warning("Supabase auth token verification failed: %s", exc)

    if token:
        token_user = _user_from_token_payload(token, supabase_url)
        if token_user:
            return token_user

    # Optional dev override for quick local testing.
    dev_user_id = request.headers.get("x-dev-user-id") or "local-user"
    dev_email = request.headers.get("x-dev-user-email")
    return {
        "user_id": dev_user_id,
        "email": dev_email,
        "auth_provider": "local-dev",
        "authenticated": False,
        "verified": False,
    }
