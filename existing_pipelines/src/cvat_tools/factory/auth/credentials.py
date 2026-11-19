"""Zero-friction CVAT credential discovery: profile -> env -> .env -> Credential Manager -> secure prompt."""
from __future__ import annotations

import getpass
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from cvat_tools.core.client import load_env, load_profile
from cvat_tools.factory.auth.wincred import CredentialStore, get_default_credential_store
from cvat_tools.factory.cvat.gateway import ConnectivityResult, CvatGateway


def credential_target(profile: str, host: str) -> str:
    """Canonical Windows Credential Manager target name (never contains PAT)."""
    clean_host = host.rstrip("/").lower()
    return f"cvat-tools:{profile}:{clean_host}"


@dataclass
class ResolvedCredentials:
    """Resolved CVAT connection details with redacted PAT representation."""

    profile: str
    host: str
    org: str
    pat: str = field(repr=False)
    source: str
    verified: bool = False
    connectivity: ConnectivityResult | None = None

    def __repr__(self) -> str:
        return (
            f"ResolvedCredentials(profile={self.profile!r}, host={self.host!r}, "
            f"org={self.org!r}, pat='[REDACTED]', source={self.source!r}, verified={self.verified!r})"
        )

    def to_gateway(self, write_enabled: bool = False) -> CvatGateway:
        return CvatGateway(
            host=self.host,
            pat=self.pat,
            org=self.org,
            write_enabled=write_enabled,
        )


def _default_prompt(prompt_text: str) -> str:
    return getpass.getpass(prompt_text)


def discover_credentials(
    profile: str = "local",
    *,
    host: str | None = None,
    org: str | None = None,
    pat: str | None = None,
    env_path: Path | None = None,
    store: CredentialStore | None = None,
    verify: bool = True,
    interactive: bool = False,
    store_on_prompt: bool = True,
    prompt_fn: Callable[[str], str] | None = None,
    gateway_factory: Callable[[str, str, str], CvatGateway] | None = None,
) -> ResolvedCredentials:
    """Discover CVAT credentials without reading browser cookies or writing plaintext to repo.

    Order of precedence:
    1. Explicit `pat` argument
    2. Process environment variables (`os.environ`)
    3. `.env` candidates via `cvat_tools.core.client.load_env` (read-only)
    4. Secure OS credential store (keyring / Windows Credential Manager)
    5. Interactive secure prompt (`getpass`) if `interactive=True`
    """
    prof = load_profile(profile) or {}
    pat_env_key = prof.get("pat_env") or "CVAT_PAT"

    dotenv_vals = load_env(path=env_path) if env_path else load_env()
    resolved_host = (
        host
        or os.environ.get("CVAT_HOST")
        or (dotenv_vals.get("CVAT_HOST") if not prof.get("pat_env") else None)
        or prof.get("host")
        or "http://localhost:8080"
    ).rstrip("/")
    resolved_org = (
        org
        if org is not None
        else (
            os.environ.get("CVAT_ORG")
            or (dotenv_vals.get("CVAT_ORG") if not prof.get("pat_env") else None)
            or prof.get("org")
            or ""
        )
    )

    cred_store = store or get_default_credential_store()
    target = credential_target(profile, resolved_host)

    resolved_pat = ""
    source = "missing"

    if pat:
        resolved_pat = pat.strip()
        source = "explicit"
    elif os.environ.get(pat_env_key):
        resolved_pat = os.environ[pat_env_key].strip()
        source = f"env:{pat_env_key}"
    elif dotenv_vals.get(pat_env_key):
        resolved_pat = str(dotenv_vals[pat_env_key]).strip()
        source = f"dotenv:{pat_env_key}"
    else:
        stored = cred_store.get(target, resolved_org or "cvat")
        if stored:
            resolved_pat = stored.strip()
            source = f"credstore:{cred_store.backend_name}"

    def _make_gw(token: str) -> CvatGateway:
        if gateway_factory is not None:
            return gateway_factory(resolved_host, token, resolved_org)
        return CvatGateway(resolved_host, token, org=resolved_org, write_enabled=False)

    connectivity: ConnectivityResult | None = None
    verified = False

    if resolved_pat and verify:
        connectivity = _make_gw(resolved_pat).test_connection()
        verified = connectivity.reachable and connectivity.authenticated

    can_prompt = interactive and (prompt_fn is not None or (sys.stdin is not None and sys.stdin.isatty()))
    if (not resolved_pat or (verify and connectivity and connectivity.reachable and not verified)) and can_prompt:
        asker = prompt_fn or _default_prompt
        entered = asker(f"Enter CVAT PAT for {resolved_host} (profile={profile}): ").strip()
        if entered:
            resolved_pat = entered
            source = "prompt"
            if verify:
                connectivity = _make_gw(resolved_pat).test_connection()
                verified = connectivity.reachable and connectivity.authenticated
            if store_on_prompt and (not verify or verified):
                cred_store.set(target, resolved_org or "cvat", resolved_pat)

    if not resolved_pat:
        raise RuntimeError(
            f"Missing CVAT PAT for profile={profile!r} ({resolved_host}). "
            f"Set {pat_env_key} in environment/.env or run interactive setup."
        )

    return ResolvedCredentials(
        profile=profile,
        host=resolved_host,
        org=resolved_org,
        pat=resolved_pat,
        source=source,
        verified=verified,
        connectivity=connectivity,
    )


def check_auth_status(
    host: str | None = None,
    profile: str = "local",
    *,
    verify: bool = True,
) -> dict:
    """Return a JSON-safe auth and permission summary (never includes raw PAT)."""
    try:
        creds = discover_credentials(profile=profile, host=host, verify=verify, interactive=False)
    except Exception as exc:
        return {"status": "missing_or_invalid", "profile": profile, "host": host, "error": str(exc)}

    out = {
        "status": "ok" if creds.verified else ("unverified" if not verify else "auth_failed"),
        "profile": creds.profile,
        "host": creds.host,
        "org": creds.org,
        "source": creds.source,
        "verified": creds.verified,
    }
    if creds.connectivity is not None:
        out["server_version"] = creds.connectivity.server_version
        out["username"] = creds.connectivity.username
        out["user_id"] = creds.connectivity.user_id
        if creds.connectivity.error:
            out["error"] = creds.connectivity.error
    return out


def run_setup(
    host: str | None = None,
    profile: str = "local",
    *,
    non_interactive: bool = False,
    force: bool = False,
    verify: bool = True,
) -> dict:
    """Run zero-friction credential discovery and optionally prompt/store PAT in OS Credential Manager."""
    if force and not non_interactive:
        # Force re-prompt by clearing explicit/stored token lookup via interactive prompt
        import getpass as _gp

        prof = load_profile(profile) or {}
        resolved_host = (host or prof.get("host") or "http://localhost:8080").rstrip("/")
        resolved_org = prof.get("org") or ""
        entered = _gp.getpass(f"Enter CVAT PAT for {resolved_host} (profile={profile}): ").strip()
        creds = discover_credentials(
            profile=profile,
            host=resolved_host,
            org=resolved_org,
            pat=entered or None,
            verify=verify,
            interactive=False,
        )
        if creds.pat and (not verify or creds.verified):
            get_default_credential_store().set(
                credential_target(profile, creds.host), creds.org or "cvat", creds.pat
            )
    else:
        creds = discover_credentials(
            profile=profile,
            host=host,
            verify=verify,
            interactive=not non_interactive,
            store_on_prompt=True,
        )

    return {
        "status": "ok" if (not verify or creds.verified) else "auth_failed",
        "profile": creds.profile,
        "host": creds.host,
        "org": creds.org,
        "source": creds.source,
        "verified": creds.verified,
        "username": creds.connectivity.username if creds.connectivity else None,
    }
