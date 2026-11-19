"""Authentication, credential discovery, and permission detection for cvat_tools.factory."""
from __future__ import annotations

from cvat_tools.factory.auth.credentials import (
    ResolvedCredentials,
    check_auth_status,
    credential_target,
    discover_credentials,
    run_setup,
)
from cvat_tools.factory.auth.permissions import (
    OrgMembership,
    PermissionReport,
    UserIdentity,
    detect_permissions,
)
from cvat_tools.factory.auth.wincred import (
    CredentialStore,
    KeyringCredentialStore,
    MemoryCredentialStore,
    Win32CredentialStore,
    get_default_credential_store,
)

__all__ = [
    "CredentialStore",
    "KeyringCredentialStore",
    "MemoryCredentialStore",
    "OrgMembership",
    "PermissionReport",
    "ResolvedCredentials",
    "UserIdentity",
    "Win32CredentialStore",
    "check_auth_status",
    "credential_target",
    "detect_permissions",
    "discover_credentials",
    "get_default_credential_store",
    "run_setup",
]
