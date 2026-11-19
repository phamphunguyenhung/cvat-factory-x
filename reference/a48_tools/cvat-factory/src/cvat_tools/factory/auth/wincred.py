"""Secure credential storage backends (keyring, Windows Credential Manager via ctypes, and in-memory for tests)."""
from __future__ import annotations

import ctypes
import sys
from typing import Protocol


class CredentialStore(Protocol):
    """Protocol for storing and retrieving PATs without plaintext files."""

    backend_name: str

    def get(self, target: str, username: str = "cvat") -> str | None: ...
    def set(self, target: str, username: str, secret: str) -> None: ...
    def delete(self, target: str, username: str = "cvat") -> bool: ...


class MemoryCredentialStore:
    """In-memory credential store for isolated unit tests."""

    backend_name = "memory"

    def __init__(self) -> None:
        self._data: dict[tuple[str, str], str] = {}

    def get(self, target: str, username: str = "cvat") -> str | None:
        return self._data.get((target, username))

    def set(self, target: str, username: str, secret: str) -> None:
        self._data[(target, username)] = secret

    def delete(self, target: str, username: str = "cvat") -> bool:
        return self._data.pop((target, username), None) is not None


class KeyringCredentialStore:
    """Credential store backed by the optional `keyring` library."""

    backend_name = "keyring"

    def __init__(self) -> None:
        import keyring  # lazy import

        self._kr = keyring

    def get(self, target: str, username: str = "cvat") -> str | None:
        return self._kr.get_password(target, username)

    def set(self, target: str, username: str, secret: str) -> None:
        self._kr.set_password(target, username, secret)

    def delete(self, target: str, username: str = "cvat") -> bool:
        try:
            self._kr.delete_password(target, username)
            return True
        except Exception:
            return False


class Win32CredentialStore:
    """Native Windows Credential Manager backend using advapi32.dll via ctypes."""

    backend_name = "win32cred"
    _CRED_TYPE_GENERIC = 1
    _CRED_PERSIST_LOCAL_MACHINE = 2

    def __init__(self) -> None:
        if sys.platform != "win32":
            raise RuntimeError("Win32CredentialStore is only supported on Windows.")
        from ctypes import wintypes

        class _CREDENTIALW(ctypes.Structure):
            _fields_ = [
                ("Flags", wintypes.DWORD),
                ("Type", wintypes.DWORD),
                ("TargetName", wintypes.LPWSTR),
                ("Comment", wintypes.LPWSTR),
                ("LastWritten", wintypes.FILETIME),
                ("CredentialBlobSize", wintypes.DWORD),
                ("CredentialBlob", ctypes.POINTER(ctypes.c_ubyte)),
                ("Persist", wintypes.DWORD),
                ("AttributeCount", wintypes.DWORD),
                ("Attributes", ctypes.c_void_p),
                ("TargetAlias", wintypes.LPWSTR),
                ("UserName", wintypes.LPWSTR),
            ]

        self._CREDENTIALW = _CREDENTIALW
        self._PCREDENTIALW = ctypes.POINTER(_CREDENTIALW)
        advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)

        advapi32.CredReadW.argtypes = [
            wintypes.LPCWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
            ctypes.POINTER(self._PCREDENTIALW),
        ]
        advapi32.CredReadW.restype = wintypes.BOOL

        advapi32.CredWriteW.argtypes = [self._PCREDENTIALW, wintypes.DWORD]
        advapi32.CredWriteW.restype = wintypes.BOOL

        advapi32.CredDeleteW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD]
        advapi32.CredDeleteW.restype = wintypes.BOOL

        advapi32.CredFree.argtypes = [ctypes.c_void_p]
        advapi32.CredFree.restype = None
        self._advapi32 = advapi32

    def get(self, target: str, username: str = "cvat") -> str | None:
        cred_ptr = self._PCREDENTIALW()
        ok = self._advapi32.CredReadW(target, self._CRED_TYPE_GENERIC, 0, ctypes.byref(cred_ptr))
        if not ok:
            return None
        try:
            cred = cred_ptr.contents
            size = int(cred.CredentialBlobSize)
            if size <= 0 or not cred.CredentialBlob:
                return ""
            raw = bytes(cred.CredentialBlob[:size])
            if len(raw) % 2 == 0:
                try:
                    return raw.decode("utf-16-le").rstrip("\x00")
                except UnicodeDecodeError:
                    pass
            return raw.decode("utf-8", errors="replace").rstrip("\x00")
        finally:
            self._advapi32.CredFree(cred_ptr)

    def set(self, target: str, username: str, secret: str) -> None:
        blob = secret.encode("utf-16-le")
        buf = (ctypes.c_ubyte * len(blob)).from_buffer_copy(blob)
        cred = self._CREDENTIALW()
        cred.Flags = 0
        cred.Type = self._CRED_TYPE_GENERIC
        cred.TargetName = target
        cred.Comment = "CVAT Factory PAT"
        cred.CredentialBlobSize = len(blob)
        cred.CredentialBlob = ctypes.cast(buf, ctypes.POINTER(ctypes.c_ubyte))
        cred.Persist = self._CRED_PERSIST_LOCAL_MACHINE
        cred.AttributeCount = 0
        cred.Attributes = None
        cred.TargetAlias = None
        cred.UserName = username
        ok = self._advapi32.CredWriteW(ctypes.byref(cred), 0)
        if not ok:
            err = ctypes.get_last_error()
            raise OSError(err, f"CredWriteW failed with Win32 error {err}")

    def delete(self, target: str, username: str = "cvat") -> bool:
        ok = self._advapi32.CredDeleteW(target, self._CRED_TYPE_GENERIC, 0)
        return bool(ok)


def get_default_credential_store() -> CredentialStore:
    """Return keyring store if available, else Win32 Credential Manager on Windows, else in-memory."""
    try:
        import keyring  # noqa: F401

        return KeyringCredentialStore()
    except Exception:
        pass
    if sys.platform == "win32":
        return Win32CredentialStore()
    return MemoryCredentialStore()
