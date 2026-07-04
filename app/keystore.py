"""Secure storage for the optional cloud API keys (OS credential vault).

Each provider (Claude, Gemini) keeps its key in the operating-system credential
store via ``keyring`` (Windows Credential Manager on Windows) — never in the
plaintext settings JSON. A provider's environment variable, if set, *overrides*
its stored key so power users / CI can inject a key without touching the vault.

Every function degrades gracefully: if ``keyring`` (or its OS backend) is
unavailable, the stored-key helpers behave as if no key were stored and the app
falls back to the env var or the offline template — consistent with the app's
offline-first design.
"""

from __future__ import annotations

import os

from . import config


def _backend():
    """Return the ``keyring`` module if a usable backend exists, else ``None``."""
    try:
        import keyring
    except Exception:  # SDK missing / import error
        return None
    try:
        active = keyring.get_keyring()
    except Exception:
        return None
    # ``keyring.backends.fail.Keyring`` is the null backend used when no OS
    # credential store is reachable; it raises on every get/set.
    if type(active).__module__.endswith("backends.fail"):
        return None
    return keyring


def keyring_available() -> bool:
    """True when a usable OS credential backend is present."""
    return _backend() is not None


def get_stored_key(provider_id: str) -> str:
    """The key saved in the OS vault for ``provider_id``, or ``""``."""
    kr = _backend()
    if kr is None:
        return ""
    provider = config.get_provider(provider_id)
    try:
        value = kr.get_password(
            provider.keyring_service, config.LLM_KEYRING_USERNAME
        )
    except Exception:
        return ""
    return (value or "").strip()


def set_stored_key(provider_id: str, key: str) -> None:
    """Persist ``key`` for ``provider_id``. Raises if storage is unavailable."""
    kr = _backend()
    if kr is None:
        raise RuntimeError(
            "Stocarea securizată (keyring) nu este disponibilă pe acest sistem."
        )
    provider = config.get_provider(provider_id)
    kr.set_password(
        provider.keyring_service, config.LLM_KEYRING_USERNAME, key.strip()
    )


def clear_stored_key(provider_id: str) -> None:
    """Remove the stored key for ``provider_id``. No-op if none / no backend."""
    kr = _backend()
    if kr is None:
        return
    provider = config.get_provider(provider_id)
    try:
        kr.delete_password(
            provider.keyring_service, config.LLM_KEYRING_USERNAME
        )
    except Exception:
        # Nothing stored, or the backend refused — treat as already cleared.
        pass


def env_key(provider_id: str) -> str:
    """The key from ``provider_id``'s environment variable, or ``""``."""
    provider = config.get_provider(provider_id)
    return os.environ.get(provider.env_var, "").strip()


def resolve_api_key(provider_id: str) -> str:
    """Effective API key for ``provider_id``: env var overrides the stored key."""
    return env_key(provider_id) or get_stored_key(provider_id)


def key_source(provider_id: str) -> str | None:
    """Origin of the effective key: ``"env"``, ``"stored"``, or ``None``."""
    if env_key(provider_id):
        return "env"
    if get_stored_key(provider_id):
        return "stored"
    return None
