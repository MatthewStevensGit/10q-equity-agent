"""
Import this before making any HTTPS request in this project.

Some networks (school/corporate) do TLS interception, which breaks Python's
bundled CA verification. `truststore` routes verification through the OS
trust store instead, which has the interception CA installed -- the correct
fix, not verify=False. Falls back to default verification if truststore
isn't installed or injection fails, so this is a no-op on a normal network.
"""

from __future__ import annotations

try:
    import truststore

    truststore.inject_into_ssl()
    TRUSTSTORE_ACTIVE = True
except Exception:  # noqa: BLE001
    TRUSTSTORE_ACTIVE = False
