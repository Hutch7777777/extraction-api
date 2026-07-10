"""Verification for signed requests from the Estimate.ai web application."""

from dataclasses import dataclass
import hashlib
import hmac
import time
from typing import Mapping, Optional, Tuple


SIGNATURE_VERSION = "v1"


@dataclass(frozen=True)
class RequestClaims:
    user_id: str
    organization_id: str
    request_id: str
    timestamp: str
    body_sha256: str


def sha256_hex(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def canonicalize_request(
    method: str,
    path: str,
    claims: RequestClaims,
) -> str:
    normalized_path = path if path.startswith("/") else f"/{path}"
    return "\n".join([
        SIGNATURE_VERSION,
        method.upper(),
        normalized_path,
        claims.user_id,
        claims.organization_id,
        claims.request_id,
        claims.timestamp,
        claims.body_sha256,
    ])


def sign_request(method: str, path: str, claims: RequestClaims, secret: str) -> str:
    return hmac.new(
        secret.encode("utf-8"),
        canonicalize_request(method, path, claims).encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def verify_signed_request(
    *,
    method: str,
    path: str,
    body: bytes,
    headers: Mapping[str, str],
    secret: str,
    now: Optional[int] = None,
    max_age_seconds: int = 300,
) -> Tuple[bool, str, Optional[RequestClaims]]:
    version = headers.get("X-Estimate-Signature-Version", "")
    user_id = headers.get("X-Estimate-User-Id", "")
    organization_id = headers.get("X-Estimate-Organization-Id", "")
    request_id = headers.get("X-Estimate-Request-Id", "")
    timestamp = headers.get("X-Estimate-Timestamp", "")
    provided_body_hash = headers.get("X-Estimate-Body-SHA256", "")
    provided_signature = headers.get("X-Estimate-Signature", "")

    if version != SIGNATURE_VERSION:
        return False, "unsupported signature version", None
    if not all([user_id, organization_id, request_id, timestamp, provided_body_hash, provided_signature]):
        return False, "missing signed request headers", None

    try:
        timestamp_seconds = int(timestamp)
    except ValueError:
        return False, "invalid timestamp", None

    current_time = int(time.time()) if now is None else now
    if abs(current_time - timestamp_seconds) > max_age_seconds:
        return False, "request timestamp outside allowed window", None

    actual_body_hash = sha256_hex(body)
    if not hmac.compare_digest(provided_body_hash, actual_body_hash):
        return False, "request body hash mismatch", None

    claims = RequestClaims(
        user_id=user_id,
        organization_id=organization_id,
        request_id=request_id,
        timestamp=timestamp,
        body_sha256=provided_body_hash,
    )
    expected_signature = sign_request(method, path, claims, secret)
    normalized_signature = provided_signature.removeprefix("sha256=")
    if not hmac.compare_digest(normalized_signature, expected_signature):
        return False, "signature mismatch", None

    return True, "", claims
