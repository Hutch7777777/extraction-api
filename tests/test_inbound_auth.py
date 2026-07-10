import time
import unittest

from app import app
from config import config
from utils.request_auth import RequestClaims, sha256_hex, sign_request


class InboundAuthenticationTests(unittest.TestCase):
    def setUp(self):
        self.original_require_signed = config.EXTRACTION_REQUIRE_SIGNED_REQUESTS
        self.original_signing_secret = config.EXTRACTION_API_SIGNING_SECRET
        self.original_api_key = config.EXTRACTION_API_KEY
        config.EXTRACTION_REQUIRE_SIGNED_REQUESTS = True
        config.EXTRACTION_API_SIGNING_SECRET = "integration-test-secret"
        config.EXTRACTION_API_KEY = None
        app.config.update(TESTING=True)
        self.client = app.test_client()

    def tearDown(self):
        config.EXTRACTION_REQUIRE_SIGNED_REQUESTS = self.original_require_signed
        config.EXTRACTION_API_SIGNING_SECRET = self.original_signing_secret
        config.EXTRACTION_API_KEY = self.original_api_key

    def signed_headers(self, body: bytes):
        timestamp = str(int(time.time()))
        claims = RequestClaims(
            user_id="user-123",
            organization_id="org-456",
            request_id="request-integration-789",
            timestamp=timestamp,
            body_sha256=sha256_hex(body),
        )
        return {
            "Content-Type": "application/json",
            "X-Estimate-Signature-Version": "v1",
            "X-Estimate-User-Id": claims.user_id,
            "X-Estimate-Organization-Id": claims.organization_id,
            "X-Estimate-Request-Id": claims.request_id,
            "X-Estimate-Timestamp": claims.timestamp,
            "X-Estimate-Body-SHA256": claims.body_sha256,
            "X-Estimate-Signature": "sha256=" + sign_request(
                "POST", "/process-job", claims, config.EXTRACTION_API_SIGNING_SECRET
            ),
        }

    def test_health_remains_public(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)

    def test_unsigned_request_is_rejected(self):
        response = self.client.post("/process-job", data=b"{}", content_type="application/json")
        self.assertEqual(response.status_code, 401)

    def test_valid_signature_reaches_route_and_preserves_request_id(self):
        body = b"{}"
        response = self.client.post(
            "/process-job",
            data=body,
            headers=self.signed_headers(body),
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.headers["X-Request-Id"], "request-integration-789")
        self.assertEqual(response.get_json()["error"], "job_id required")


if __name__ == "__main__":
    unittest.main()
