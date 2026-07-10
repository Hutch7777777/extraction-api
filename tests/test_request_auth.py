import unittest

from utils.request_auth import (
    RequestClaims,
    canonicalize_request,
    sha256_hex,
    sign_request,
    verify_signed_request,
)


class SignedRequestTests(unittest.TestCase):
    def setUp(self):
        self.secret = "contract-test-secret"
        self.body = b'{"job_id":"job-123"}'
        self.claims = RequestClaims(
            user_id="user-123",
            organization_id="org-456",
            request_id="request-789",
            timestamp="1700000000",
            body_sha256=sha256_hex(self.body),
        )

    def headers(self):
        return {
            "X-Estimate-Signature-Version": "v1",
            "X-Estimate-User-Id": self.claims.user_id,
            "X-Estimate-Organization-Id": self.claims.organization_id,
            "X-Estimate-Request-Id": self.claims.request_id,
            "X-Estimate-Timestamp": self.claims.timestamp,
            "X-Estimate-Body-SHA256": self.claims.body_sha256,
            "X-Estimate-Signature": "sha256=" + sign_request(
                "POST", "/process-job", self.claims, self.secret
            ),
        }

    def test_contract_canonicalization(self):
        self.assertEqual(
            canonicalize_request("POST", "/process-job", self.claims),
            "\n".join([
                "v1",
                "POST",
                "/process-job",
                "user-123",
                "org-456",
                "request-789",
                "1700000000",
                self.claims.body_sha256,
            ]),
        )
        self.assertEqual(
            sign_request("POST", "/process-job", self.claims, self.secret),
            "26cd902a1cd352c940fbc4f622a79f4db89e4fdbe405578ef3d8aa7280905737",
        )

    def test_accepts_valid_request(self):
        valid, reason, claims = verify_signed_request(
            method="POST",
            path="/process-job",
            body=self.body,
            headers=self.headers(),
            secret=self.secret,
            now=1700000000,
        )
        self.assertTrue(valid, reason)
        self.assertEqual(claims, self.claims)

    def test_rejects_body_tampering(self):
        valid, reason, _ = verify_signed_request(
            method="POST",
            path="/process-job",
            body=b'{"job_id":"another-job"}',
            headers=self.headers(),
            secret=self.secret,
            now=1700000000,
        )
        self.assertFalse(valid)
        self.assertEqual(reason, "request body hash mismatch")

    def test_rejects_expired_request(self):
        valid, reason, _ = verify_signed_request(
            method="POST",
            path="/process-job",
            body=self.body,
            headers=self.headers(),
            secret=self.secret,
            now=1700001000,
            max_age_seconds=300,
        )
        self.assertFalse(valid)
        self.assertEqual(reason, "request timestamp outside allowed window")


if __name__ == "__main__":
    unittest.main()
