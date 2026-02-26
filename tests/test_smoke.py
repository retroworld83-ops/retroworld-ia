import os
import unittest

# Ensure admin routes are testable
os.environ.setdefault("ADMIN_API_TOKEN", "testtoken")
os.environ.setdefault("ADMIN_DASHBOARD_TOKEN", "testtoken")

from app import app  # noqa: E402


class SmokeTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_public_routes(self):
        self.assertEqual(self.client.get("/health").status_code, 200)
        self.assertEqual(self.client.get("/brands.json").status_code, 200)
        self.assertEqual(self.client.get("/faq.json?brand_id=retroworld").status_code, 200)

    def test_chat_without_openai_key_returns_graceful_message(self):
        r = self.client.post("/chat", json={"message": "bonjour"})
        self.assertEqual(r.status_code, 200)
        payload = r.get_json()
        self.assertTrue(payload.get("ok"))
        self.assertIn("answer", payload)

    def test_admin_requires_token(self):
        self.assertEqual(self.client.get("/admin/api/diag").status_code, 401)

    def test_admin_with_token(self):
        r = self.client.get("/admin/api/diag?token=testtoken")
        self.assertEqual(r.status_code, 200)
        payload = r.get_json()
        self.assertTrue(payload.get("ok"))
        self.assertIn("recent_logs", payload)


if __name__ == "__main__":
    unittest.main()
