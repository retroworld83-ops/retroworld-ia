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
        self.assertEqual(self.client.get("/faq_enigmaniac.json").status_code, 200)
        self.assertEqual(self.client.get("/faq/retroworld").status_code, 302)
        self.assertEqual(self.client.get("/faq/runningman").status_code, 302)
        self.assertEqual(self.client.get("/faq/runningman/").status_code, 302)
        self.assertEqual(self.client.get("/robots.txt").status_code, 200)

    def test_chat_without_openai_key_returns_graceful_message(self):
        r = self.client.post("/chat", json={"message": "bonjour"})
        self.assertEqual(r.status_code, 200)
        payload = r.get_json()
        self.assertTrue(payload.get("ok"))
        self.assertIn("answer", payload)

    def test_chat_handles_non_object_json_payload(self):
        r = self.client.post("/chat", json=["bonjour"])
        self.assertEqual(r.status_code, 400)
        payload = r.get_json()
        self.assertFalse(payload.get("ok"))
        self.assertIn("error", payload)

    def test_chat_brand_alias_routes(self):
        opt = self.client.options("/chat/retroworld")
        self.assertEqual(opt.status_code, 204)
        opt_trailing = self.client.options("/chat/retroworld/")
        self.assertEqual(opt_trailing.status_code, 204)

        post = self.client.post("/chat/retroworld", json={"message": "bonjour"})
        self.assertEqual(post.status_code, 200)
        payload = post.get_json()
        self.assertTrue(payload.get("ok"))
        self.assertEqual(payload.get("brand_id"), "retroworld")

    def test_not_found_chat_prefix_returns_json(self):
        r = self.client.get("/chat/unknown/path")
        self.assertEqual(r.status_code, 404)
        payload = r.get_json()
        self.assertFalse(payload.get("ok"))
        self.assertEqual(payload.get("error"), "not_found")

    def test_method_not_allowed_chat_prefix_returns_json(self):
        r = self.client.get("/chat/retroworld")
        self.assertEqual(r.status_code, 405)
        payload = r.get_json()
        self.assertFalse(payload.get("ok"))
        self.assertEqual(payload.get("error"), "method_not_allowed")

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
