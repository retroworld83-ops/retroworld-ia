import os
import shutil
import unittest


TMP_DIR = os.path.join(os.path.dirname(__file__), "_tmp_runtime")
shutil.rmtree(TMP_DIR, ignore_errors=True)
os.makedirs(TMP_DIR, exist_ok=True)
os.environ["APP_DATA_DIR"] = TMP_DIR
os.environ["APP_DB_PATH"] = os.path.join(TMP_DIR, "test.db")
os.environ["ADMIN_USERNAME"] = "admin"
os.environ["ADMIN_PASSWORD"] = "testpass123"
os.environ["SECRET_KEY"] = "test-secret-key"

from app import app  # noqa: E402
from src.retroworld_ia.services.ai import build_openai_messages  # noqa: E402

FAQ_RETROWORLD_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "static",
    "faq_retroworld.json",
)


class SmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(FAQ_RETROWORLD_PATH, "r", encoding="utf-8") as handle:
            cls.original_retroworld_faq = handle.read()

    @classmethod
    def tearDownClass(cls):
        with open(FAQ_RETROWORLD_PATH, "w", encoding="utf-8") as handle:
            handle.write(cls.original_retroworld_faq)
        shutil.rmtree(TMP_DIR, ignore_errors=True)

    def setUp(self):
        self.client = app.test_client()

    def login(self):
        response = self.client.post(
            "/admin/api/auth/login",
            json={"username": "admin", "password": "testpass123"},
        )
        self.assertEqual(response.status_code, 200)
        return response.get_json()["csrf_token"]

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
        response = self.client.post("/chat", json={"message": "bonjour"})
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload.get("ok"))
        self.assertIn("answer", payload)

    def test_chat_handles_non_object_json_payload(self):
        response = self.client.post("/chat", json=["bonjour"])
        self.assertEqual(response.status_code, 400)
        payload = response.get_json()
        self.assertFalse(payload.get("ok"))
        self.assertIn("error", payload)

    def test_chat_brand_alias_routes(self):
        self.assertEqual(self.client.options("/chat/retroworld").status_code, 204)
        self.assertEqual(self.client.options("/chat/retroworld/").status_code, 204)

        response = self.client.post("/chat/retroworld", json={"message": "bonjour"})
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload.get("ok"))
        self.assertEqual(payload.get("brand_id"), "retroworld")

    def test_not_found_chat_prefix_returns_json(self):
        response = self.client.get("/chat/unknown/path")
        self.assertEqual(response.status_code, 404)
        payload = response.get_json()
        self.assertFalse(payload.get("ok"))
        self.assertEqual(payload.get("error"), "not_found")

    def test_method_not_allowed_chat_prefix_returns_json(self):
        response = self.client.get("/chat/retroworld")
        self.assertEqual(response.status_code, 405)
        payload = response.get_json()
        self.assertFalse(payload.get("ok"))
        self.assertEqual(payload.get("error"), "method_not_allowed")

    def test_admin_requires_login(self):
        self.assertEqual(self.client.get("/admin/api/diag").status_code, 401)
        self.assertEqual(self.client.get("/admin").status_code, 302)

    def test_admin_login_session_and_diag(self):
        csrf_token = self.login()
        session_response = self.client.get("/admin/api/session")
        self.assertEqual(session_response.status_code, 200)
        self.assertTrue(session_response.get_json()["authenticated"])
        self.assertEqual(session_response.get_json()["username"], "admin")
        self.assertEqual(session_response.get_json()["csrf_token"], csrf_token)

        diag = self.client.get("/admin/api/diag")
        self.assertEqual(diag.status_code, 200)
        self.assertTrue(diag.get_json()["ok"])
        analytics = self.client.get("/admin/api/analytics")
        self.assertEqual(analytics.status_code, 200)
        self.assertTrue(analytics.get_json()["ok"])

    def test_knowledge_endpoints(self):
        csrf_token = self.login()
        get_response = self.client.get("/admin/api/knowledge/retroworld")
        self.assertEqual(get_response.status_code, 200)
        payload = get_response.get_json()
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["brand"]["id"], "retroworld")

        update_response = self.client.put(
            "/admin/api/knowledge/retroworld",
            json={"name": "Retroworld", "short": "Retroworld", "knowledge_cards": [{"title": "Test", "summary": "Carte"}]},
            headers={"X-CSRF-Token": csrf_token},
        )
        self.assertEqual(update_response.status_code, 200)
        self.assertTrue(update_response.get_json()["ok"])

    def test_admin_user_management(self):
        csrf_token = self.login()
        create_response = self.client.post(
            "/admin/api/users",
            json={"username": "editor", "password": "editor-pass"},
            headers={"X-CSRF-Token": csrf_token},
        )
        self.assertEqual(create_response.status_code, 200)
        self.assertTrue(create_response.get_json()["ok"])

        users_response = self.client.get("/admin/api/users")
        self.assertEqual(users_response.status_code, 200)
        users = users_response.get_json()["items"]
        editor = next(user for user in users if user["username"] == "editor")

        disable_response = self.client.put(
            f"/admin/api/users/{editor['id']}",
            json={"is_active": 0},
            headers={"X-CSRF-Token": csrf_token},
        )
        self.assertEqual(disable_response.status_code, 200)
        self.assertTrue(disable_response.get_json()["ok"])

    def test_admin_faq_save_requires_csrf(self):
        self.login()
        response = self.client.post("/admin/api/faq/save", json={"brand_id": "retroworld", "items": []})
        self.assertEqual(response.status_code, 403)

    def test_admin_faq_save_with_csrf(self):
        csrf_token = self.login()
        response = self.client.post(
            "/admin/api/faq/save",
            json={"brand_id": "retroworld", "items": [{"question": "Test ?", "answer": "Oui", "tags": ["test"]}]},
            headers={"X-CSRF-Token": csrf_token},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["count"], 1)

    def test_build_openai_messages_includes_history(self):
        history = [
            {"role": "user", "content": "Bonjour"},
            {"role": "assistant", "content": "Bonjour, que puis-je faire ?"},
        ]
        messages = build_openai_messages("system", history, "Je veux reserver")
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["content"][0]["text"], "Bonjour")
        self.assertEqual(messages[2]["content"][0]["text"], "Bonjour, que puis-je faire ?")
        self.assertEqual(messages[3]["content"][0]["text"], "Je veux reserver")

    def test_chat_persists_conversation_and_reuses_id(self):
        first = self.client.post("/chat", json={"message": "bonjour", "brand_id": "retroworld"})
        self.assertEqual(first.status_code, 200)
        conv_id = first.get_json()["conversation_id"]

        second = self.client.post("/chat", json={"message": "encore", "conversation_id": conv_id, "brand_id": "retroworld"})
        self.assertEqual(second.status_code, 200)

        csrf_token = self.login()
        detail = self.client.get("/admin/api/conversation/" + conv_id)
        self.assertEqual(detail.status_code, 200)
        messages = detail.get_json()["conversation"]["messages"]
        self.assertGreaterEqual(len(messages), 4)
        self.assertTrue(csrf_token)


if __name__ == "__main__":
    unittest.main()
