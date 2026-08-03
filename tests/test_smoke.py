import os
import shutil
import unittest
from unittest.mock import patch


ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
TMP_DIR = os.path.join(os.path.dirname(__file__), "_tmp_runtime")
TMP_STATIC_DIR = os.path.join(TMP_DIR, "static")
TMP_KNOWLEDGE_BASE_PATH = os.path.join(TMP_DIR, "knowledge_base.json")
shutil.rmtree(TMP_DIR, ignore_errors=True)
os.makedirs(TMP_DIR, exist_ok=True)
shutil.copytree(os.path.join(ROOT_DIR, "static"), TMP_STATIC_DIR)
shutil.copy2(os.path.join(ROOT_DIR, "src", "data", "knowledge_base.json"), TMP_KNOWLEDGE_BASE_PATH)
os.environ["APP_DATA_DIR"] = TMP_DIR
os.environ["APP_DB_PATH"] = os.path.join(TMP_DIR, "test.db")
os.environ["APP_STATIC_DIR"] = TMP_STATIC_DIR
os.environ["APP_KNOWLEDGE_PATH"] = TMP_KNOWLEDGE_BASE_PATH
os.environ["ADMIN_USERNAME"] = "admin"
os.environ["ADMIN_PASSWORD"] = "testpass123"
os.environ["SECRET_KEY"] = "test-secret-key"

from app import app  # noqa: E402
from src.retroworld_ia import config as app_config  # noqa: E402
from src.retroworld_ia.services import ai as ai_service  # noqa: E402
from src.retroworld_ia.services.ai import (  # noqa: E402
    build_openai_messages,
    enforce_grounded_price_claims,
    enforce_no_live_availability_claims,
    responses_answer,
)
from src.retroworld_ia.services.corrections import find_relevant_corrections  # noqa: E402
from src.retroworld_ia.services.knowledge import build_system_prompt  # noqa: E402

FAQ_RETROWORLD_PATH = os.path.join(
    TMP_STATIC_DIR,
    "faq_retroworld.json",
)
FAQ_RUNNINGMAN_PATH = os.path.join(
    TMP_STATIC_DIR,
    "faq_runningman.json",
)
KNOWLEDGE_BASE_PATH = TMP_KNOWLEDGE_BASE_PATH


class SmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(FAQ_RETROWORLD_PATH, "r", encoding="utf-8") as handle:
            cls.original_retroworld_faq = handle.read()
        with open(FAQ_RUNNINGMAN_PATH, "r", encoding="utf-8") as handle:
            cls.original_runningman_faq = handle.read()
        with open(KNOWLEDGE_BASE_PATH, "r", encoding="utf-8") as handle:
            cls.original_knowledge_base = handle.read()

    @classmethod
    def tearDownClass(cls):
        with open(FAQ_RETROWORLD_PATH, "w", encoding="utf-8") as handle:
            handle.write(cls.original_retroworld_faq)
        with open(FAQ_RUNNINGMAN_PATH, "w", encoding="utf-8") as handle:
            handle.write(cls.original_runningman_faq)
        with open(KNOWLEDGE_BASE_PATH, "w", encoding="utf-8") as handle:
            handle.write(cls.original_knowledge_base)
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
        faq_brand = self.client.get("/faq/retroworld")
        self.assertEqual(faq_brand.status_code, 200)
        self.assertTrue(faq_brand.is_json)
        self.assertEqual(self.client.get("/faq/runningman").status_code, 200)
        self.assertEqual(self.client.get("/faq/runningman/").status_code, 200)
        self.assertEqual(self.client.get("/robots.txt").status_code, 200)

    def test_runningman_faq_is_useful(self):
        response = self.client.get("/faq.json?brand_id=runningman")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        items = payload.get("items", [])
        self.assertGreaterEqual(len(items), 6)
        questions = " ".join(item.get("question", "") for item in items)
        answers = " ".join(item.get("answer", "") for item in items)
        self.assertTrue("Runningman" in questions or "Running Man" in questions)
        self.assertIn("04 98 09 30 59", answers)

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
        diag_payload = diag.get_json()
        self.assertTrue(diag_payload["ok"])
        self.assertTrue(diag_payload["openai_model"])
        self.assertIn(diag_payload["openai_api_mode"], {"responses", "chat_completions"})
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

    def test_admin_correction_save_requires_csrf(self):
        self.login()
        response = self.client.post(
            "/admin/api/corrections",
            json={"brand_id": "retroworld", "trigger_text": "Age minimum", "corrected_answer": "A partir de 7 ans."},
        )
        self.assertEqual(response.status_code, 403)

    def test_correction_memory_is_reused_in_prompt(self):
        csrf_token = self.login()
        response = self.client.post(
            "/admin/api/corrections",
            json={
                "brand_id": "retroworld",
                "trigger_text": "age minimum enfant activite",
                "corrected_answer": "Retroworld accueille les enfants a partir de 7 ans pour les activites adaptees.",
                "priority": 80,
            },
            headers={"X-CSRF-Token": csrf_token},
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.get_json()["ok"])

        corrections = find_relevant_corrections("retroworld", "A partir de quel age peut venir mon enfant ?", limit=3)
        self.assertGreaterEqual(len(corrections), 1)
        prompt = build_system_prompt("retroworld", "A partir de quel age peut venir mon enfant ?", corrections=corrections)
        self.assertIn("Corrections approuvees", prompt)
        self.assertIn("Retroworld accueille les enfants", prompt)

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

    def test_system_prompt_has_strict_grounding_contract(self):
        prompt = build_system_prompt(
            "enigmaniac",
            "Quel est le tarif exact pour 5 joueurs demain ?",
        )
        self.assertIn("Une information absente est inconnue", prompt)
        self.assertIn("Ne calcule et n'interpole jamais un prix exact", prompt)
        self.assertIn("pas leur disponibilite a une date donnee", prompt)

    def test_price_guard_blocks_interpolated_amount(self):
        user_text = "Quel est le tarif exact pour 5 joueurs ?"
        answer, changed = enforce_grounded_price_claims(
            "Le tarif est de 20 € par personne.",
            "Le tarif connu est de 15 EUR à 25 EUR selon la formule.",
            user_text,
        )
        self.assertTrue(changed)
        self.assertNotIn("20 €", answer)
        self.assertIn("à confirmer", answer)

    def test_price_guard_keeps_known_amount(self):
        user_text = "Combien coûte la VR ?"
        answer, changed = enforce_grounded_price_claims(
            "Les jeux VR arcade coûtent 15 EUR par joueur.",
            "Jeux VR arcade: 15 EUR par joueur.",
            user_text,
        )
        self.assertFalse(changed)
        self.assertIn("15 EUR", answer)

    def test_live_availability_guard_relabels_known_inventory(self):
        answer, changed = enforce_no_live_availability_claims(
            "Les salles disponibles sont : La Loi de la Jungle et Terreur Nocturne.",
            "Quelles salles sont disponibles demain ?",
        )
        self.assertTrue(changed)
        self.assertIn("figurant dans mes informations", answer)
        self.assertNotIn(": :", answer)

    def test_responses_api_payload_and_output_parsing(self):
        captured = {}

        class FakeResponse:
            status_code = 200
            text = ""

            def raise_for_status(self):
                return None

            def json(self):
                return {
                    "output": [
                        {
                            "type": "message",
                            "content": [{"type": "output_text", "text": "Réponse test"}],
                        }
                    ]
                }

        class FakeRequests:
            @staticmethod
            def post(url, headers, json, timeout):
                captured.update({"url": url, "headers": headers, "json": json, "timeout": timeout})
                return FakeResponse()

        messages = build_openai_messages("Instruction test", [], "Question test")
        with patch.object(ai_service, "requests", FakeRequests()), patch.object(app_config, "OPENAI_MODEL", "gpt-test"), patch.object(app_config, "OPENAI_REASONING_EFFORT", "low"), patch.object(app_config, "OPENAI_TEXT_VERBOSITY", "low"):
            answer = responses_answer(messages, safety_identifier="rw_test")

        self.assertEqual(answer, "Réponse test")
        self.assertEqual(captured["url"], "https://api.openai.com/v1/responses")
        self.assertEqual(captured["json"]["instructions"], "Instruction test")
        self.assertEqual(captured["json"]["input"][0]["role"], "user")
        self.assertFalse(captured["json"]["store"])
        self.assertEqual(captured["json"]["reasoning"]["effort"], "low")
        self.assertEqual(captured["json"]["text"]["verbosity"], "low")
        self.assertEqual(captured["json"]["safety_identifier"], "rw_test")

    def test_chat_rejects_overlong_message(self):
        response = self.client.post(
            "/chat",
            json={"message": "x" * (app_config.CHAT_MAX_MESSAGE_CHARS + 1)},
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["error"], "message_too_long")

    def test_widget_scopes_conversation_by_brand(self):
        widget_path = os.path.join(TMP_STATIC_DIR, "chat-widget.html")
        with open(widget_path, "r", encoding="utf-8") as handle:
            widget = handle.read()
        self.assertIn("'rw_conversation_id_' + safeBrand", widget)
        self.assertIn('id="rw-new-conversation"', widget)

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
