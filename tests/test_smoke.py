import os
import unittest
from unittest import mock
import tempfile
import shutil

# Ensure admin routes are testable
os.environ.setdefault("ADMIN_API_TOKEN", "testtoken")
os.environ.setdefault("ADMIN_DASHBOARD_TOKEN", "testtoken")

import app as app_module  # noqa: E402
from app import app  # noqa: E402


class SmokeTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        self._tmp_static = tempfile.mkdtemp(prefix="faqgen_")
        self._old_static = app_module.STATIC_DIR
        app_module.STATIC_DIR = app_module.Path(self._tmp_static)
        app_module.STATIC_DIR.mkdir(parents=True, exist_ok=True)
        for bid in app_module.BRANDS:
            src = app_module.BASE_DIR / "static" / f"faq_{bid}.json"
            if src.exists():
                (app_module.STATIC_DIR / f"faq_{bid}.json").write_text(src.read_text("utf-8"), "utf-8")

    def tearDown(self):
        app_module.STATIC_DIR = self._old_static
        shutil.rmtree(self._tmp_static, ignore_errors=True)

    def test_public_routes(self):
        self.assertEqual(self.client.get("/health").status_code, 200)
        self.assertEqual(self.client.get("/brands.json").status_code, 200)
        self.assertEqual(self.client.get("/faq.json?brand_id=retroworld").status_code, 200)
        self.assertEqual(self.client.get("/faq_enigmaniac.json").status_code, 200)
        self.assertEqual(self.client.get("/faq/retroworld").status_code, 302)
        self.assertEqual(self.client.get("/faq/runningman").status_code, 302)
        self.assertEqual(self.client.get("/faq/runningman/").status_code, 302)
        self.assertEqual(self.client.get("/faq/retroworld.json").status_code, 200)
        self.assertEqual(self.client.get("/faq/retroworld", headers={"Accept": "application/json"}).status_code, 200)
        self.assertEqual(self.client.get("/robots.txt").status_code, 200)

    def test_cors_allows_enigmaniac_domain(self):
        r = self.client.get(
            "/faq/retroworld.json",
            headers={"Origin": "https://www.enigmaniac-escapegame.com"},
        )
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.headers.get("Access-Control-Allow-Origin"), "https://www.enigmaniac-escapegame.com")

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
        self.assertIn("runtime_metrics", payload)
        self.assertIn("conversation_backend", payload)

    def test_admin_faq_generate_creates_missing_file(self):
        missing = app_module.STATIC_DIR / "faq_runningman.json"
        if missing.exists():
            missing.unlink()
        r = self.client.post(
            "/admin/api/faq/generate?token=testtoken",
            json={"brand_id": "runningman", "force": False, "include_default_items": True},
        )
        self.assertEqual(r.status_code, 200)
        payload = r.get_json()
        self.assertTrue(payload.get("ok"))
        self.assertTrue(missing.exists())

    def test_admin_faq_generate_force_all(self):
        r = self.client.post(
            "/admin/api/faq/generate?token=testtoken",
            json={"force": True, "include_default_items": True},
        )
        self.assertEqual(r.status_code, 200)
        payload = r.get_json()
        self.assertTrue(payload.get("ok"))
        self.assertGreaterEqual(len(payload.get("items", [])), 1)

    def test_admin_faq_legacy_editor_routes(self):
        brands = self.client.get("/admin/api/brands?faq_only=1&token=testtoken")
        self.assertEqual(brands.status_code, 200)
        bpayload = brands.get_json()
        self.assertTrue(bpayload.get("ok"))
        self.assertTrue(any((it.get("id") == "retroworld") for it in bpayload.get("brands", [])))

        get_faq = self.client.get("/admin/api/faq/retroworld?token=testtoken")
        self.assertEqual(get_faq.status_code, 200)
        gpayload = get_faq.get_json()
        self.assertIn("items", gpayload)

        put_faq = self.client.put(
            "/admin/api/faq/retroworld?token=testtoken",
            json={"items": [{"q": "Test Q", "a": "Test A", "tags": ["test"]}]},
        )
        self.assertEqual(put_faq.status_code, 200)
        ppayload = put_faq.get_json()
        self.assertEqual(ppayload.get("brand"), "retroworld")
        self.assertEqual(len(ppayload.get("items", [])), 1)

    def test_rate_limit_returns_429(self):
        old_limit = app_module.CHAT_RATE_LIMITER.limit
        app_module.CHAT_RATE_LIMITER.limit = 1
        app_module.CHAT_RATE_LIMITER._hits.clear()
        try:
            first = self.client.post("/chat", json={"message": "bonjour"}, environ_base={"REMOTE_ADDR": "9.9.9.9"})
            self.assertEqual(first.status_code, 200)
            second = self.client.post("/chat", json={"message": "encore"}, environ_base={"REMOTE_ADDR": "9.9.9.9"})
            self.assertEqual(second.status_code, 429)
            payload = second.get_json()
            self.assertEqual(payload.get("error"), "rate_limited")
            self.assertIn("retry_after", payload)
        finally:
            app_module.CHAT_RATE_LIMITER.limit = old_limit
            app_module.CHAT_RATE_LIMITER._hits.clear()

    def test_lead_request_triggers_email_send(self):
        with mock.patch.object(app_module, "LEAD_EMAIL_ENABLED", True):
            with mock.patch.object(app_module, "SMTP_HOST", "smtp.test"):
                with mock.patch.object(app_module, "SMTP_FROM", "bot@test.local"):
                    with mock.patch.object(app_module, "LEAD_EMAIL_TO", "sales@test.local"):
                        with mock.patch.object(app_module, "_send_lead_email", return_value=True) as sender:
                            r = self.client.post("/chat", json={"message": "je veux un devis team building"})
                            self.assertEqual(r.status_code, 200)
                            self.assertTrue(sender.called)

    def test_lead_helper_detects_contact_keywords(self):
        self.assertTrue(app_module._is_lead_request("je veux un devis"))
        self.assertTrue(app_module._is_lead_request("merci de me recontacter"))
        self.assertFalse(app_module._is_lead_request("bonjour"))

    def test_openai_400_retries_with_fallback_model(self):
        class _Resp:
            def __init__(self, status_code, body):
                self.status_code = status_code
                self._body = body

            def raise_for_status(self):
                if self.status_code >= 400:
                    import requests
                    raise requests.exceptions.HTTPError("bad", response=self)

            def json(self):
                return self._body

        calls = []

        def _post(url, headers=None, json=None, timeout=None):
            calls.append(json)
            if len(calls) == 1:
                return _Resp(400, {"error": {"message": "bad request"}})
            return _Resp(200, {"output": [{"content": [{"type": "output_text", "text": "ok fallback"}]}]})

        with mock.patch.object(app_module, "OPENAI_API_KEY", "testkey"):
            with mock.patch.object(app_module, "OPENAI_MODEL", "gpt-5.2"):
                with mock.patch.object(app_module, "OPENAI_FALLBACK_MODEL", "gpt-4.1-mini"):
                    with mock.patch.object(app_module, "OPENAI_API_MODE", "responses"):
                        with mock.patch.object(app_module.requests, "post", side_effect=_post):
                            out = app_module.openai_answer("sys", "user")

        self.assertEqual(out, "ok fallback")
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0].get("model"), "gpt-5.2")
        self.assertEqual(calls[1].get("model"), "gpt-4.1-mini")

    def test_openai_double_400_falls_back_to_chat_completions(self):
        class _Resp:
            def __init__(self, status_code, body):
                self.status_code = status_code
                self._body = body

            def raise_for_status(self):
                if self.status_code >= 400:
                    import requests
                    raise requests.exceptions.HTTPError("bad", response=self)

            def json(self):
                return self._body

        calls = []

        def _post(url, headers=None, json=None, timeout=None):
            calls.append((url, json))
            if "responses" in url:
                return _Resp(400, {"error": {"message": "bad request"}})
            return _Resp(200, {"choices": [{"message": {"content": "ok chat fallback"}}]})

        with mock.patch.object(app_module, "OPENAI_API_KEY", "testkey"):
            with mock.patch.object(app_module, "OPENAI_MODEL", "gpt-5.2"):
                with mock.patch.object(app_module, "OPENAI_FALLBACK_MODEL", "gpt-4.1-mini"):
                    with mock.patch.object(app_module, "OPENAI_API_MODE", "responses"):
                        with mock.patch.object(app_module.requests, "post", side_effect=_post):
                            out = app_module.openai_answer("sys", "user")

        self.assertEqual(out, "ok chat fallback")
        self.assertEqual(len(calls), 3)
        self.assertIn("/v1/responses", calls[0][0])
        self.assertIn("/v1/responses", calls[1][0])
        self.assertIn("/v1/chat/completions", calls[2][0])

    def test_openai_gpt5_prefers_chat_completions_first(self):
        class _Resp:
            def __init__(self, status_code, body):
                self.status_code = status_code
                self._body = body

            def raise_for_status(self):
                if self.status_code >= 400:
                    import requests
                    raise requests.exceptions.HTTPError("bad", response=self)

            def json(self):
                return self._body

        calls = []

        def _post(url, headers=None, json=None, timeout=None):
            calls.append(url)
            return _Resp(200, {"choices": [{"message": {"content": "ok primary chat"}}]})

        with mock.patch.object(app_module, "OPENAI_API_KEY", "testkey"):
            with mock.patch.object(app_module, "OPENAI_MODEL", "gpt-5.2"):
                with mock.patch.object(app_module, "OPENAI_FALLBACK_MODEL", "gpt-4.1-mini"):
                    with mock.patch.object(app_module, "OPENAI_API_MODE", "auto"):
                        with mock.patch.object(app_module.requests, "post", side_effect=_post):
                            out = app_module.openai_answer("sys", "user")

        self.assertEqual(out, "ok primary chat")
        self.assertEqual(len(calls), 1)
        self.assertIn("/v1/chat/completions", calls[0])

    def test_build_openai_history_keeps_recent_messages(self):
        conv = {
            "messages": [
                {"role": "user", "content": "salut"},
                {"role": "assistant", "content": "bonjour"},
                {"role": "system", "content": "ignored"},
                {"role": "user", "content": "escape"},
            ]
        }
        hist = app_module.build_openai_history(conv, max_items=3)
        self.assertEqual(len(hist), 2)
        self.assertEqual(hist[0]["role"], "assistant")
        self.assertEqual(hist[1]["content"], "escape")


if __name__ == "__main__":
    unittest.main()
