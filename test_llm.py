import json
import os
import unittest
from io import BytesIO
from unittest.mock import patch

import llm


class LLMClientTests(unittest.TestCase):
    def test_default_endpoint_and_model_use_deepseek(self):
        response = BytesIO(json.dumps({
            "choices": [{"message": {"content": json.dumps({
                "answer": "Answer from evidence.",
                "evidence_ids": [],
                "refused": True,
            })}}]
        }).encode("utf-8"))
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True):
            with patch.object(llm, "urlopen", return_value=response) as open_url:
                llm.answer("Question?", [])

        request = open_url.call_args.args[0]
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(request.full_url, "https://api.deepseek.com/chat/completions")
        self.assertEqual(payload["model"], "deepseek-chat")

    def test_request_uses_configured_model_and_only_supplied_evidence(self):
        response = BytesIO(json.dumps({
            "choices": [{"message": {"content": json.dumps({
                "answer": "Answer from evidence.",
                "evidence_ids": ["E1"],
                "refused": False,
            })}}]
        }).encode("utf-8"))
        evidence = [{
            "source": "Module Handbook 2026–27",
            "section": "3 Assessment",
            "date": "2026-09-12",
            "text": "Coursework 1 is due on 14 November.",
        }]
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "OPENAI_MODEL": "test-model"}):
            with patch.object(llm, "urlopen", return_value=response) as open_url:
                answer = llm.answer("When is Coursework 1 due?", evidence)

        request = open_url.call_args.args[0]
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(answer["answer"], "Answer from evidence.")
        self.assertEqual(answer["evidence_ids"], ["E1"])
        self.assertEqual(request.full_url, llm.API_URL)
        self.assertEqual(request.get_header("Authorization"), "Bearer test-key")
        self.assertEqual(payload["model"], "test-model")
        self.assertIn("Coursework 1 is due on 14 November.", payload["messages"][1]["content"])
        self.assertIn('"id": "E1"', payload["messages"][1]["content"])
        system_prompt = payload["messages"][0]["content"]
        self.assertIn("natural, conversational answer", system_prompt)
        self.assertIn("do not use a fixed template", system_prompt)
        self.assertIn("using only the supplied evidence", system_prompt)
        self.assertIn("confidence, emotion, repetition", system_prompt)
        self.assertIn("correct it politely", system_prompt)
        self.assertIn("replying with a date only", system_prompt)
        self.assertEqual(payload["temperature"], 0.1)

    def test_request_uses_configured_api_url_key_and_model(self):
        response = BytesIO(json.dumps({
            "choices": [{"message": {"content": json.dumps({
                "answer": "Answer from evidence.",
                "evidence_ids": [],
                "refused": True,
            })}}]
        }).encode("utf-8"))
        with patch.dict(os.environ, {
            "LLM_API_KEY": "test-key",
            "LLM_API_URL": "https://api.deepseek.com/chat/completions",
            "LLM_MODEL": "deepseek-chat",
        }):
            with patch.object(llm, "urlopen", return_value=response) as open_url:
                llm.answer("Question?", [])

        request = open_url.call_args.args[0]
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(request.full_url, "https://api.deepseek.com/chat/completions")
        self.assertEqual(request.get_header("Authorization"), "Bearer test-key")
        self.assertEqual(payload["model"], "deepseek-chat")

    def test_follow_up_context_is_sent_only_for_reference_resolution(self):
        response = BytesIO(json.dumps({
            "choices": [{"message": {"content": json.dumps({
                "answer": "Office hours are on Tuesday afternoons.",
                "evidence_ids": ["E1"],
                "refused": False,
            })}}]
        }).encode("utf-8"))
        with patch.dict(os.environ, {"LLM_API_KEY": "test-key"}, clear=True):
            with patch.object(llm, "urlopen", return_value=response) as open_url:
                llm.answer(
                    "I'm certain they are held on Sunday.",
                    [{"source": "Timetable", "section": "1 Sessions",
                      "date": "2026-09-01", "text": "Office hours are on Tuesday afternoons."}],
                    history=["Are office hours held on Sunday?"],
                )

        payload = json.loads(open_url.call_args.args[0].data.decode("utf-8"))
        self.assertIn("only to resolve references", payload["messages"][0]["content"])
        self.assertIn("not evidence", payload["messages"][1]["content"])
        self.assertIn("Are office hours held on Sunday?", payload["messages"][1]["content"])
        self.assertIn("I'm certain they are held on Sunday.", payload["messages"][1]["content"])

    def test_intent_classifier_answers_greeting_from_scope_evidence(self):
        response = BytesIO(json.dumps({
            "choices": [{"message": {"content": json.dumps({
                "intent": "greeting",
                "reply": "Hi! I can help with questions about the course topics.",
                "evidence_ids": ["G1"],
                "resolved_question": "",
            })}}]
        }).encode("utf-8"))
        scope = [{
            "section": "Assistant scope",
            "text": "You can ask about coursework and office hours.",
        }]
        with patch.dict(os.environ, {"LLM_API_KEY": "test-key"}, clear=True):
            with patch.object(llm, "urlopen", return_value=response) as open_url:
                result = llm.classify_intent("Hi", scope)

        payload = json.loads(open_url.call_args.args[0].data.decode("utf-8"))
        self.assertEqual(result["intent"], "greeting")
        self.assertEqual(result["evidence_ids"], ["G1"])
        self.assertEqual(result["resolved_question"], "")
        self.assertIn("assistant_scope", payload["messages"][0]["content"])
        self.assertIn('"section": "Assistant scope"', payload["messages"][1]["content"])
        self.assertIn("Current message:\nHi", payload["messages"][1]["content"])

    def test_intent_classifier_resolves_course_follow_up_without_adding_facts(self):
        resolved_question = "When is Coursework 1 due? Reply with the date only."
        response = BytesIO(json.dumps({
            "choices": [{"message": {"content": json.dumps({
                "intent": "course_policy",
                "reply": "",
                "evidence_ids": [],
                "resolved_question": resolved_question,
            })}}]
        }).encode("utf-8"))
        with patch.dict(os.environ, {"LLM_API_KEY": "test-key"}, clear=True):
            with patch.object(llm, "urlopen", return_value=response) as open_url:
                result = llm.classify_intent(
                    "You only need to reply the date of the deadline",
                    [],
                    history=["When is Coursework 1 due?"],
                )

        payload = json.loads(open_url.call_args.args[0].data.decode("utf-8"))
        self.assertEqual(result["resolved_question"], resolved_question)
        self.assertIn("preserving constraints", payload["messages"][0]["content"])
        self.assertIn("When is Coursework 1 due?", payload["messages"][1]["content"])

    def test_missing_api_key_fails_before_network_request(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch.object(llm, "urlopen") as open_url:
                with self.assertRaisesRegex(llm.LLMError, "LLM_API_KEY"):
                    llm.answer("Question?", [])
        open_url.assert_not_called()


if __name__ == "__main__":
    unittest.main()
