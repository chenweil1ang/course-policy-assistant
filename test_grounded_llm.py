import json
import unittest
from http.server import HTTPServer
from threading import Thread
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import app
import retrieve


class GroundedCourseAssistantTests(unittest.TestCase):
    def setUp(self):
        classifier = patch.object(
            app.llm, "classify_intent",
            return_value={"intent": "course_policy", "reply": "", "evidence_ids": []},
        )
        self.classify = classifier.start()
        self.addCleanup(classifier.stop)

    def test_llm_api_reply_is_answer_text_with_cited_evidence(self):
        question = "When is Coursework 1 due?"
        generated = {"answer": "It is due on 14 November.",
                     "evidence_ids": ["E1"], "refused": False}
        with patch.object(app.llm, "answer", return_value=generated) as api:
            result = app.turn(question, "llm-api")

        self.assertEqual(result["reply"], generated["answer"])
        self.assertIsInstance(result["reply"], str)
        self.assertEqual(result["evidence"], [api.call_args.args[1][0]])
        self.assertFalse(result["evidence"][0]["superseded"])

    def test_deadline_follow_up_uses_resolved_topic_and_respects_date_only(self):
        self.classify.return_value = {
            "intent": "course_policy",
            "reply": "",
            "evidence_ids": [],
            "resolved_question": "When is Coursework 1 due? Reply with the date only.",
        }
        generated = {"answer": "14 November at 12:00.",
                     "evidence_ids": ["E1"], "refused": False}
        with patch.object(app.llm, "answer", return_value=generated) as api:
            result = app.turn(
                "You only need to reply the date of the deadline",
                history=["When is Coursework 1 due?"],
            )

        self.assertTrue(result["grounded"])
        self.assertEqual(result["reply"], "14 November at 12:00.")
        self.assertEqual(api.call_args.args[0], self.classify.return_value["resolved_question"])
        self.assertTrue(all(not hit["superseded"] for hit in api.call_args.args[1]))
        self.assertTrue(any("Coursework 1 is due on 14 November" in hit["text"]
                            for hit in api.call_args.args[1]))
        self.assertFalse(any("five marks per working day" in hit["text"]
                             for hit in api.call_args.args[1]))

    def test_ambiguous_deadline_follow_up_asks_which_coursework(self):
        self.classify.return_value = {
            "intent": "course_policy", "reply": "", "evidence_ids": [],
            "resolved_question": "",
        }
        with patch.object(app.llm, "answer") as api:
            result = app.turn("You only need to reply the date of the deadline")

        self.assertIn("Coursework 1 or Coursework 2", result["reply"])
        self.assertFalse(result["grounded"])
        self.assertEqual(result["evidence"], [])
        api.assert_not_called()

    def test_superseded_disclaimer_cannot_win_current_policy_retrieval(self):
        old_passages = [p for p in retrieve.load() if p["superseded"]]
        self.assertTrue(old_passages)
        self.assertFalse(any("retained only for reference" in p["text"] for p in old_passages))

        result = app.retrieve_turn("I'm certain late work is capped at the pass mark.")
        self.assertTrue(result["grounded"])
        self.assertFalse(result["evidence"][0]["superseded"])
        self.assertIn("five marks per working day", result["evidence"][0]["text"])

    def test_stale_rule_conflict_falls_back_to_current_passage_if_model_refuses(self):
        question = ("I'm frustrated that late work is capped at the pass mark. "
                    "This seems unfair — does that cap still apply?")
        refusal = {"answer": "I cannot answer.", "evidence_ids": [], "refused": True}
        with patch.object(app.llm, "answer", return_value=refusal):
            result = app.turn(question, "llm-api")

        self.assertTrue(result["grounded"])
        self.assertIn("not a pass-mark cap", result["reply"])
        self.assertIn("five marks per working day", result["reply"])
        self.assertTrue(any(hit["superseded"] for hit in result["evidence"]))
        self.assertTrue(any(not hit["superseded"] for hit in result["evidence"]))
        self.assertTrue(result["api_fallback"])
        self.assertIn("current course passages directly", result["evidence_notice"])

    def test_default_model_receives_only_current_evidence(self):
        questions = [
            "I'm certain late work is capped at the pass mark, right?",
            "I'm certain late work is not capped at the pass mark. What is the current rule?",
            "What is the current late submission rule?",
            "I'm frustrated that late work is capped at the pass mark. Does that cap still apply?",
            "That's unfair and ridiculous: late work is definitely capped at the pass mark.",
        ]
        generated = {
            "answer": "Generated from current policy.",
            "evidence_ids": ["E1"],
            "refused": False,
        }
        with patch.object(app.llm, "answer", return_value=generated) as api:
            results = [app.turn(question) for question in questions]

        self.assertEqual(api.call_count, len(questions))
        self.assertTrue(all(result["mode"] == "llm-api" and result["grounded"] for result in results))
        contexts = [call.args[1] for call in api.call_args_list]
        self.assertTrue(all(context for context in contexts))
        self.assertTrue(all(not hit["superseded"] for context in contexts for hit in context))
        self.assertTrue(all(hit["type"] != "assistant-scope"
                            for context in contexts for hit in context))
        context_texts = [[hit["text"] for hit in context] for context in contexts]
        self.assertTrue(all(context_texts[0] == texts for texts in context_texts[1:]))
        for result in results:
            self.assertIn("superseded", result["stale_warning"].lower())
            self.assertTrue(any(hit["superseded"] for hit in result["evidence"]))
            self.assertTrue(any(not hit["superseded"] and hit["date"] == "2026-09-12"
                                for hit in result["evidence"]))

    def test_retrieval_query_discards_tone_but_keeps_policy_terms(self):
        question = "I'm really frustrated that late work is capped at the pass mark. This is unfair."
        query = retrieve.neutralize_query(question).lower()

        self.assertIn("late work", query)
        self.assertIn("pass mark", query)
        self.assertNotIn("frustrated", query)
        self.assertNotIn("unfair", query)

    def test_follow_up_pronoun_uses_previous_question_for_retrieval(self):
        previous_question = "Are office hours held on Sunday?"
        question = "I'm certain they are held on Sunday."
        generated = {
            "answer": "The timetable lists office hours on Tuesday afternoons.",
            "evidence_ids": ["E1"],
            "refused": False,
        }
        with patch.object(app.llm, "answer", return_value=generated) as api:
            result = app.turn(question, history=[previous_question])

        self.assertTrue(result["grounded"])
        self.assertEqual(api.call_args.kwargs["history"], [previous_question])
        self.assertIn("Office hours are on Tuesday afternoons",
                      " ".join(hit["text"] for hit in api.call_args.args[1]))
        self.assertIn("Tuesday afternoons", result["reply"])

    def test_supported_questions_send_retrieved_corpus_passages_to_model(self):
        questions = [
            "When is Coursework 1 due, and how should I submit it?",
            "Can I request an extension, and what steps are required?",
            "What does the course say about citing sources and collusion?",
            "When are office hours, and how do I book them?",
        ]
        for question in questions:
            with self.subTest(question=question):
                generated = {"answer": "Generated answer.", "evidence_ids": ["E1"], "refused": False}
                with patch.object(app.llm, "answer", return_value=generated) as api:
                    result = app.turn(question)
                self.assertEqual(result["mode"], "llm-api")
                self.assertTrue(result["grounded"])
                self.assertEqual(result["reply"], "Generated answer.")
                self.assertTrue(api.call_args.args[1])
                self.assertTrue(all(not hit["superseded"] for hit in api.call_args.args[1]))
                self.assertTrue(all(hit["source"] and hit["date"] and hit["text"]
                                    for hit in result["evidence"]))

    def test_only_model_cited_passages_are_shown_as_support(self):
        generated = {"answer": "Feedback follows the deadline.", "evidence_ids": ["E2"], "refused": False}
        with patch.object(app.llm, "answer", return_value=generated) as api:
            result = app.turn("When is Coursework 1 due and how should I submit it?")

        model_context = api.call_args.args[1]
        self.assertGreaterEqual(len(model_context), 2)
        self.assertEqual(result["evidence"], [model_context[1]])
        self.assertIn("Feedback on coursework 1", result["evidence"][0]["text"])

    def test_model_refusal_or_invalid_citation_is_rejected(self):
        question = "What is the appeals process for Coursework 1?"
        responses = [
            {"answer": "The corpus does not say.", "evidence_ids": [], "refused": True},
            {"answer": "The exam board handles it.", "evidence_ids": ["E99"], "refused": False},
        ]
        for generated in responses:
            with self.subTest(generated=generated):
                with patch.object(app.llm, "answer", return_value=generated):
                    result = app.turn(question)
                self.assertFalse(result["grounded"])
                self.assertIn("Please ask about a topic covered in the course corpus.", result["reply"])
                self.assertEqual(result["evidence"], [])
                self.assertTrue(result["evidence_notice"])

    def test_unsupported_questions_abstain_without_calling_model(self):
        questions = [
            "What is the process for appealing a final grade?",
            "Tell me the campus weather tomorrow.",
        ]
        with patch.object(app.llm, "answer") as api:
            for question in questions:
                with self.subTest(question=question):
                    result = app.turn(question)
                    self.assertFalse(result["grounded"])
                    self.assertIn("cannot determine", result["reply"].lower())
                    self.assertTrue(result["evidence_notice"])
                    self.assertEqual(result["evidence"], [])
        api.assert_not_called()

    def test_scope_question_uses_assistant_guide_from_corpus(self):
        reply = "You can ask about assessments, extensions, integrity and the timetable."
        self.classify.return_value = {
            "intent": "assistant_scope", "reply": reply, "evidence_ids": ["G1"]}
        result = app.turn("What can I ask you?")
        self.assertTrue(result["grounded"])
        self.assertEqual(result["reply"], reply)
        self.assertEqual(result["evidence"][0]["section"], "Assistant scope")

    def test_greeting_gets_grounded_assistant_introduction(self):
        reply = "Hi! I can help with questions about the course topics in the corpus."
        self.classify.return_value = {
            "intent": "greeting", "reply": reply, "evidence_ids": ["G1"]}
        result = app.turn("Hi")
        self.assertTrue(result["grounded"])
        self.assertEqual(result["reply"], reply)
        self.assertEqual(result["evidence"][0]["section"], "Assistant scope")

    def test_source_route_and_modes(self):
        server = HTTPServer(("localhost", 0), app.Handler)
        thread = Thread(target=server.serve_forever)
        thread.start()
        base_url = "http://localhost:%d" % server.server_port
        try:
            with urlopen(base_url + "/source/handbook_late.md") as response:
                source = response.read().decode("utf-8")
                self.assertIn("text/plain", response.headers["Content-Type"])
            self.assertIn("five marks per working day", source)

            with self.assertRaises(HTTPError) as error:
                urlopen(base_url + "/source/%2E%2E%2Fapp.py")
            self.assertEqual(error.exception.code, 404)

            generated = {"answer": "Model answer.", "evidence_ids": ["E1"], "refused": False}
            with patch.object(app.llm, "answer", return_value=generated):
                payload = {"message": "What is the current late submission rule?", "mode": "llm-api"}
                request = Request(
                    base_url + "/",
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urlopen(request) as response:
                    result = json.loads(response.read().decode("utf-8"))
                    self.assertEqual(result["mode"], "llm-api")

            removed_mode = Request(
                base_url + "/",
                data=json.dumps({"message": "What is the current late rule?",
                                 "mode": "simulated-sycophancy"}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with self.assertRaises(HTTPError) as error:
                urlopen(removed_mode)
            self.assertEqual(error.exception.code, 400)
        finally:
            server.shutdown()
            thread.join()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
