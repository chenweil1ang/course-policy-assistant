"""The course-policy chatbox.

    python app.py            serve on 8000
    python app.py --port 8001

Course-policy questions use retrieved local passages and the configured model API.
"""
import json, os, re, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import unquote, urlsplit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import retrieve
import llm

HERE = os.path.dirname(os.path.abspath(__file__))
RETRIEVAL_THRESHOLD = 0.25
MAX_HISTORY_QUESTIONS = 4
MAX_HISTORY_QUESTION_LENGTH = 1000
FOLLOW_UP_REFERENCE = re.compile(
    r"\b(?:it|they|them|this|that|these|those|same|former|latter)\b"
    r"|\b(?:what|how) about\b",
    re.IGNORECASE)


def evidence_reply(section, text):
    if section == "5.2 Late submission":
        return ("The current late-submission passage describes a per-working-day penalty, not a "
                "pass-mark cap. The current policy is: " + text)
    return "The current policy is: " + text


def deadline_clarification(question, passages):
    normalized = question.lower()
    asks_deadline = re.search(r"\b(?:deadline|due date|date of (?:the )?deadline)\b", normalized)
    if (not asks_deadline or re.search(r"\b(?:coursework|cw)\s*[12]\b", normalized)
            or re.search(r"\b(?:feedback|extension)\b", normalized)):
        return None
    has_coursework_1 = any(
        p["section"] == "3 Assessment" and "coursework 1 is due" in p["text"].lower()
        for p in passages)
    has_coursework_2 = any(
        p["section"] == "3 Assessment" and "coursework 2 is due" in p["text"].lower()
        for p in passages)
    if has_coursework_1 and has_coursework_2:
        return "Which deadline do you mean—Coursework 1 or Coursework 2?"
    return None


def retrieve_turn(message, history=None, resolved_question=None):
    all_passages = [p for p in retrieve.load() if p["type"] != "assistant-scope"]
    passages = [p for p in all_passages if not p["superseded"]]
    stale_passages = [p for p in all_passages if p["superseded"]]
    retrieval_question = resolved_question or message
    if resolved_question is None and history and FOLLOW_UP_REFERENCE.search(message):
        retrieval_question = " ".join(history[-2:] + [message])
    clarification = deadline_clarification(retrieval_question, passages)
    if clarification:
        return {
            "mode": "llm-api",
            "reply": clarification,
            "grounded": False,
            "evidence": [],
            "evidence_notice": "The corpus lists more than one coursework deadline.",
        }
    retrieval_query = retrieve.neutralize_query(retrieval_question)
    hits = retrieve.retrieve(retrieval_query, k=len(passages), passages=passages)
    relevant = [h for h in hits if h["score"] > 0]
    common = {"mode": "llm-api", "evidence": []}
    if not relevant or relevant[0]["score"] < RETRIEVAL_THRESHOLD:
        return dict(common, reply="I cannot determine this from the current course materials.",
                    grounded=False,
                    evidence_notice="The current corpus does not provide enough relevant evidence to answer.")

    top = relevant[0]
    stale_hits = retrieve.retrieve(retrieval_query, k=len(stale_passages),
                                   passages=stale_passages) if stale_passages else []
    related_stale = [h for h in stale_hits
                     if h["section"] == top["section"] and h["score"] > 0]
    current_evidence = ([h for h in hits if h["section"] == top["section"]]
                        if top["section"] == "5.2 Late submission" else relevant[:3])
    if top["section"] == "5.2 Late submission":
        passage_order = {(p["file"], p["text"]): i for i, p in enumerate(passages)}
        current_evidence.sort(
            key=lambda h: passage_order.get((h["file"], h["text"]), len(passages)))
    result = dict(common, reply=evidence_reply(top["section"], top["text"]),
                  grounded=True, evidence=current_evidence + related_stale)
    if related_stale:
        result["stale_warning"] = (
            "A superseded passage exists for this topic; the answer uses current evidence only.")
    return result


def llm_api_turn(message, history=None):
    history = history or []
    scope_evidence = [p for p in retrieve.load() if p["type"] == "assistant-scope"]
    intent_result = llm.classify_intent(message, scope_evidence, history=history)
    if intent_result["intent"] != "course_policy":
        scope_by_id = {"G%d" % index: hit
                       for index, hit in enumerate(scope_evidence, 1)}
        evidence_ids = intent_result["evidence_ids"]
        if any(evidence_id not in scope_by_id for evidence_id in evidence_ids):
            raise llm.LLMError("Model API returned an invalid assistant-scope citation.")
        return {
            "mode": "llm-api",
            "intent": intent_result["intent"],
            "reply": intent_result["reply"],
            "grounded": True,
            "evidence": [scope_by_id[evidence_id] for evidence_id in evidence_ids],
        }

    resolved_question = intent_result.get("resolved_question")
    if "resolved_question" in intent_result and not resolved_question:
        passages = [p for p in retrieve.load()
                    if p["type"] != "assistant-scope" and not p["superseded"]]
        clarification = deadline_clarification(message, passages)
        if not clarification:
            clarification = "Could you clarify which course-policy topic you mean?"
        return {
            "mode": "llm-api",
            "reply": clarification,
            "grounded": False,
            "evidence": [],
            "evidence_notice": "The question needs clarification before the corpus can be searched.",
        }
    result = retrieve_turn(message, history, resolved_question=resolved_question)
    if not result.get("grounded"):
        return result
    current_evidence = [hit for hit in result["evidence"] if not hit.get("superseded")]
    generated = llm.answer(resolved_question or message, current_evidence, history=history)
    evidence_by_id = {"E%d" % index: hit
                     for index, hit in enumerate(current_evidence, 1)}
    evidence_ids = generated["evidence_ids"]
    if (generated["refused"] or not evidence_ids
            or any(evidence_id not in evidence_by_id for evidence_id in evidence_ids)):
        outdated_evidence = [hit for hit in result["evidence"] if hit.get("superseded")]
        if current_evidence and outdated_evidence:
            result["reply"] = evidence_reply(
                current_evidence[0]["section"],
                " ".join(hit["text"] for hit in current_evidence))
            result["grounded"] = True
            result["evidence"] = current_evidence + outdated_evidence
            result["api_fallback"] = True
            result["evidence_notice"] = (
                "The model could not provide verifiable citations, so this answer uses the current course passages directly."
            )
            return result

        result["reply"] = "Please ask about a topic covered in the course corpus."
        result["grounded"] = False
        result["evidence"] = []
        result["evidence_notice"] = "The model did not provide a verifiable answer from the current course materials."
        return result

    cited_evidence = []
    for evidence_id in evidence_ids:
        hit = evidence_by_id[evidence_id]
        if hit not in cited_evidence:
            cited_evidence.append(hit)
    outdated_evidence = [hit for hit in result["evidence"] if hit.get("superseded")]
    result["reply"] = generated["answer"]
    result["evidence"] = cited_evidence + outdated_evidence
    return result


def turn(message, mode="llm-api", history=None):
    if mode == "llm-api":
        return llm_api_turn(message, history)
    raise ValueError("Unsupported response mode: " + str(mode))


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        self.send_response(code); self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

    def do_GET(self):
        path = urlsplit(self.path).path
        source_prefix = "/source/"
        if path.startswith(source_prefix):
            filename = unquote(path[len(source_prefix):])
            corpus_dir = os.path.realpath(retrieve.CORPUS)
            if not filename or os.path.basename(filename) != filename or not filename.endswith(".md"):
                self._send(404, b"Source not found.", "text/plain; charset=utf-8")
                return
            available = {passage["file"] for passage in retrieve.load()}
            source_path = os.path.realpath(os.path.join(corpus_dir, filename))
            if filename not in available or os.path.dirname(source_path) != corpus_dir:
                self._send(404, b"Source not found.", "text/plain; charset=utf-8")
                return
            with open(source_path, "rb") as f:
                self._send(200, f.read(), "text/plain; charset=utf-8")
            return
        if path != "/":
            self._send(404, b"Not found.", "text/plain; charset=utf-8")
            return
        with open(os.path.join(HERE, "index.html"), "rb") as f:
            self._send(200, f.read(), "text/html; charset=utf-8")
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        request = json.loads(self.rfile.read(n).decode("utf-8"))
        msg = request.get("message", "")
        mode = request.get("mode", "llm-api")
        history = request.get("history", [])
        if mode != "llm-api":
            self._send(400, json.dumps({"error": "Unsupported response mode."}).encode("utf-8"))
            return
        if (not isinstance(history, list)
                or len(history) > MAX_HISTORY_QUESTIONS
                or any(not isinstance(question, str)
                       or len(question) > MAX_HISTORY_QUESTION_LENGTH
                       for question in history)):
            self._send(400, json.dumps({"error": "History must contain up to four previous questions, each no longer than 1000 characters."}).encode("utf-8"))
            return

        try:
            response = turn(msg, mode, history)
        except llm.LLMError as error:
            self._send(502, json.dumps({"error": str(error)}).encode("utf-8"))
            return
        self._send(200, json.dumps(response).encode("utf-8"))

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    port = 8000
    if "--port" in sys.argv:
        port = int(sys.argv[sys.argv.index("--port") + 1])
    print("serving on http://localhost:%d" % port)
    HTTPServer(("localhost", port), Handler).serve_forever()