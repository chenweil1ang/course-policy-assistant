# Course Policy Assistant

**README prose word count:** approximately 280 words (headings, bullets, paths and commands excluded).

## Project status

This is a local web prototype for answering questions about the included course-policy corpus. It retrieves current passages locally, sends them to a configured language model, displays the answer and cited evidence, and warns about superseded material. Greetings, assistant-scope questions, follow-up references, unsupported questions and ambiguous deadlines are handled separately. The prototype is implemented and tested. Live model responses require an API key; response quality is not measured by the automated tests.

## Technology stack

- Python 3 standard library: HTTP server, lexical retrieval, corpus loading and API client.
- HTML, CSS and JavaScript: browser interface, interaction feedback and source display.
- OpenAI-compatible chat-completions API: configurable through `LLM_API_URL`, `LLM_MODEL` and `LLM_API_KEY`; DeepSeek is the default.
- No third-party Python packages are required.

## Running and inspecting

Set an API key, then run:

```powershell
$env:LLM_API_KEY = "your-key"
python app.py
```

Open <http://localhost:8000>. To run the automated checks:

```powershell
python -B -m unittest -v
python -B retrieve.py --selftest
```

`demo_walkthrough.md` lists representative interactions. `corpus/` contains the source passages. `app.py` and `index.html` implement the server and interface; `retrieve.py` implements local retrieval; `llm.py` implements the model client.

## Development record

- Built a local lexical retriever so source passages remain inspectable and no embedding service is required.
- Added current-versus-superseded policy handling to avoid presenting an old rule as current.
- Added model intent routing, citation validation, abstention for unsupported questions and follow-up reference resolution.
- Added source links, loading feedback, warnings and response history to make the interaction inspectable.
- Checked the implementation with 22 unit tests and the retriever self-test.

## AI assistance

GitHub Copilot was used in three ways:

- **Code:** it helped inspect the requirements, review implementation choices and check the project structure. It gave suggestions and guidance on connecting the model API to the application. It also advised on stopping the model from being swayed by confident or emotional wording, which led to the tone-neutralisation step and the prompt rule that tone is not evidence.
- **Storyboard:** the three-panel storyboard image was generated with Copilot, then checked against the real interface and behaviour.
- **Report:** Copilot helped review whether the report text, system diagram and storyboard match the current code, and suggested wording and accuracy corrections.

Suggestions were reviewed against the source code and the automated tests, and I kept, changed or rejected them accordingly. No API key or private data was supplied.

## Additional files

No separate ZIP file is required. The walkthrough, corpus and reference test record are included in this repository.
