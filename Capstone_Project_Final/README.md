# Wikipedia Question-Answering Chatbot — Final Capstone Project

**Author:** Duncan Earl  
**Version:** v4 (Final)
**Source file reviewed:** `Capstone_Project_Final_v4.py`

## Purpose

This project helps users find and connect information across approximately 2,400 saved Wikipedia articles without reading each article individually. Users can ask factual questions, ask questions that require information from multiple articles, and ask follow-up questions that refer to earlier conversation. Answers include references to the supporting local articles.

The system uses retrieval-augmented generation (RAG). My design priorities are accuracy, cost, and response time, in that order. It runs as a command-line chatbot; it is not a hosted website or a model trained on the articles.

## How the system works

1. **Prepare the corpus:** Extract readable text from local Wikipedia HTML files and divide it into passages. Save the text and persist passage embeddings in Chroma for reuse.
2. **Understand the question:** Resolve conversational references and, when needed, break the question into the individual facts that must be established. Simple standalone questions can bypass the model-based planning step.
3. **Retrieve evidence:** Combine BM25 keyword search with vector similarity search. Give additional attention to articles explicitly named in the question and balance evidence across requested facts.
4. **Generate and check an answer:** Route simpler requests to the smaller model and use the stronger model for harder requests or escalation. Validate citation references against actual passage excerpts. Selectively use an additional model call to check factual support and completeness.
5. **Search again when necessary:** Perform bounded follow-up searches for missing evidence, replacing less relevant context while retaining supporting citations. Return supported claims and acknowledge when a complete answer cannot be established.

Supported answers can be reused from an in-memory session cache. Evaluation disables this answer cache.

## Files and folder setup

Place this README in the final project folder along with:

| Item beside the script | Purpose |
| --- | --- |
| `Wikipedia/` | Local `.html` or `.htm` articles; required for a fresh build. Subfolders are supported. |
| `.env` | Local configuration containing the reviewer's OpenRouter API key. |
| `testInputs.json` | Evaluation questions and reference answers; required only for evaluation. |
| `Wikipedia_text/` | Generated or reused article text and conversion manifest. |
| `Chroma/` | Generated or reused vectors, passage snapshot, and index metadata. |
| `evaluation_results/` | Generated timestamped evaluation output folders. |

## Installation

Use Python 3.12 as a practical starting point. This submission was run with Python 3.12.14 with the following commands.
py -3.12 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install beautifulsoup4 tiktoken rank-bm25 langchain-chroma chromadb openai python-dotenv

| Role | Configured identifier |
| --- | --- |
| Planning, stronger answering, verification, evaluation judge | `openai/gpt-5.4-mini` |
| Simpler answering | `openai/gpt-5-nano` |
| Embeddings | `openai/text-embedding-3-small` |
| API base URL | `https://openrouter.ai/api/v1` |


## Running the program
python Capstone_Project_Final_v4.py

The first run prepares missing text and embeddings, which can take time and incur API charges. Later runs reuse a consistent saved index. At the `You:` prompt, try:

Who played Phoebe in Friends?
When was she born?
What TV series costarred Lisa Kudrow and Jennifer Aniston?

These examples require the relevant articles to be present in the supplied corpus. Responses display source labels and article filenames, along with routing, elapsed time, model attempts, and token usage.

| Chat command | Action |
| --- | --- |
| `/reset` | Clear conversation memory and the answer cache. |
| `/cache-clear` | Clear only the answer cache. |
| `/stats` | Show model usage and answer-cache statistics. |
| `/exit` or `/quit` | End the session. |

### Run evaluation

python Capstone_Project_Final_v4.py --evaluate

Each evaluation run creates a timestamped folder under `evaluation_results/` containing:

- **`report.json`:** Detailed results, configuration, fingerprints, judgments, usage, and costs. Updated after each evaluated question.
- **`results.csv`:** Summary rows for completed evaluation runs.
- **`human_review.json`:** Failures and representative cases for manual assessment, written when evaluation finishes.

## Known limitations and reviewer considerations

- **Snapshot scope:** Answers reflect the saved articles, not live Wikipedia or current web information. Missing or poorly extracted evidence can lead to an incomplete answer even when the fact exists elsewhere.
- **Verification is imperfect:** Citation checks establish that excerpts exist in the supplied passages. They do not, by themselves, prove that a claim follows from them. Selective verification skips the additional model check for some simple answers; model-based verification can also make mistakes.
- **Bounded retrieval:** Passage limits, context budgets, and capped search rounds can omit relevant evidence. Multipart questions, ambiguous references, and comparisons remain challenging. This version uses keyword/vector retrieval, not a knowledge graph.
- **Cache freshness:** When an HTML fingerprint exists, changed source HTML triggers an error rather than an automatic rebuild. Back up and move both `Chroma/` and `Wikipedia_text/` out of the project folder, then rerun against the intended HTML corpus. Legacy caches without a fingerprint may be reused with a warning; missing HTML prevents freshness verification. Rebuilding embeddings costs money.
- **Cost and latency:** Complex questions can require multiple searches and model calls. Rate-limit retries may add substantial delay. The chat-model attempt cap does not cap embedding calls or evaluation judge calls. Answer-cost reporting includes query embeddings and internal model work, while startup/index embeddings and evaluation judging are reported separately. Missing charges are marked incomplete; fallback list-price estimates are labeled as estimates.
- **Evaluation interpretation:** The judge is an LLM, so its pass/fail and grounding assessments need human review. Source recall measures overlap with the expected article list, not full answer quality. Reported answer latency excludes judging; warm caches and provider conditions affect comparisons. No live performance figures were established during README preparation.
- **Privacy and logging:** Conversation state and the answer cache are session-only, but the default `retrieval_diagnostics.jsonl` persists questions, resolved queries, plans, and retrieval details. `diagnostics.jsonl` records failure metadata. Evaluation reports contain questions, answers, and evidence. Retrieved text and requests are sent to the external API provider. Review generated logs before public sharing.
- **Application scope:** This is a local instructional prototype without a graphical interface or multi-user service. Prompt-injection defenses and output checks reduce some risks but are not a security guarantee. Corpus-only answering is the default. The optional `--allow-general-knowledge` mode disables answer caching and permits supplemental text in generation, but verification can remove that text; it is not a reliable general-purpose fallback.
- **Reproducibility:** Package versions, provider availability, and model behavior may change. Keep the corpus, test file, dependency versions, and evaluation report together when comparing results. The offline test result above should not be interpreted as validation of every runtime path.

## Suggested review sequence
Run the chatbot with the supplied corpus, test a direct question and a follow-up. Then run the evaluation attempting the automated full test set. Inspect cited articles alongside the automated scores.
