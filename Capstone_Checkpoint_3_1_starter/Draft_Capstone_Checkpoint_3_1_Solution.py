# %% [markdown]
# # Capstone Checkpoint 3.1 — Whole-File Wikipedia Hybrid Retrieval
#
# Each nonempty HTML file becomes ONE searchable document containing its full
# converted text. BM25 and vector search both rank complete files. Long articles
# are split ONLY for embedding, then combined into one vector per file using a
# token-weighted average and unit normalization. No text is truncated, and no
# sections become independently retrievable records. The original prompts, system prompt,
# and weighted hybrid ranking approach are retained.
#
# Setup:
# 1. Use your course Python 3.11 or 3.12 virtual environment.
# 2. Install dependencies:
#    python -m pip install langchain-openai langchain-core python-dotenv langchain-chroma rank-bm25 beautifulsoup4 tiktoken
# 3. Put OPENROUTER_API_KEY=your-key in a .env file beside this script.
# 4. Put Wikipedia beside this script, or supply the corpus folder explicitly.
#
# Run:
#    python Capstone_Checkpoint_3_1_Solution.py
#    python Capstone_Checkpoint_3_1_Solution.py "path/to/Wikipedia"
#
# Sources for retrieval tests are exact paths relative to the Wikipedia folder,
# using forward slashes on every operating system, e.g. "Example.html" or
# "History/Example.html". Use metadata["source"] or retrieved_sources(query)
# when comparing results with your expected source files. Each returned source
# represents the complete converted file, not an excerpt.
#
# The first run embeds whole articles in chroma_wikipedia_whole_files_db.
# A corpus/content/model fingerprint selects the collection, so changes to the
# files create a fresh index automatically. Unchanged complete indexes are reused.
#
# Embedding sections are limited to 7,000 tokens for the configured model,
# below the 8,192-token limit reported by the provider. Sections are temporary;
# Chroma stores the full text and one combined vector for each file. A versioned
# fingerprint prevents reuse of the earlier embedding method.
# The combined retrieved articles must still fit the chat model's context limit.
# Lower NUM_RETRIEVED if needed; the program does not truncate answer context.
# Both embeddings and chat calls use your configured OpenRouter account.
# %%

from __future__ import annotations

import hashlib
import math
import os
import re
import sys
import warnings
from datetime import datetime
from pathlib import Path

import tiktoken

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from rank_bm25 import BM25Okapi

SCRIPT_DIR = Path(__file__).resolve().parent
load_dotenv(SCRIPT_DIR / ".env")


# %%
# Model and API settings
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = "openai/gpt-5.4-mini"
EMBEDDING_MODEL = "openai/text-embedding-3-small"
TEMPERATURE = 0.2

# These settings match text-embedding-3-small. Review them if changing models.
EMBEDDING_ENCODING = "cl100k_base"
EMBEDDING_SECTION_TOKENS = 7000
EMBEDDING_REQUEST_BATCH_SIZE = 16

# Retrieval settings
CHROMA_DIR = str(SCRIPT_DIR / "chroma_wikipedia_whole_files_db")
INDEX_BATCH_SIZE = 16
NUM_RETRIEVED = 4
CANDIDATE_POOL = 10
WEIGHT_BM25 = 0.5
WEIGHT_VECTOR = 0.5

SCENARIO = "wikipedia"
LOG_PATH = Path.cwd() / "checkpoint_3_1_retrieval.log"

# This is the same system prompt used in Checkpoint 1.1.
SYSTEM_PROMPT = (
    "You are a helpful research assistant. Answer the user's question as "
    "accurately as you can. When you reference a specific document, quote "
    "from it directly. If you are not certain, say so rather than guessing."
)


# %%
def check_api_key() -> str:
    """Load and return the OpenRouter key, or stop with a clear error."""
    load_dotenv(SCRIPT_DIR / ".env")
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Put the key provided for this "
            "course in a .env file next to this script and rerun."
        )
    return key


class WholeFileEmbeddings(Embeddings):
    """Produce one normalized, token-weighted embedding per complete input file.

    Only temporary API inputs are split. Original Document text and source IDs
    remain unchanged in Chroma, BM25, retrieval results, and answer context.
    Method: https://developers.openai.com/cookbook/examples/embedding_long_inputs
    """

    def __init__(self, client: OpenAIEmbeddings):
        self.client = client
        self.encoding = tiktoken.get_encoding(EMBEDDING_ENCODING)
        if EMBEDDING_SECTION_TOKENS <= 0 or EMBEDDING_REQUEST_BATCH_SIZE <= 0:
            raise ValueError("Embedding token and batch limits must be positive.")

    def _sections(self, text: str) -> list[tuple[str, int]]:
        """Split on character boundaries, preserving all text and valid Unicode.

        Count tokens in the actual strings sent to the API. Splitting decoded
        token slices can damage Unicode characters, so bisect strings instead.
        No overlap: each part of the original text contributes exactly once.
        """
        if not text:
            raise ValueError("Cannot embed an empty input.")
        pending = [text]
        sections = []
        while pending:
            part = pending.pop()
            count = len(self.encoding.encode(part, disallowed_special=()))
            if count <= EMBEDDING_SECTION_TOKENS:
                sections.append((part, count))
            else:
                if len(part) < 2:
                    raise ValueError("Embedding token limit is too small for one character.")
                midpoint = len(part) // 2
                # Stack order preserves the article's original text order.
                pending.append(part[midpoint:])
                pending.append(part[:midpoint])
        return sections

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        weighted_sums: list[list[float] | None] = [None] * len(texts)
        token_totals = [0] * len(texts)
        pending: list[tuple[int, str, int]] = []

        def flush() -> None:
            vectors = self.client.embed_documents([part for _, part, _ in pending])
            if len(vectors) != len(pending):
                raise RuntimeError("Embedding provider returned an unexpected vector count.")
            for (owner, _, weight), vector in zip(pending, vectors):
                if not vector or not all(math.isfinite(value) for value in vector):
                    raise RuntimeError("Embedding provider returned an invalid vector.")
                if weighted_sums[owner] is None:
                    weighted_sums[owner] = [0.0] * len(vector)
                total = weighted_sums[owner]
                if len(total) != len(vector):
                    raise RuntimeError("Embedding dimensions changed within an article.")
                for dimension, value in enumerate(vector):
                    total[dimension] += weight * value
                token_totals[owner] += weight
            pending.clear()

        for owner, text in enumerate(texts):
            for part, weight in self._sections(text):
                pending.append((owner, part, weight))
                if len(pending) >= EMBEDDING_REQUEST_BATCH_SIZE:
                    flush()
        if pending:
            flush()

        results = []
        for total, tokens in zip(weighted_sums, token_totals):
            if total is None or tokens == 0:
                raise RuntimeError("No embedding was produced for an input document.")
            average = [value / tokens for value in total]
            norm = math.sqrt(sum(value * value for value in average))
            if not math.isfinite(norm) or norm == 0:
                raise RuntimeError("Cannot normalize the combined document embedding.")
            results.append([value / norm for value in average])
        return results

    def embed_query(self, text: str) -> list[float]:
        # Apply the same normalization and length handling to query vectors.
        return self.embed_documents([text])[0]


def get_embeddings() -> WholeFileEmbeddings:
    client = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=check_api_key(),
        base_url=OPENROUTER_BASE_URL,
        # Our wrapper handles token limits; send its sections as raw strings.
        check_embedding_ctx_length=False,
    )
    return WholeFileEmbeddings(client)


def make_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=LLM_MODEL,
        temperature=TEMPERATURE,
        api_key=check_api_key(),
        base_url=OPENROUTER_BASE_URL,
    )


def log(label: str, text: str) -> None:
    timestamp = datetime.now().isoformat(timespec="seconds")
    with LOG_PATH.open("a", encoding="utf-8") as file:
        file.write(f"[{timestamp}] {label}\n{text}\n{'-' * 72}\n")


# %%
def resolve_wikipedia_dir(command_line_path: str | None = None) -> Path:
    """Find the Wikipedia folder or use the path supplied on the command line."""
    if command_line_path:
        candidates = [Path(command_line_path)]
    else:
        candidates = [
            Path("Wikipedia"),
            SCRIPT_DIR / "Wikipedia",
            Path("Labs") / "CapstoneDatasets" / "Wikipedia",
            SCRIPT_DIR / "Labs" / "CapstoneDatasets" / "Wikipedia",
        ]

    for candidate in candidates:
        if candidate.is_dir():
            return candidate

    checked = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"Could not find the Wikipedia folder. Checked: {checked}. "
        "Place the folder beside this script or pass its path as the first "
        "command-line argument."
    )


def html_to_text(path: Path) -> tuple[str, str]:
    """Return a Wikipedia article's title and readable text from an HTML file."""
    html = path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(html, "html.parser")

    # Remove content that is not part of the readable article.
    for element in soup(["script", "style", "noscript"]):
        element.decompose()

    title = soup.title.get_text(" ", strip=True) if soup.title else path.stem
    text = soup.get_text("\n", strip=True)

    # Collapse excessive whitespace while retaining paragraph-like line breaks.
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    text = "\n".join(line for line in lines if line)
    return title, text


def load_wikipedia_documents(wikipedia_dir: Path) -> list[Document]:
    """Load one complete, source-labelled document per nonempty HTML file."""
    paths = sorted(
        path
        for path in wikipedia_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".html", ".htm"}
    )
    if not paths:
        raise RuntimeError(f"No .html or .htm files found in {wikipedia_dir}.")

    documents: list[Document] = []
    for path in paths:
        title, article_text = html_to_text(path)
        relative_source = path.relative_to(wikipedia_dir).as_posix()
        if not article_text.strip():
            warnings.warn(f"Skipping HTML file with no readable text: {relative_source}")
            continue
        documents.append(
            Document(
                page_content=f"Article title: {title}\n\n{article_text}",
                metadata={"source": relative_source, "title": title,
                          "document_id": relative_source},
            )
        )

    if not documents:
        raise RuntimeError("The Wikipedia files were found but contained no text.")
    print(f"Loaded {len(documents)} whole documents from {len(paths)} HTML files.")
    return documents


# %%
_STOPWORDS = {
    "a", "an", "the", "and", "but", "or", "nor", "so", "yet", "for",
    "in", "on", "at", "to", "of", "by", "with", "from", "into", "onto",
    "upon", "about", "above", "below", "between", "through", "during",
    "before", "after", "under", "over", "around", "along", "across", "is",
    "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "i", "we", "you", "he", "she", "it", "they",
    "me", "us", "him", "her", "them", "my", "our", "your", "his", "its",
    "their", "this", "that", "these", "those", "as", "if", "up", "out",
    "not", "no",
}


def tokenize(text: str) -> list[str]:
    """Tokenization used by the BM25 keyword retriever."""
    return [
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if token not in _STOPWORDS
    ]


def normalize(scores: list[float], invert: bool = False) -> list[float]:
    """Min-max normalize scores; optionally make lower raw scores rank higher."""
    if not scores:
        return []

    low, high = min(scores), max(scores)
    if high == low:
        return [0.5] * len(scores)

    normalized = [(score - low) / (high - low) for score in scores]
    return [1.0 - score for score in normalized] if invert else normalized


def build_or_load_vector_db(documents: list[Document]) -> Chroma:
    """Reuse a complete matching index, or index full articles with stable IDs."""
    digest = hashlib.sha256()
    for value in ["whole-files-weighted-v2", EMBEDDING_MODEL, OPENROUTER_BASE_URL,
                  EMBEDDING_ENCODING, str(EMBEDDING_SECTION_TOKENS)]:
        digest.update(value.encode("utf-8") + b"\0")
    for document in documents:
        digest.update(document.metadata["document_id"].encode("utf-8") + b"\0")
        digest.update(document.page_content.encode("utf-8") + b"\0")
    collection_name = "wikipedia_files_" + digest.hexdigest()[:32]
    database = Chroma(
        collection_name=collection_name,
        persist_directory=CHROMA_DIR,
        embedding_function=get_embeddings(),
    )
    expected_ids = {doc.metadata["document_id"] for doc in documents}
    existing_ids = set(database.get(include=[])["ids"])
    if existing_ids == expected_ids:
        print(f"Loading existing whole-file vector database from {CHROMA_DIR}/")
        return database

    # Resume an interrupted first run; only successfully indexed IDs are reused.
    pending = [doc for doc in documents if doc.metadata["document_id"] not in existing_ids]
    print(f"Embedding {len(pending)} whole files (the first run may take several minutes)...")
    print("Long files use temporary embedding sections; each file gets one combined vector.")
    for start in range(0, len(pending), INDEX_BATCH_SIZE):
        batch = pending[start:start + INDEX_BATCH_SIZE]
        try:
            database.add_documents(
                documents=batch,
                ids=[doc.metadata["document_id"] for doc in batch],
            )
        except Exception as error:
            sources = ", ".join(doc.metadata["source"] for doc in batch)
            raise RuntimeError(
                f"Whole-file embedding failed for batch containing: {sources}. "
                "The program keeps complete files and uses token-limited sections "
                "only for embedding. If the provider reports an input-length error, "
                "check EMBEDDING_ENCODING and EMBEDDING_SECTION_TOKENS against "
                "the selected model. Successfully indexed batches will be reused. "
                f"Provider error: {error}"
            ) from error
        print(f"  Indexed {min(start + len(batch), len(pending))}/{len(pending)} new files.")
    print(f"Indexed {len(documents)} whole files in {CHROMA_DIR}/")
    return database


# %%
class HybridWikipediaRetriever:
    """Combine BM25 keyword rankings with Chroma vector rankings."""

    def __init__(self, documents: list[Document], database: Chroma):
        self._documents = documents
        self._database = database
        self._content_by_id = {
            document.metadata["document_id"]: document.page_content
            for document in documents
        }
        self._metadata_by_id = {
            document.metadata["document_id"]: document.metadata
            for document in documents
        }
        self._bm25 = BM25Okapi(
            [tokenize(document.page_content) for document in documents]
        )
        print(f"Hybrid retriever ready over {len(documents)} whole files.")

    def bm25_top_k(
        self, query: str, k: int
    ) -> list[tuple[str, str, dict, float]]:
        scores = self._bm25.get_scores(tokenize(query))
        top_indexes = sorted(
            range(len(scores)), key=lambda index: scores[index], reverse=True
        )[:k]

        results = []
        for index in top_indexes:
            document = self._documents[index]
            results.append(
                (
                    document.metadata["document_id"],
                    document.page_content,
                    document.metadata,
                    float(scores[index]),
                )
            )
        return results

    def vector_top_k(
        self, query: str, k: int
    ) -> list[tuple[str, str, dict, float]]:
        results = self._database.similarity_search_with_score(query, k=k)
        return [
            (
                document.metadata["document_id"],
                document.page_content,
                document.metadata,
                float(distance),
            )
            for document, distance in results
        ]

    def get_top_k(
        self, query: str, k: int = NUM_RETRIEVED
    ) -> list[tuple[str, str, dict, float]]:
        """Fuse normalized BM25 scores and inverted vector distances."""
        bm25_results = self.bm25_top_k(query, CANDIDATE_POOL)
        vector_results = self.vector_top_k(query, CANDIDATE_POOL)

        content_by_id: dict[str, str] = {}
        metadata_by_id: dict[str, dict] = {}
        bm25_normalized: dict[str, float] = {}
        vector_normalized: dict[str, float] = {}

        bm25_scores = normalize([result[3] for result in bm25_results])
        for (document_id, content, metadata, _), score in zip(
            bm25_results, bm25_scores
        ):
            content_by_id[document_id] = content
            metadata_by_id[document_id] = metadata
            bm25_normalized[document_id] = score

        # Chroma returns distance, where a lower value means a better match.
        vector_scores = normalize(
            [result[3] for result in vector_results], invert=True
        )
        for (document_id, content, metadata, _), score in zip(
            vector_results, vector_scores
        ):
            content_by_id[document_id] = content
            metadata_by_id[document_id] = metadata
            vector_normalized[document_id] = score

        fused = [
            (
                document_id,
                content,
                metadata_by_id[document_id],
                WEIGHT_BM25 * bm25_normalized.get(document_id, 0.0)
                + WEIGHT_VECTOR * vector_normalized.get(document_id, 0.0),
            )
            for document_id, content in content_by_id.items()
        ]
        fused.sort(key=lambda result: result[3], reverse=True)
        return fused[:k]

    def retrieved_sources(self, query: str, k: int = NUM_RETRIEVED) -> list[str]:
        """Return ranked file paths for tests, without calling the answer LLM.

        Example: expected_sources = {"Example.html", "History/Example.html"}
        Compare those paths with set(retriever.retrieved_sources(query)).
        """
        return [metadata["source"] for _, _, metadata, _ in self.get_top_k(query, k)]

    def retrieved_context(self, query: str) -> tuple[str, list[tuple[str, float]]]:
        results = self.get_top_k(query, NUM_RETRIEVED)
        context_parts = []
        evidence = []

        for _, content, metadata, score in results:
            source = metadata["source"]
            context_parts.append(
                f"[Source: {source}]\n{content}"
            )
            evidence.append((source, score))

        context = "\n\n---\n\n".join(context_parts)
        return context, evidence


# %%
def my_probe_prompts() -> list[str]:
    """The same custom probe prompts used in Checkpoint 1.1."""
    return [
        "When did Britney Spears record her vocals for the song ...Baby One More Time? Quote the article.",
        "What song did Yankovic parody with My Bologna? Quote the article.",
        "Were the television broadcaster the same for the 28th and 52nd Annual Tony award ceremonies?",
        "When were the official invitations to compete in the 1912 Olympics sent out? Quote the article.",
        "What country hosted the 1912 Olympics?",
        "Where did principal photography for the movie A Beautiful Day in the Neighborhood take place? Quote the article.",
        "How old is Chris Parker in the movie Adventures in Babysitting? Quote the article.",
        "When was the movie A Beautiful Day in teh Neighborhood originally going to be released?",
        "How many false starts is a player allowed in the 100 meter dash before they are disqualified? Quote the article.",
    ]


def answer_question(llm: ChatOpenAI, question: str, context: str) -> str:
    """Ask the LLM to answer from the context returned by hybrid retrieval."""
    user_message = (
        "Use the following retrieved Wikipedia articles as your source material. "
        "If they do not contain enough information, say that you are not certain "
        "rather than using outside knowledge. Identify the source article when "
        "you quote it.\n\n"
        f"Retrieved Wikipedia articles:\n{context}\n\n"
        f"Question: {question}"
    )
    response = llm.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ]
    )
    return response.content if hasattr(response, "content") else str(response)


# %%
def run() -> None:
    check_api_key()
    supplied_path = sys.argv[1] if len(sys.argv) > 1 else None
    wikipedia_dir = resolve_wikipedia_dir(supplied_path)

    documents = load_wikipedia_documents(wikipedia_dir)
    database = build_or_load_vector_db(documents)
    retriever = HybridWikipediaRetriever(documents, database)
    llm = make_llm()
    prompts = my_probe_prompts()

    print("Capstone Checkpoint 3.1 — Wikipedia hybrid retrieval")
    print(f"Scenario: {SCENARIO} | Model: {LLM_MODEL}")
    print(f"Corpus: {wikipedia_dir}")
    print(f"Logging to: {LOG_PATH.name}\n")

    for number, prompt in enumerate(prompts, start=1):
        print("=" * 72)
        print(f"PROMPT {number}/{len(prompts)}: {prompt}")

        try:
            context, evidence = retriever.retrieved_context(prompt)
            print("\nRETRIEVED:")
            for source, score in evidence:
                print(f"  {source}: hybrid score={score:.3f}")

            log(f"RETRIEVAL {number}: {prompt}", f"retrieved={evidence}")
            response = answer_question(llm, prompt, context)
        except Exception as error:
            print(f"\nERROR: {error}\n")
            log(f"ERROR {number}: {prompt}", str(error))
            continue

        print(f"\nRESPONSE:\n{response}\n")
        log(
            f"PROMPT {number}: {prompt}",
            f"retrieved={evidence}\nresponse={response}",
        )

    print("=" * 72)
    print("Done. Review the retrieved sources and answers in the output and log.")


if __name__ == "__main__":
    run()

# %% [markdown]
# Capstone Checkpoint 3.1: compare the retrieved source paths with the expected
# files for your test queries. Retrieval is evaluated at the whole-file level.
