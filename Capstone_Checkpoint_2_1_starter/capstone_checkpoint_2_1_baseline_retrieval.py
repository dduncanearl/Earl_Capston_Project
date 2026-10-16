# %% [markdown]
# # Capstone Checkpoint 2.1 — Retrieval Strategy Design and Baseline Implementation
# **MO-LLM Module 2 / Required Capstone Checkpoint (120 minutes)**
#
# ## What this checkpoint is
#
# In Checkpoint 1.1, you showed that a plain LLM can't reliably answer questions about
# your corpus. Now, you will **add retrieval**: Design a retrieval strategy for your scenario
# and build a **baseline retrieval system** that finds the most relevant documents for
# a query, so the model can ground its answers in them.
#
# This mirrors the Module 2 labs — keyword (BM25), vector (semantic), and hybrid
# retrieval — applied to your own capstone corpus. The graded deliverable is the completed 
# Capstone Checkpoint 2.1 worksheet, which includes your written responses and evidence of your 
# retrieval system implementation and testing. This script provides a small working example of 
# baseline retrieval. Use it to understand the retrieval workflow, then adapt the code to implement 
# and test a baseline retriever using your selected capstone dataset.
#
# **Learning outcomes (Module 2):**
# 1. Design a retrieval strategy appropriate for a given dataset and query type.
# 2. Implement and test a baseline retrieval system using structured and/or semantic
#    approaches.

# %% [markdown]
# ## Step 1 — Keep your capstone scenario
#
# Use the **same scenario** you chose in Checkpoint 1.1.
#
# | Scenario | Corpus | Retrieval considerations |
# |---|---|---|
# | **Research Paper Navigator** | ~150 research-paper PDFs (`Labs/CapstoneDatasets/ResearchPapers/`) | long documents; you'll likely chunk them; questions often name a specific paper or compare papers. |
# | **Wikipedia Retrieval Engine** | ~2,400 Wikipedia HTML articles (`Labs/CapstoneDatasets/Wikipedia/`) | many short-to-medium articles; questions name a figure/place or span several articles. |
#
# A good baseline is keyword (BM25), semantic (embeddings + vector search), or a
# hybrid of both — exactly what you built in Labs 1.2–2.2.

# %% [markdown]
# ## Setup (~5 min)
#
# 1. **Python 3.11 or 3.12**
# 2. `pip install langchain-openai langchain-core python-dotenv langchain-chroma rank-bm25 beautifulsoup4'
# 3. Use the OpenRouter API key provided for this program. This checkpoint uses
#  the `openai/gpt-5.4-mini` model, with usage covered by the course credits. (this uses the paid gpt-5.4-mini chat model — covered by your course credits — and a keyword retriever, no embeddings).
# 4. Create a `.env` file next to this script: `OPENROUTER_API_KEY=sk-or-v1-...`
#
# This version:
# 1. Loads Wikipedia HTML articles from a folder named ``Wikipedia``.
# 2. Uses the custom prompts and system prompt that were used in Checkpoint 1.1.
# 3. Retrieves context with a hybrid of BM25 keyword search and Chroma vector
#    search, following the approach demonstrated in Lab 2.2.
#
# The first run creates ``chroma_wikipedia_db`` and embeds the article chunks.
# Later runs reuse that database. I need to delete the database folder if the corpus or
# chunking settings change and I need to rebuild the index.
# %%

from __future__ import annotations

import os
import re
import sys
import warnings
from datetime import datetime
from pathlib import Path

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from rank_bm25 import BM25Okapi

warnings.filterwarnings("ignore")
load_dotenv()


# %%
# Model and API settings
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = "openai/gpt-5.4-mini"
EMBEDDING_MODEL = "openai/text-embedding-3-small"
TEMPERATURE = 0.2

# Retrieval settings
CHROMA_DIR = "chroma_wikipedia_db"
NUM_RETRIEVED = 4
CANDIDATE_POOL = 10
WEIGHT_BM25 = 0.5
WEIGHT_VECTOR = 0.5

# Chunking keeps long Wikipedia articles within embedding/context limits.
CHUNK_SIZE = 2500
CHUNK_OVERLAP = 300

SCENARIO = "wikipedia"
LOG_PATH = Path.cwd() / "checkpoint_2_1_retrieval.log"

# This is the same system prompt used in Checkpoint 1.1.
SYSTEM_PROMPT = (
    "You are a helpful research assistant. Answer the user's question as "
    "accurately as you can. When you reference a specific document, quote "
    "from it directly. If you are not certain, say so rather than guessing."
)


# %%
def check_api_key() -> str:
    """Load and return the OpenRouter key, or stop with a clear error."""
    load_dotenv()
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Put the key provided for this "
            "course in a .env file next to this script and rerun."
        )
    return key


def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=check_api_key(),
        base_url=OPENROUTER_BASE_URL,
    )


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
            Path("Labs") / "CapstoneDatasets" / "Wikipedia",
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


def split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping chunks, preferring paragraph boundaries."""
    if overlap >= chunk_size:
        raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE.")

    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        if end < len(text):
            # Prefer ending at a newline; otherwise try a sentence or a space.
            search_start = start + chunk_size // 2
            boundary = text.rfind("\n", search_start, end)
            if boundary == -1:
                boundary = text.rfind(". ", search_start, end)
                if boundary != -1:
                    boundary += 1
            if boundary == -1:
                boundary = text.rfind(" ", search_start, end)
            if boundary > start:
                end = boundary

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break
        start = max(end - overlap, start + 1)

    return chunks


def load_wikipedia_chunks(wikipedia_dir: Path) -> list[Document]:
    """Load every HTML article and convert it into source-labelled chunks."""
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
        relative_source = str(path.relative_to(wikipedia_dir))

        for chunk_number, chunk in enumerate(
            split_text(article_text, CHUNK_SIZE, CHUNK_OVERLAP)
        ):
            chunk_id = f"{relative_source}::chunk-{chunk_number}"
            page_content = f"Article title: {title}\n\n{chunk}"
            documents.append(
                Document(
                    page_content=page_content,
                    metadata={
                        "source": relative_source,
                        "title": title,
                        "chunk": chunk_number,
                        "chunk_id": chunk_id,
                    },
                )
            )

    if not documents:
        raise RuntimeError("The Wikipedia files were found but contained no text.")

    print(f"Loaded {len(paths)} articles as {len(documents)} searchable chunks.")
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
    """Load the persisted vector DB, or embed the corpus on the first run."""
    chroma_path = Path(CHROMA_DIR)
    embeddings = get_embeddings()

    if chroma_path.is_dir() and any(chroma_path.iterdir()):
        print(f"Loading existing vector database from {CHROMA_DIR}/")
        return Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings,
        )

    print("Building vector database (the first run may take several minutes)...")
    database = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )
    print(f"Indexed {len(documents)} chunks in {CHROMA_DIR}/")
    return database


# %%
class HybridWikipediaRetriever:
    """Combine BM25 keyword rankings with Chroma vector rankings."""

    def __init__(self, documents: list[Document], database: Chroma):
        self._documents = documents
        self._database = database
        self._content_by_id = {
            document.metadata["chunk_id"]: document.page_content
            for document in documents
        }
        self._metadata_by_id = {
            document.metadata["chunk_id"]: document.metadata
            for document in documents
        }
        self._bm25 = BM25Okapi(
            [tokenize(document.page_content) for document in documents]
        )
        print(f"Hybrid retriever ready over {len(documents)} chunks.")

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
                    document.metadata["chunk_id"],
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
                document.metadata["chunk_id"],
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
        for (chunk_id, content, metadata, _), score in zip(
            bm25_results, bm25_scores
        ):
            content_by_id[chunk_id] = content
            metadata_by_id[chunk_id] = metadata
            bm25_normalized[chunk_id] = score

        # Chroma returns distance, where a lower value means a better match.
        vector_scores = normalize(
            [result[3] for result in vector_results], invert=True
        )
        for (chunk_id, content, metadata, _), score in zip(
            vector_results, vector_scores
        ):
            content_by_id[chunk_id] = content
            metadata_by_id[chunk_id] = metadata
            vector_normalized[chunk_id] = score

        fused = [
            (
                chunk_id,
                content,
                metadata_by_id[chunk_id],
                WEIGHT_BM25 * bm25_normalized.get(chunk_id, 0.0)
                + WEIGHT_VECTOR * vector_normalized.get(chunk_id, 0.0),
            )
            for chunk_id, content in content_by_id.items()
        ]
        fused.sort(key=lambda result: result[3], reverse=True)
        return fused[:k]

    def retrieved_context(self, query: str) -> tuple[str, list[tuple[str, float]]]:
        results = self.get_top_k(query, NUM_RETRIEVED)
        context_parts = []
        evidence = []

        for _, content, metadata, score in results:
            source = metadata["source"]
            chunk = metadata["chunk"]
            context_parts.append(
                f"[Source: {source}, chunk {chunk}]\n{content}"
            )
            evidence.append((f"{source} (chunk {chunk})", score))

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
        "Use the following retrieved Wikipedia passages as your source material. "
        "If they do not contain enough information, say that you are not certain "
        "rather than using outside knowledge. Identify the source article when "
        "you quote it.\n\n"
        f"Retrieved Wikipedia passages:\n{context}\n\n"
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

    documents = load_wikipedia_chunks(wikipedia_dir)
    database = build_or_load_vector_db(documents)
    retriever = HybridWikipediaRetriever(documents, database)
    llm = make_llm()
    prompts = my_probe_prompts()

    print("Capstone Checkpoint 2.1 — Wikipedia hybrid retrieval")
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

            response = answer_question(llm, prompt, context)
        except Exception as error:
            print(f"\nERROR: {error}\n")
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
# ## Step 5 — Your written submission (the graded deliverable)
#
# Use your completed retrieval implementation and test results to complete the Capstone Checkpoint 2.1
# worksheet. In the worksheet, you will document your retrieval approach, provide evidence that your 
# system is functioning, include 3–5 representative queries and retrieved results, and reflect on where
# your approach performs well and where it struggles.  
 
# Save your completed Python file in the appropriate checkpoint folder in your GitHub repository. 
# Upload the completed worksheet only to the learning platform as your graded submission.
