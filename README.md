# 31_rag_pipeline (Gemini + Chroma + HF embeddings)

A minimal Retrieval-Augmented Generation (RAG) pipeline using:
- Loader: WebBaseLoader (BeautifulSoup-strained HTML)
- Splitter: RecursiveCharacterTextSplitter (tiktoken-aware)
- Embeddings: sentence-transformers/all-MiniLM-L6-v2 (HuggingFace)
- Vector store: Chroma
- LLM: Google Gemini via langchain-google-genai
- Orchestration: LangChain LCEL

Directory
- components/
  - aa_tokenCounter.py
  - ab_retriever.py
  - ac_splitter.py
  - ad_embeddings.py
  - ae_generation.py
  - af_retrival_chain.py
  - prompts.py
  - sampledoc.py

Quick start (Windows, uv)
1) Create .env at repo root (do NOT quote keys)
   GOOGLE_API_KEY=your_gemini_api_key
   LANGCHAIN_TRACING_V2=false

2) Install deps
   uv add langchain-core langchain-community langchain-text-splitters \
  langchain-google-genai langchain-huggingface sentence-transformers \
# 31_rag_pipeline — Detailed README

This repository is a compact, componentized Retrieval-Augmented Generation (RAG) example that demonstrates: web/document loading, token-aware splitting, creating embeddings, building a Chroma vector store, multi-query retrieval, Reciprocal Rank Fusion (RRF) reranking, and using Google Gemini as the LLM for grounded generation via LangChain LCEL constructs.

This README explains project structure, how each component works, how to run the code, configuration details, and recommended changes for open-source consumption.

Table of contents
- Project overview
- Files & responsibilities
- Setup & dependencies
- Running the pipeline (examples)
- How the multi-query + RRF flow works
- Tuning RRF (`k`) and expected effects
- Troubleshooting & common issues
- Development notes, tests & contributing
- License suggestion

Project overview
----------------
The repo demonstrates a small RAG pipeline split into focused modules under `components/`. The data flow is:

  - Document load (live web scraping example or `sampledoc.py`) -> split into chunks -> embed -> store in Chroma -> create a retriever.
  - Query rewriting creates multiple candidate queries. Each query is executed against the retriever producing multiple ranked lists. These lists are fused using Reciprocal Rank Fusion (RRF) to form a single ranked candidate set. The top fused documents are passed into a final prompt and LLM to produce a grounded answer.

          Files & responsibilities
          ------------------------
          components/ (core modules)
          - `aa_tokenCounter.py` — small helper (cosine similarity helper in repo; not central to pipeline).
          - `ab_retriever.py` — example Web loader using `WebBaseLoader` with BeautifulSoup's `SoupStrainer` to limit parsed HTML (produces `blog_docs`). Live scraping example; optional.
          - `ac_splitter.py` — uses `RecursiveCharacterTextSplitter.from_tiktoken_encoder(...)` to split Documents token‑aware into `splits` (chunk_size=500, overlap=100).
          - `ad_embeddings.py` — embeds `splits` with HuggingFace (`all-MiniLM-L6-v2`) and creates a Chroma vectorstore; exports `retriever`.
          - `ae_generation.py` — sets up Gemini LLM runnable (`llm`), the question-rewriting runnables (`generate_queries`, `generate_queries_fusion`), and a basic RAG runnable. Uses `langchain_core` LCEL runnables (prompts, parsers, passthroughs).
          - `af_retrival_chain.py` — single-query RAG example: rewrites -> `retriever.map()` -> dedupe -> `basic_prompt` -> `llm` -> print answer.
          - `ag_rag_fusion.py` — multi-query retrieval + Reciprocal Rank Fusion (RRF) implementation; fuses multiple ranked lists then runs final prompt pipeline.
          - `prompts.py` — central place for `basic_prompt`, `questions_generator_prompt`, and `prompt_rag_fusion` templates.
          - `sampledoc.py` — development convenience: in-repo `docs` and `splits` populated from a long article (Lilian Weng blog post) so examples run without Internet.

          Setup & dependencies
          --------------------
          1) Create a project `.env` (root of repo) and DO NOT commit it. Example:

          ```dotenv
          GOOGLE_API_KEY=your_gemini_api_key
          LANGCHAIN_TRACING_V2=false
          # Optional if using LangSmith tracing with an org-scoped key:
          # LANGSMITH_WORKSPACE_ID=your_workspace_id
          ```

          2) Install (recommended minimal list):

          ```cmd
          pip install langchain-core langchain-google-genai langchain-community langchain-huggingface
          pip install chromadb tiktoken sentence-transformers python-dotenv beautifulsoup4
          ```

          Notes
          - This repo mixes `langchain_core` (LCEL) runnables and some community connectors; package names and import paths change between LangChain versions. If you see import errors, try installing `langchain-core` and `langchain-community`.
          - If you cannot use Gemini or don't have a Google API key, you can switch `ad_embeddings.py` to use Hugging Face embeddings (already configured) and `ae_generation.py` to a local LLM (HuggingFace) or OpenAI (if you have a key).

          Running the pipeline (examples)
          ------------------------------
          - Build splits only (uses `sampledoc` by default):

          ```cmd
          python components/ac_splitter.py
          ```

          - Build embeddings + retriever (ensure `ac_splitter.py` exports `splits`):

          ```cmd
          python components/ad_embeddings.py
          ```

          - Run the single-query RAG example (rewrites -> dedupe -> answer):

          ```cmd
          python components/af_retrival_chain.py
          ```

          - Run the multi-query + RRF fusion example (fusion -> answer):

          ```cmd
          python components/ag_rag_fusion.py
          ```

          How the multi-query + RRF flow works (short)
          ------------------------------------------
          1. `generate_queries_fusion` (from `ae_generation.py`) expands the user question into multiple search queries (e.g., 4–5 rewrites).
          2. `retriever.map()` runs the retriever for each rewritten query returning a list-of-lists: `[[docs_for_q1], [docs_for_q2], ...]`.
          3. `reciprocal_rank_fusion(results, k=60)` takes these ranked lists and computes a fused score for each unique document using the formula:

             contribution per appearance = 1 / (rank + k)

             where `rank` is the 0-based position of the document within a particular list and `k` is a tunable offset.

          4. Documents are sorted by fused score and the top-k are returned for grounding the final prompt.

          Why RRF?
          - RRF is rank-based (doesn't rely on raw similarity score scales) and rewards items that are consistently ranked well across multiple rewrites. It's robust when retrieval scores are not comparable across queries or when different retrievers are combined.

          Choosing `k` (tuning)
          ---------------------
          - `k` controls how much advantage top positions get.
          - Examples:
            - `k=5` (small): rank differences matter a lot. Top‑1 positions provide large boosts.
            - `k=60` (default in this repo): a balance between top-position preference and consensus.
            - `k=200` (large): reduces top‑position dominance; favors consensus across lists.

          Recommendation: run a small sweep (5, 20, 60, 150) on sample queries and inspect fused top-10 to pick a good `k`.

          Troubleshooting & common issues
          ------------------------------
          - `GOOGLE_API_KEY` is None / TypeError: Make sure `.env` exists and `python-dotenv` is installed; call `load_dotenv()` before creating the Gemini client. Avoid assigning environment variables from None.
          - LangSmith / tracing errors about org-scoped keys: either set `LANGSMITH_WORKSPACE_ID` or disable tracing with `LANGCHAIN_TRACING_V2=false` in `.env`.
          - Pylance import warnings (`langchain.prompts` etc.): prefer `langchain_core.prompts` and `langchain_core.load` imports for newer LangChain packages; install `langchain-core`.
          - Empty vector store: ensure `ad_embeddings.py` imports `splits` from `ac_splitter.py` (so splits are generated). If `splits` is empty, the Chroma collection will be empty.
          - Only five questions printed (no final answer): ensure you're using the `basic_prompt` (with `{context}`) for the final answer, not the rewrite prompt. Also ensure final runnable chains are wired correctly in `af_retrival_chain.py` / `ag_rag_fusion.py`.
          - Deduping instability: `dumps(doc)` is used to create keys — this can be brittle if metadata order changes. Prefer a stable key (e.g., `doc.metadata['id']` or SHA1 of `doc.page_content`).

          Development notes & recommendations
          -----------------------------------
          - Make `k` configurable in `ag_rag_fusion.py` (function parameter or env var).
          - Replace `dumps(doc)` dedupe with a stable key function:

          ```py
          import hashlib

          def doc_key(doc):
              return doc.metadata.get('id') or doc.metadata.get('source') or hashlib.sha1(doc.page_content.encode('utf-8')).hexdigest()
          ```

          - Add debug prints during development:

          ```py
          results = retrieval_chain_rag_fusion.invoke({"question": question})
          print([len(lst) for lst in results])  # per-query retrieval sizes
          ```

          - Limit outputs in production (e.g., only return top-10 fused docs) for speed and clarity.

          Contributing & open source readiness
          ------------------------------------
          - Add a `CONTRIBUTING.md` with how-tos for running the project locally and how to submit PRs.
          - Add a `CODE_OF_CONDUCT.md` (e.g., Contributor Covenant) if you plan to accept community contributions.
          - Add tests for critical parts (e.g., unit tests for `reciprocal_rank_fusion`, test data for retrieval format).
          - Add a `requirements.txt` or `pyproject.toml` with pinned versions used during development.

          License
          -------
          This project currently has no license file. For open source distribution consider one of:
          - `MIT` (permissive)
          - `Apache-2.0` (permissive + patent grant)

          Add `LICENSE` with your preferred license and note it at the top of the README.

          Appendix — small code pointers
          -----------------------------
          - Use `load_dotenv()` at script start in `ae_generation.py` and other entrypoint scripts.
          - Ensure LCEL runnables return the shapes you expect: `generate_queries_fusion` should return `list[str]`, and `retriever.map()` should return `list[list[Document]]`.
          - If you want to replace Gemini with another LLM, update `ae_generation.py` to create a different `llm` and keep runnables the same.

          If you'd like, I can: 
          - apply the stable-key + debug print patch to `ag_rag_fusion.py`,
          - add a small unit test for `reciprocal_rank_fusion`, or
          - produce a small demo script that shows the effect of different `k` values on synthetic ranked lists.

          ---
          Last updated: automatic README generation based on project files.
