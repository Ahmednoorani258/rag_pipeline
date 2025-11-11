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
          chromadb tiktoken python-dotenv beautifulsoup4

3) Run the final chain
   uv run components/af_retrival_chain.py

Environment
- GOOGLE_API_KEY: Gemini API key from Google AI Studio.
- Optional: Disable LangSmith tracing globally with LANGCHAIN_TRACING_V2=false. If you enable it and use an org-scoped key, set LANGSMITH_WORKSPACE_ID to avoid errors.

Pipeline overview
1) Load documents (ab_retriever.py)
   - Fetches https://lilianweng.github.io/posts/2023-06-23-agent/
   - Uses bs4.SoupStrainer to keep only post-content/title/header.
   - Produces blog_docs: list[Document].
   - In this project, docs are provided by components/sampledoc.py.

2) Split documents (ac_splitter.py)
   - RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=500, overlap=100)
   - Produces splits: list[Document] (metadata preserved).
   - Token-aware splitting reduces truncation and improves retrieval quality.

3) Build embeddings and vector store (ad_embeddings.py)
   - Uses HuggingFaceEmbeddings("all-MiniLM-L6-v2")
   - Chroma.from_documents(documents=splits, embedding=…)
   - Exposes retriever = vectorstore.as_retriever(k=3)
   Important: Ensure ad_embeddings imports splits from ac_splitter:
   from ac_splitter import splits

4) Query generation and LLM (ae_generation.py)
   - llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
   - questions_generator_prompt: rewrites the original question into 5 alternatives.
   - generate_queries chain returns a list[str] via StrOutputParser + lambda x: x.split("\n")
   - basic_prompt: your RAG answer template (must include {context} and {question}).

5) Retrieval + answer (af_retrival_chain.py)
   - generate_queries | retriever.map() | get_unique_union
     • Expands query, retrieves for each rewrite, then dedupes Documents.
   - final_rag_chain maps:
     { "context": retrieval_chain -> join docs into text, "question": original }
     -> basic_prompt -> llm -> StrOutputParser
   - Prints a grounded answer.

Why you sometimes only see “five questions”
- If you feed the “rewrite” prompt to the final stage, the LLM will output only the five rewrites.
- Ensure final_rag_chain uses a prompt that references {context} and not the rewrite template.

Expected wiring between files
- ac_splitter.py exports: splits
- ad_embeddings.py imports: from ac_splitter import splits
- ae_generation.py exports: generate_queries and llm
- af_retrival_chain.py imports: generate_queries, retriever, llm, basic_prompt

Example answer prompt (prompts.py)
- basic_prompt should include both placeholders:
  "Use ONLY the context to answer. If unknown, say you don't know.\n\nContext:\n{context}\n\nQuestion: {question}"

Run targets
- Load/split only:
  uv run components/ac_splitter.py
- Build embeddings/retriever:
  uv run components/ad_embeddings.py
- Generate rewrites (debug):
  uv run components/ae_generation.py
- Full RAG answer:
  uv run components/af_retrival_chain.py

Troubleshooting
- Import "langchain.prompts" unresolved:
  Use from langchain_core.prompts import ChatPromptTemplate

- LangSmith error (org-scoped key/workspace):
  Set LANGCHAIN_TRACING_V2=false or provide LANGSMITH_WORKSPACE_ID

- GOOGLE_API_KEY None / TypeError:
  Ensure python-dotenv is installed and you call load_dotenv() before instantiating ChatGoogleGenerativeAI.
  Do not set os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY") when it’s None.

- No retrieval results:
  • Confirm ad_embeddings imports splits from ac_splitter
  • Ensure sentence-transformers/all-MiniLM-L6-v2 downloads (internet access)
  • Increase k or relax SoupStrainer in ab_retriever if using live scraping

- Only five questions printed:
  Use the answer prompt (basic_prompt) with {context} in af_retrival_chain.py, not the rewrite prompt.

Security
- Never commit real API keys. Rotate any keys you previously exposed in .env.
- Prefer unquoted values in .env (e.g., LANGCHAIN_TRACING_V2=false).