# Have some error sith parent retriever need to resolve

import uuid


from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.stores import InMemoryByteStore
from ad_embeddings import embeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import ParentDocumentRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter

from sampledoc import docs
from ae_generation import llm

# Build summarization chain
chain = (
    {"doc": lambda x: x.page_content}
    | ChatPromptTemplate.from_template("Summarize the following document:\n\n{doc}")
    | llm
    | StrOutputParser()
)

summaries = chain.batch(docs, {"max_concurrency": 5})

# Child vectorstore (summaries)
vectorstore = Chroma(collection_name="summaries", embedding_function=embeddings)

# Parent document store
store = InMemoryByteStore()
id_key = "doc_id"

child_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

# Parent retriever
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    byte_store=store,
    child_splitter=child_splitter,
    id_key=id_key
)

doc_ids = [str(uuid.uuid4()) for _ in docs]

# Add summary docs
summary_docs = [
    Document(page_content=summaries[i], metadata={id_key: doc_ids[i]})
    for i in range(len(summaries))
]

vectorstore.add_documents(summary_docs)

# Store parent docs
retriever.byte_store.mset(list(zip(doc_ids, docs)))

# Query
query = "Memory in agents"
retrieved_docs = retriever(query)  # ← now callable, no n_results

print(f"[info] Retrieved {len(retrieved_docs)} documents")
if retrieved_docs:
    print("[info] Preview of first document (500 chars):")
    print(retrieved_docs[0].page_content[:500].replace("\n", " "))
