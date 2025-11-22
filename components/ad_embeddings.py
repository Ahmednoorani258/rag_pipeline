from sampledoc import splits  # list[Document]
# from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

vectorstore = Chroma.from_documents(
    documents=splits,
    embedding = embeddings,
)


# 4) Retriever and sample query
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
# print("Retriever ready")
# print(retriever.vectorstore)
# results = retriever.invoke("What is task decomposition?")

# print(f"Retrieved {len(results)} chunks")
# for i, d in enumerate(results, 1):
#     print(f"--- Hit {i} ---")
#     print(d.page_content[:400].replace("\n", " "))
#     print(d.metadata)