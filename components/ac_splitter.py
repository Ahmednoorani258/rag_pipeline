from langchain_text_splitters import RecursiveCharacterTextSplitter
from sampledoc import docs


text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=500,
    chunk_overlap=100,
)

splits= text_splitter.split_documents(docs)

# print(f"Number of splits: {len(splits)}")
# for i, split in enumerate(splits):
#     print(f"--- Split {i} ---")
#     print(split.page_content)
#     print()
#     print(split.metadata)
#     print()