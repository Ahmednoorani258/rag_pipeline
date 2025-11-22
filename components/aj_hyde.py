from ae_generation import generate_docs_for_retrieval_hyde, retriever,llm
from langchain_core.output_parsers import StrOutputParser
from prompts import basic_prompt

question = "What is task decomposition for LLM agents?"
# generate_docs_for_retrieval_hyde.invoke({"question":question})

# Retrieve
retrieval_chain = generate_docs_for_retrieval_hyde | retriever 
retrieved_docs = retrieval_chain.invoke({"question":question})


final_rag_chain = (
    basic_prompt
    | llm
    | StrOutputParser()
)

final_rag_chain.invoke({"context":retrieved_docs,"question":question})