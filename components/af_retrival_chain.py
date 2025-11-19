from langchain_core.load import dumps,loads
from ad_embeddings import retriever
from ae_generation import generate_queries , llm
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from prompts import basic_prompt

def get_unique_union(documents: list[list]):
    """ Unique union of retrieved docs """
    # Flatten list of lists, and convert each Document to string
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    # Get unique documents
    unique_docs = list(set(flattened_docs))
    # Return
    return [loads(doc) for doc in unique_docs]


question = "What is task decomposition for LLM agents?"
retrieval_chain = generate_queries | retriever.map() | get_unique_union
# docs = retrieval_chain.invoke({"question":question})
# print(len(docs))
# RAG


final_rag_chain = (
    {"context": retrieval_chain, 
     "question": itemgetter("question")} 
    | basic_prompt
    | llm
    | StrOutputParser()
)

print(final_rag_chain.invoke({"question":question}))