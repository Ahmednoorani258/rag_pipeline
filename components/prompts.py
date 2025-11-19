from langchain_core.prompts import ChatPromptTemplate

basic_template = """Answer the following question based on this context:

{context}

Question: {question}
"""


questions_generator_template = """You are an AI language model assistant. Your task is to generate five 
different versions of the given user question to retrieve relevant documents from a vector 
database. By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of the distance-based similarity search. 
Provide these alternative questions separated by newlines. Original question: {question}"""


rag_fusion_template = """You are a helpful assistant that generates multiple search queries based on a single input query. \n
Generate multiple search queries related to: {question} \n
Output (4 queries):"""


# Decomposition
decomposition_queries_template = """You are a helpful assistant that generates multiple sub-questions related to an input question. \n
The goal is to break down the input into a set of sub-problems / sub-questions that can be answers in isolation. \n
Generate multiple search queries related to: {question} \n
Output: only give (3 queries/questions):"""


# Prompt
decomposition_template = """Here is the question you need to answer:

\n --- \n {question} \n --- \n

Here is any available background question + answer pairs:

\n --- \n {q_a_pairs} \n --- \n

Here is additional context relevant to the question: 

\n --- \n {context} \n --- \n

Use the above context and any background question + answer pairs to answer the question: \n {question}
"""


questions_generator_prompt = ChatPromptTemplate.from_template(questions_generator_template)
basic_prompt = ChatPromptTemplate.from_template(basic_template)
prompt_rag_fusion = ChatPromptTemplate.from_template(rag_fusion_template)
prompt_queries_decomposition = ChatPromptTemplate.from_template(decomposition_queries_template)
decomposition_prompt = ChatPromptTemplate.from_template(decomposition_template)