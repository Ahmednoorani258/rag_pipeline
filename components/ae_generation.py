import os
from dotenv import load_dotenv
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "false"

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from sampledoc import docs
from ad_embeddings import retriever
from prompts import questions_generator_prompt, basic_prompt, prompt_rag_fusion, prompt_queries_decomposition,stepback_prompt



# Build context text from Documents
context_text = "\n\n---\n\n".join(d.page_content for d in docs)

# prompt = ChatPromptTemplate.from_template(
#     "Answer the question based only on the following context.\n{context}\n\nQuestion: {question}"
# )

# chain = prompt | llm

# response = chain.invoke({
#     "context": context_text,
#     "question": "What is ai?"
# })
# print(response)


llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash",temperature=0,)

# Chains
rag_chain = ({"context": retriever, "question": RunnablePassthrough()} | basic_prompt | llm | StrOutputParser())
generate_queries = (questions_generator_prompt | llm | StrOutputParser() | (lambda x: x.split("\n")))
generate_queries_fusion = ( prompt_rag_fusion  | llm | StrOutputParser()  | (lambda x: x.split("\n")))
generate_queries_decomposition = ( prompt_queries_decomposition  | llm | StrOutputParser()  | (lambda x: x.split("\n")))
generate_queries_step_back = stepback_prompt | llm | StrOutputParser()
# res = rag_chain.invoke("What is short term memory?")
# print(res)
