from ae_generation import generate_queries_step_back ,retriever,llm
from langchain_core.output_parsers import StrOutputParser
from prompts import stepback_response_prompt
from langchain_core.runnables import RunnableLambda

question = "What is task decomposition for LLM agents?"
# generate_queries_step_back.invoke({"question": question})


chain = (
    {
        # Retrieve context using the normal question
        "normal_context": RunnableLambda(lambda x: x["question"]) | retriever,
        # Retrieve context using the step-back question
        "step_back_context": generate_queries_step_back | retriever,
        # Pass on the question
        "question": lambda x: x["question"],
    }
    | stepback_response_prompt
    | llm
    | StrOutputParser()
)

print(chain.invoke({"question": question}))