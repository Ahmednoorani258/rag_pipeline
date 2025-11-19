from operator import itemgetter
from ae_generation import generate_queries_decomposition, retriever,llm
from langchain_core.output_parsers import StrOutputParser
from prompts import decomposition_prompt, basic_template

# ____________________________________________________________________
#     Case:1 - Question answers depending on previous Q&A pairs
# ____________________________________________________________________

def format_qa_pair(question, answer):
    """Format Q and A pair"""
    
    formatted_string = ""
    formatted_string += f"Question: {question}\nAnswer: {answer}\n\n"
    return formatted_string.strip()

q_a_pairs = ""

question = "What are the main components of an LLM-powered autonomous agent system?"
questions = generate_queries_decomposition.invoke({"question":question})
for q in questions:
    decomposition_rag_chain = (
    {"context": itemgetter("question") | retriever, 
     "question": itemgetter("question"),
     "q_a_pairs": itemgetter("q_a_pairs")} 
    | decomposition_prompt
    | llm
    | StrOutputParser())

    answer = decomposition_rag_chain.invoke({"question":q,"q_a_pairs":q_a_pairs})
    q_a_pair = format_qa_pair(q,answer)
    q_a_pairs = q_a_pairs + "\n---\n"+  q_a_pair
    # print(answer)
