from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

def create_agents(retriever):
    local_llm = 'llama3'
    #LLM
    llm = ChatOllama(model=local_llm, format="json", temparature = 0)

    # Retrieval grader agent
    retrival_grader = PromptTemplate(
        template ="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether
    a document is useful to resolve a question. Give a binary score 'yes' or 'no' score to indicate
    wheater the document is useful to resolve the question. Provide the binary score as a JSON with a
    single key 'score' and no preamble or explanation.
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    Here is the document:
    \n ------- \n
    {document}
    \n ------- \n
    Here is the question: {question} <|eot_id|><start_header_id|>assistant<|end_header_id|>
    """,
        input_variable=["question", "document"]
    ) | llm | JsonOutputParser()

    # RAG chain agent
    rag_chain = PromptTemplate(
        template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know.
    Use six sentences maximum to keep the answer concise
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    Question: {question}
    Context: {context}
    Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """,
    input_variable=["question","document"]
    ) | llm | StrOutputParser()

    # Hallucination grader agent
    hallucination_grader = PromptTemplate(
        template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether
    an answer is grounded in / supported by a set of facts. Give a binary score 'yes' or 'no' score to indicate
    wheater the answer is grounded in / supported by facts. Provide the binary score as a JSON with a
    single key 'score' and no preamble or explanation.
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    Here are the facts:
    \n ------- \n
    {documents}
    \n ------- \n
    Here is the answer: {generation} <|eot_id|><start_header_id|>assistant<|end_header_id|>
    """,
    input_variable=["question","document"]
    ) | llm | JsonOutputParser()

    # Answer grader agent
    answer_grader = PromptTemplate(
        template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether
    an answer is useful to resolve a question. Give a binary score 'yes' or 'no' score to indicate
    wheater the answer is useful to resolve the question. Provide the binary score as a JSON with a
    single key 'score' and no preamble or explanation.
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    Here is the answer:
    \n ------- \n
    {generation}
    \n ------- \n
    Here is the question: {question} <|eot_id|><start_header_id|>assistant<|end_header_id|>
    """,
    input_variable=["question","document"]
    ) | llm | JsonOutputParser()

    # Tavily web search tool
    from langchain_community.tools.tavily_search import TavilySearchResults
    web_search_tool = TavilySearchResults(k=3)

    return retrival_grader, rag_chain, hallucination_grader, answer_grader, web_search_tool
