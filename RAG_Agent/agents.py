from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

def create_llm_json():
  
    local_llm = "C:/Users/acer/Documents/Accedemic_Folder_E19254/Training_02_TIEC_docs/RIS_project/RAG_Agent/model/Llama-3.1-Storm-8B-Q5_K_S.gguf"

    # Callbacks support token-wise streaming
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

    llm = LlamaCpp(
        model_path=local_llm,
        temperature=0,
        n_ctx=4096, 
        n_gpu_layers=12,
        n_batch = 1024,
        callback_manager=callback_manager,
        verbose=True,  # Verbose is required to pass to the callback manager
        )

    return llm

def create_retrival_grader():
    llm = create_llm_json()

    # Retrieval grader agent
    retrival_grader = PromptTemplate(
        template ="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether
    a document is useful to resolve a question. Give a binary score 'yes' or 'no' score to indicate
    wheater the document is useful to resolve the question. Provide the binary score as a JSON with a
    single key 'score', like this: {{"score": "yes"}} or {{"score": "no"}} and no preamble or explanation.
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    Here is the document:
    \n ------- \n
    {document}
    \n ------- \n
    Here is the question: {question} <|eot_id|><start_header_id|>assistant<|end_header_id|>
    """,
    input_variables=["question", "document"]
    ) | llm | JsonOutputParser()

    return retrival_grader

def create_rag_chain():
    llm = create_llm_json()

    # RAG chain agent
    rag_chain = PromptTemplate(
        template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know.
    Give discriptive answers if requested else give concise answers. Provide the answer as a JSON with a single key 'generation', no preamble or explanation.
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    Question: {question}
    Context: {context}
    Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """,
    input_variables=["question","context"]
    ) | llm | StrOutputParser()

    return rag_chain

def create_hallucination_grader():
    llm = create_llm_json()

    # Hallucination grader agent
    hallucination_grader = PromptTemplate(
        template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether
    an answer is grounded in / supported by a set of facts. Give a binary score 'yes' or 'no' score to indicate
    wheater the answer is grounded in / supported by facts. Provide the binary score as a JSON with a
    single key 'score', like this: {{"score": "yes"}} or {{"score": "no"}} and no preamble or explanation.
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    Here are the facts:
    \n ------- \n
    {documents}
    \n ------- \n
    Here is the answer: {generation} <|eot_id|><start_header_id|>assistant<|end_header_id|>
    """,
    input_variables=["generation","document"]
    ) | llm | JsonOutputParser()

    return hallucination_grader

def create_answer_grader():
    llm = create_llm_json()

    # Answer grader agent
    answer_grader = PromptTemplate(
        template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether
    an answer is useful to resolve a question. Give a binary score 'yes' or 'no' score to indicate
    wheater the answer is useful to resolve the question. Provide the binary score as a JSON with a
    single key 'score', like this: {{"score": "yes"}} or {{"score": "no"}} and no preamble or explanation.
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    Here is the answer:
    \n ------- \n
    {generation}
    \n ------- \n
    Here is the question: {question} <|eot_id|><start_header_id|>assistant<|end_header_id|>
    """,
    input_variables=["question","generation"]
    ) | llm | JsonOutputParser()

    return answer_grader

def create_web_search_tool():

    # Tavily web search tool
    from langchain_community.tools.tavily_search import TavilySearchResults
    web_search_tool = TavilySearchResults(k=3)

    return web_search_tool
