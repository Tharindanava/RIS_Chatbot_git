from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl

DB_FAISS_PATH = "vectorstores/db_faiss"

custom_prompt_template = """Use the following pieces of information to answer the user's questions.
If you do not know the answer, please say that you don't know the answer, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vector store.
    """
    prompt = PromptTemplate(
        template=custom_prompt_template, 
        input_variables=['context', 'question']
    )
    return prompt

def load_llm():
    llm = CTransformers(
        model="llama-2-7b-chat.ggmlv3.q2_K.bin",
        model_type="llama",
        max_new_tokens=2048,
        temperature=0.5
    )
    return llm

def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={'k': 2}),
        chain_type_kwargs={'prompt': prompt}
    )
    return qa_chain

def qa_bot():
    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        model_kwargs={'device': 'cpu'}
    )
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)
    return qa

def final_result(query):
    qa_result = qa_bot()
    response = qa_result({'query': query})
    return response

# Chainlit

@cl.on_chat_start
async def start():
    chain = qa_bot()
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hi, Welcome to the RIS assistant. How can I help you today?"
    await msg.update()
    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message):
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    res = await chain.ainvoke({'query': message.content}, callbacks=[cb])
    answer = res["result"]

    sources = res.get("source_documents", [])
    source_texts = [doc.page_content for doc in sources]

    if sources:
        answer += f"\nSources: " + ", ".join(source_texts)
    else:
        answer += f"\nNo Sources Found"

    await cl.Message(content=answer).send()
