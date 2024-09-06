from langchain.schema import document

def retrieve(state, retriever):
  """
  Retrieve documents form vectorstore

  Arg:
    state (dict): The current graph state

  Returns:
    state (dict): New key added to state, documents, that contains retrieved documents

  """
  print("---RETRIEVE---")
  question = state["question"]

  #Retrieval
  documents = retriever.invoke(question)
  return {"documents": documents, "question": question}

def grade_documents(state, retrival_grader):
    """
    Determines whether the retrieved documents are relevent to the question
    If any document is not relevant, we will set a flag to run web search

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelavent documents and update web_search state

    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    #Score each doc
    filtered_docs = []
    web_search = "No"

    for d in documents:
        score = retrival_grader.invoke({"question": question, "document":d.page_content})
        grade = score['score']
        #Document relevent
        if grade.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        #Document not relavent
        else:
            print("---GRADE: DOCUMENT IRRELEVANT---")
            web_search = "Yes"
        continue

    return {"documents": filtered_docs, "question": question, "web_search": web_search}

def generate(state, rag_chain):
    """
    Genarate answer using RAG on retrieved documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation

    """
    print("---GENARATE---")
    question = state["question"]
    documents = state["documents"]

    #RAG generation
    generation = rag_chain.invoke({"context":documents, "question":question})
    return {"documents": documents, "question":question, "generation": generation}

def web_search(state, web_search_tool):
    """
    Web search based on the grade of documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Appended web results to documents

    """

    print("---WEB SEARCH---")
    question = state["question"]
    documents = state["documents"]

    #Web search
    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = document(page_content=web_results)
    if documents is not None:
        documents.append(web_results)
    else:
        docuemnts = [web_results]

    return {"documents": documents, "question": question}

### Conditional Nodes

def decide_to_generate(state):
  """
  Determines whether to genarate an answer, or add web search

  Args:
    state (dict): The current graph state

  Returns:
    str: Binary decision for next node to call

  """

  print("---ASSESS GRADED DOCUMENTS---")
  question = state["question"]
  web_search = state["web_search"]
  filtered_documents = state["documents"]

  if web_search == "Yes":
    print("---DESION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION WEB SEARCH---")
    return "websearch"

  else:
    print("---DESION: GENERATE---")
    return "generate"

def grade_generation_v_documents_and_question(state, hallucination_grader, answer_grader):
  """
  Determines whether or not the answer genarate is relevant to the question

  Args:
    state (dict): The current graph state

  Returns:
    str: Binary decision for next node to call

  """

  print("---CHECK HALLUCINATION---")
  question = state["question"]
  documents = state["documents"]
  generation = state["generation"]

  # Calculate 'grade' before using it
  score = hallucination_grader.invoke({"documents": documents, "generation": generation})
  grade = score['score']

  if grade == "yes":
    print("---DECISION: generation IS GROUNDED IN DOCUMENTS---")
    print("---GRADE generation vs QUESTION---")
    score = answer_grader.invoke({"question": question, "generation": generation})
    grade = score['score']
    if grade == "yes":
      print("---DECISION: GEANRATION ADDRESSES QUESTION---")
      return "useful"
    else:
      print("---DECISION: generation DOSE NOT ADDRESS QUESTION---")
      return "not useful"

  else:
    print("---DECISION: generation IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
    return "not supported"
