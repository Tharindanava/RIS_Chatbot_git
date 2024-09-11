from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict
from typing import List
from methods import retrieve, grade_documents, generate, web_search, decide_to_generate, grade_generation_v_documents_and_question

class GraphState(TypedDict):
  """
  Represents the state of our graph.

  Attributes:
    question: question
    generation: LLM generation
    web_search: whether to add search
    documents: List of documents
  """
  question: str
  generation: str
  web_search: str
  documents: List [str]

def build_graph():
    workflow = StateGraph(GraphState)

    #Define the nodes
    workflow.add_node("websearch", web_search)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)

    # Build graph
    # Clear existing start point and edges if they exist.
    start_node = '__start__'

    # If a path from `__start__` exists, remove it.
    if start_node in workflow.nodes:
        connected_edges = list(workflow.edges.keys())
        for edge in connected_edges:
            if edge[0] == start_node:
                workflow.remove_edge(edge[0], edge[1])
        workflow.remove_node(start_node)

    # Set new entry point and build the workflow
    workflow.set_entry_point("retrieve") # Use set_entry_point to designate the starting node.
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "websearch": "websearch",
            "generate": "generate"
        }
    )
    workflow.add_edge("websearch", "generate")
    workflow.add_conditional_edges(
        "generate",
        grade_generation_v_documents_and_question,
        {
            "not supported": "generate",
            "useful": END,
            "not useful": "websearch"
        }
    )

    return workflow.compile()

