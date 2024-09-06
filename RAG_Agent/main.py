from vectorstore import create_vectorstore
from agents import create_agents
from methods import retrieve, grade_documents, generate, web_search
from graph import build_graph
import os

os.environ["LANGCHAIN_TRACING_V2"] = 'True'
os.environ["LANGCHAIN_ENDPOINT"] = 'https://api.smith.langchain.com'
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["TAVILY_API_KEY"] = "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"

def main():
    # Set up vectorstore
    retriever = create_vectorstore()

    # Initialize agents
    retrival_grader, rag_chain, hallucination_grader, answer_grader, web_search_tool = create_agents(retriever)

    # Build the graph
    app = build_graph(retrival_grader, rag_chain, hallucination_grader, answer_grader, web_search_tool)

    # Test with a question
    inputs = {"question": "who won the last test match between Sri Lanka and India"}
    for output in app.stream(inputs):
        for key, value in output.items():
            print(f"Finished running: {key}:")
            print(value["generation"])

if __name__ == "__main__":
    main()