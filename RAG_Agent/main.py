from graph import build_graph
import os

os.environ["LANGCHAIN_TRACING_V2"] = 'True'
os.environ["LANGCHAIN_ENDPOINT"] = 'https://api.smith.langchain.com'
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["TAVILY_API_KEY"] = "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"

def main():

    # Build the graph
    app = build_graph()

    # Test with a question
    from pprint import pprint
    inputs = {"question": "who won the last test match between Sri Lanka and India"}
    for output in app.stream(inputs):
      for key, value in output.items():
        pprint(f"Finished running: {key}:")
    print(value["generation"])

if __name__ == "__main__":
    main()