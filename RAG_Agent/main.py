from graph import build_graph
import os
import subprocess
from pprint import pprint

def main():

    # Set environment variables
    os.environ["LANGCHAIN_TRACING_V2"] = 'True'
    os.environ["LANGCHAIN_ENDPOINT"] = 'https://api.smith.langchain.com'
    os.environ["LANGCHAIN_API_KEY"] = "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"
    os.environ["TAVILY_API_KEY"] = "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"

    # Build the graph
    app = build_graph()

    # Test with a question
    inputs = {"question": "Who is the current president of Sri Lanka"}
    try:
        for output in app.stream(inputs):
            for key, value in output.items():
                pprint(f"Finished running: {key}:")
                pprint(value)  # Print the value for debugging

                # Once a valid generation is found, exit the loop
                if key == "generate" and "generation" in value:
                    print("Valid answer generated. Exiting the process.")
                    break  # Exit the main function once the answer is generated

        print(value.get("generation", "No generation key found in output"))

    except Exception as e:
        print(f"Error during processing: {e}")

if __name__ == "__main__":
    main()