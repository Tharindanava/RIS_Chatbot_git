from graph import build_graph
import os
import subprocess
from pprint import pprint

def is_llama_model_loaded():
    """Check if the llama3 model is already loaded by using the ollama list command."""
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=True)
        if 'llama3' in result.stdout:
            print("Llama3 model is already loaded.")
            return True
        else:
            print("Llama3 model not found in ollama list.")
            return False
    except FileNotFoundError:
        print("The 'ollama' command was not found. Ollama is not installed.")
        return False
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while checking for the llama model: {e}")
        return False

def install_ollama():
    """Install Ollama using curl command."""
    try:
        # Run the curl command to install Ollama
        subprocess.run("curl -fsSL https://ollama.com/install.sh | sh", shell=True, check=True)
        print("Ollama installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to install Ollama: {e}")
        raise

def main():
    # # Check if llama3 is already loaded
    # if not is_llama_model_loaded():
    #     # Install Ollama if it's not installed
    #     install_ollama()

    #     # Start Ollama in the background
    #     subprocess.Popen(["ollama", "serve"])
    #     print("Ollama server started in the background.")

    #     # Pull the llama3 model
    #     try:
    #         subprocess.run(["ollama", "pull", "llama3"], check=True)
    #         print("Llama3 model pulled successfully.")
    #     except subprocess.CalledProcessError as e:
    #         print(f"Error pulling the llama3 model: {e}")
    #         return

    # Set environment variables
    os.environ["LANGCHAIN_TRACING_V2"] = 'True'
    os.environ["LANGCHAIN_ENDPOINT"] = 'https://api.smith.langchain.com'
    os.environ["LANGCHAIN_API_KEY"] = "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"
    os.environ["TAVILY_API_KEY"] = "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"

    # Build the graph
    app = build_graph()

    # Test with a question
    inputs = {"question": "what is an mri machine"}
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
