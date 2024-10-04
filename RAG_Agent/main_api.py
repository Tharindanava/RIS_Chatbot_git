import tkinter as tk
from tkinter import scrolledtext, filedialog, messagebox
from tkinter import filedialog, simpledialog, messagebox
import shutil
from tkinter import Menu
from graph import build_graph
import os
from pprint import pprint

# Paths and constants
DATA_PATH = "C:/Users/acer/Documents/Accedemic_Folder_E19254/Training_02_TIEC_docs/RIS_project/RAG_Agent/data/"
URLS_FILE = "C:/Users/acer/Documents/Accedemic_Folder_E19254/Training_02_TIEC_docs/RIS_project/RAG_Agent/urls.txt"

# Function to build the graph
app_graph = None
def initialize_graph():
    global app_graph
    os.environ["LANGCHAIN_TRACING_V2"] = 'True'
    os.environ["LANGCHAIN_ENDPOINT"] = 'https://api.smith.langchain.com'
    os.environ["LANGCHAIN_API_KEY"] = "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"
    os.environ["TAVILY_API_KEY"] = "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"
    app_graph = build_graph()

# Function to process the user's question
def ask_question(question):
    inputs = {"question": question}
    try:
        for output in app_graph.stream(inputs):
            for key, value in output.items():
                pprint(f"Finished running: {key}:")
                pprint(value)

                if key == "generate" and "generation" in value:
                    return value["generation"]

        return "No valid generation found."
    except Exception as e:
        return f"Error during processing: {str(e)}"
    
# Function to load URLs from the text file
def load_urls():
    if os.path.exists(URLS_FILE):
        with open(URLS_FILE, "r") as f:
            return [line.strip() for line in f.readlines() if line.strip()]
    return []

# Function to save URLs to the text file
def save_urls(urls):
    with open(URLS_FILE, "w") as f:
        for url in urls:
            f.write(url + "\n")

# Tkinter interface
class ChatBotApp(tk.Tk):
    def __init__(self):
        super().__init__()
        
        self.title("ChatBot Interface")
        self.geometry("600x600")  # Initial size

        # Make the window resizable
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=0)
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=0)

        # Chat window (Scrollable text)
        self.chat_window = scrolledtext.ScrolledText(self, wrap=tk.WORD, height=20, state='disabled')
        self.chat_window.grid(column=0, row=0, padx=10, pady=10, columnspan=2, sticky="nsew")

        # User input entry (Resizable text area)
        self.user_input = tk.Text(self, height=4)
        self.user_input.grid(column=0, row=1, padx=10, pady=10, sticky="ew")

        # Square Send button (Resizing)
        self.send_button = tk.Button(self, text="Send", command=self.send_message, width=10, height=4)
        self.send_button.grid(column=1, row=1, padx=5, pady=5, sticky='ns')

        # Menu bar for file options
        self.create_menu()

    def create_menu(self):
        # Create a menu bar
        menu_bar = Menu(self)

        # Create File menu and add commands
        file_menu = Menu(menu_bar, tearoff=0)
        file_menu.add_command(label="New Chat", command=self.new_chat)
        file_menu.add_command(label="Open Chat", command=self.open_chat)
        file_menu.add_command(label="Save Chat", command=self.save_chat)
        file_menu.add_command(label="Save Chat As...", command=self.save_chat_as)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.quit)
        menu_bar.add_cascade(label="File", menu=file_menu)

        edit_menu = tk.Menu(menu_bar, tearoff=0)
        edit_menu.add_command(label="Add Data File", command=self.add_pdf_file)
        edit_menu.add_command(label="Edit URL File", command=self.edit_urls)
        menu_bar.add_cascade(label="Edit", menu=edit_menu)

        # Attach the menu bar to the app
        self.config(menu=menu_bar)

    def send_message(self):
        user_message = self.user_input.get("1.0", tk.END).strip()
        
        # Loop back if empty string is passed
        if not user_message:
            messagebox.showwarning("Input Error", "Please enter a message before sending.")
            return  # Stop the function if no message is provided

        self.display_message(f"You: {user_message}")

        # Clear the input box
        self.user_input.delete("1.0", tk.END)

        # Get the bot's answer
        bot_answer = ask_question(user_message)
        self.display_message(f"Bot: {bot_answer}")

    def display_message(self, message):
        self.chat_window.config(state='normal')
        self.chat_window.insert(tk.END, message + '\n')
        self.chat_window.yview(tk.END)
        self.chat_window.config(state='disabled')

    def new_chat(self):
        """Open a new chat window."""
        self.chat_window.config(state='normal')
        self.chat_window.delete(1.0, tk.END)
        self.chat_window.config(state='disabled')
        messagebox.showinfo("New Chat", "Started a new chat session.")

    def open_chat(self):
        """Open an existing chat log."""
        file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
        if file_path:
            with open(file_path, 'r') as file:
                content = file.read()
                self.chat_window.config(state='normal')
                self.chat_window.delete(1.0, tk.END)
                self.chat_window.insert(tk.END, content)
                self.chat_window.config(state='disabled')

    def save_chat(self):
        """Save the current chat session."""
        file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text Files", "*.txt")])
        if file_path:
            self.save_chat_to_file(file_path)

    def save_chat_as(self):
        """Save the current chat session to a new file."""
        self.save_chat()

    def save_chat_to_file(self, file_path):
        """Helper function to save chat to file."""
        content = self.chat_window.get(1.0, tk.END)
        with open(file_path, 'w') as file:
            file.write(content)
        messagebox.showinfo("Chat Saved", f"Chat saved to {file_path}")

    def add_pdf_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("PDF Files", "*.pdf")])
        if file_path:
            dest_path = os.path.join(DATA_PATH, os.path.basename(file_path))
            shutil.copy(file_path, dest_path)
            messagebox.showinfo("Success", f"PDF added to {DATA_PATH}")

    def edit_urls(self):
        urls = load_urls()
        if not urls:
            messagebox.showwarning("No URLs", "No URLs found to edit.")
            return

        # Create a new window for editing URLs
        edit_window = tk.Toplevel(self)
        edit_window.title("Edit URLs")
        edit_window.geometry("400x300")

        # URL listbox
        url_listbox = tk.Listbox(edit_window, selectmode=tk.SINGLE, height=10, width=50)
        for url in urls:
            url_listbox.insert(tk.END, url)
        url_listbox.pack(pady=10)

        def delete_url():
            selected = url_listbox.curselection()
            if selected:
                index = selected[0]
                del urls[index]
                save_urls(urls)  # Save the updated URLs back to the file
                url_listbox.delete(index)

        def add_new_url():
            new_url = simpledialog.askstring("Input", "Enter a new URL:")
            if new_url:
                urls.append(new_url)
                save_urls(urls)
                url_listbox.insert(tk.END, new_url)

        # Buttons for editing URLs
        delete_button = tk.Button(edit_window, text="Delete URL", command=delete_url)
        delete_button.pack(side=tk.LEFT, padx=10)

        add_button = tk.Button(edit_window, text="Add URL", command=add_new_url)
        add_button.pack(side=tk.RIGHT, padx=10)

# Initialize the graph and run the chat interface
if __name__ == "__main__":
    initialize_graph()  # Load the graph
    app = ChatBotApp()  # Create chat interface
    app.mainloop()      # Run the Tkinter app
