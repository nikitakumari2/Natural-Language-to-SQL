# Natural Language-to-SQL Copilot

This project provides a powerful tool for converting natural language questions into executable SQL queries. It's designed to make data insights accessible to everyone, regardless of their SQL knowledge, by leveraging AI to generate, validate, and interpret queries against real databases.

## Features

  * **Natural Language Input:** Simply ask a question in plain English.
  * **Automatic Schema Recognition:** Uses a vector search to intelligently identify relevant database schemas.
  * **AI-Powered Query Generation:** Converts your question into a valid SQL query.
  * **Query Execution & Analysis:** Runs the generated query and provides a clear, formatted result.
  * **Flexible Backend:** Built with FastAPI for a robust and scalable API.
  * **Interactive Frontend:** A user-friendly interface powered by Streamlit.
  * **Model Agnostic:** Supports both OpenAI and HuggingFace models (e.g., `flan-t5-xxl`).
  * **Spider Benchmark Integration:** Utilizes the [Spider dataset](https://yale-lily.github.io/spider) for training and evaluation.

## ⚙️ Project Structure

```
NL2SQL/
├── .env
├── requirements.txt
├── setup.py
├── README.md
├── src/
│   ├── app/
│   │   ├── backend.py       # FastAPI backend
│   │   ├── frontend.py      # Streamlit frontend
│   │   ├── main.py          # Core SQL Copilot logic
│   │   ├── sql_functions.py # DB logic, embedding, SQL generation
│   │   └── test.py          # Optional test runner
│   └── data/
│       └── raw/             # Place Spider dataset here
│
└── venv/                    # Python virtual environment (optional)
```

## 📦 Installation & Setup

### 1\. 📁 Get the Dataset

This project relies on the **Spider benchmark dataset**.

```bash
# Step 1: Download
wget --content-disposition "https://drive.google.com/uc?export=download&id=1TqleXec_OykOYFREKKtschzY29dUcVAQ"

# Step 2: Unzip
unzip spider.zip

# Step 3: Move it under src/data/raw/
mv spider src/data/raw/
```

### 2\. 🛠️ Install Dependencies

Navigate to the project root and install the required packages.

```bash
pip install -r requirements.txt
```

### 3\. 🧠 Vector Database Setup

This project uses a vector database to perform efficient schema searches. You'll need to build the ChromaDB embeddings from your Spider dataset.

```bash
python setup.py
```

*This process may take 10-15 minutes to index all database schemas and sample data.*

### 4\. 🔐 HuggingFace / OpenAI API Key

Create a `.env` file in the root directory to store your API keys.

```bash
touch .env
```

Add your keys to the file:

```dotenv
hf_token=your_huggingface_api_key
OPENAI_API_KEY=your_openai_key_if_used
```

## 🧪 Running the Application

### ✳️ Run Backend (FastAPI)

Open a terminal and start the backend server.

```bash
cd src/app
uvicorn backend:app --reload --host 0.0.0.0 --port 8000
```

### 🎨 Run Frontend (Streamlit)

In a new terminal, run the Streamlit frontend.

```bash
cd src/app
streamlit run frontend.py
```

The application will be available in your browser at `http://localhost:8501`.

## ✏️ Example Workflow

1.  Open the Streamlit frontend in your browser.
2.  Enter a question, such as: *"How many heads of the departments are older than 56?"*
3.  The application will then:
      * Find the most relevant database schemas.
      * Generate a corresponding SQL query.
      * Execute the query against the database.
      * Return a clear and formatted result.

## 📝 Citation & Acknowledgements

  * **LangChain**:
      * Chase, H. (2022). *LangChain GitHub*.
  * **Spider Dataset**:
      * Yu, T., et al. (2018). *Spider: A large-scale human-labeled dataset for complex and cross-domain semantic parsing*. **arXiv:1809.08887**.
