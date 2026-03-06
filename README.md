# Movie Maven Chatbot

A production-grade CLI Chatbot that answers questions about 45,000+ movies using **LangGraph**, **Google Gemini**, and **FAISS**.

## Features

- **Structured Queries**: Accurate answers for budget, revenue, release dates, and cast.
- **Semantic Search**: Find movies by plot, theme, or vague description (e.g., "movies about time travel").
- **Multi-turn Conversations**: Maintains context (e.g., "Who directed it?").
- **Local Vectors**: Uses FAISS and HuggingFace embeddings (efficient, no extra API costs).

## Prerequisites

- **Dataset**: `movies_metadata.csv`, `keywords.csv`, `credits.csv` in the `dataset/` directory.
- **Google API Key**: Needed for the Chatbot.

---

## Option 1: Run with Docker (Recommended)

This method handles all dependencies and data processing automatically inside a container.

1.  **Setup Environment**
    Create a `.env` file:
    ```env
    GOOGLE_API_KEY=your_api_key_here
    ```

2.  **Run the Bot**
    ```bash
    docker-compose run --rm movie-bot
    ```
    *The container will automatically check for processed data. If missing, it will process the CSVs (takes ~2-3 mins) before starting the chat.*

3.  **Interact**
    The chatbot will start in your terminal. Type `quit` to exit.

---

## Option 2: Run Locally (No Docker)

1.  **Install Python 3.11+**

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Setup Environment**
    Create a `.env` file:
    ```env
    GOOGLE_API_KEY=your_api_key_here
    ```

4.  **Process Data** (One-time setup)
    ```bash
    python src/data_processor.py
    ```

5.  **Run the App**
    ```bash
    python main.py
    ```

## Architecture
- **LLM**: Google Gemini 1.5 Flash.
- **Agent**: LangGraph StateGraph.
- **Vector Store**: FAISS with `all-MiniLM-L6-v2` embeddings.
