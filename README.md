# MedMind: AI Backend API (LangServe) 🤖

This repository contains the **AI Backend API** for the MedMind project. It is a high-performance, asynchronous service built with **FastAPI** and **LangServe**, designed to host and expose specialized LLM (Large Language Model) chains for the frontend application to consume.

## 🧠 AI Capabilities (LangServe Endpoints)

The API utilizes LangChain to orchestrate specialized AI chains, primarily using **Ollama** for local, open-source LLM inference (e.g., MedGemma).

| Path | Description | Chain Type | Input Key |
| :--- | :--- | :--- | :--- |
| `/chat` | General Medical Chatbot with conversational memory. | `chat_chain_with_memory` | `query` |
| `/symptom_checker` | Analyzes symptoms to provide preliminary health information. | `medgemma_symptoms_chain` | `query` |
| `/lab_report_analysis` | Interprets and explains medical lab report results. | `medgemma_lab_report_chain` | Custom (`LabReportInput`) |
| `/mental_health_support` | Empathetic and supportive conversational agent with memory. | `mental_health_chain_with_memory` | `query` |

## 🛠️ Tech Stack

* **Package Manager:** [uv](https://docs.astral.sh/uv/) (Astral)
* **API Framework:** FastAPI
* **LLM Orchestration:** LangChain and LangServe
* **LLM Inference:** Ollama (local server)

## 🔗 Project Dependency (Frontend Web App)

This API is the intelligence core for the MedMind web application. It is crucial that this service is running before starting the frontend.

* **Frontend Repository:** [https://github.com/smriti2805/MedMind](https://github.com/smriti2805/MedMind)

## 🚀 Setup & Installation

### Prerequisites
1.  **uv:** This project uses `uv` for fast package management.
    * [**Install uv (Official Guide)**](https://docs.astral.sh/uv/getting-started/installation/)
2.  **Ollama:** Must be installed and running on your system to serve the LLMs.
    * [Download Ollama](https://ollama.com/)
    * Pull the required model (e.g., `ollama pull medgemma` or `ollama pull gemma:2b`).

### Installation Steps

1.  **Clone this repository:**
    ```bash
    git clone [https://github.com/kishandev2509/MedMind---Backend](https://github.com/kishandev2509/MedMind---Backend)
    cd MedMind---Backend
    ```

2.  **Sync Dependencies:**
    Use `uv` to automatically create the environment and install dependencies:
    ```bash
    uv sync
    ```

3.  **Run the Server:**
    Start the backend service using `uv`:
    ```bash
    uv run main.py
    ```
    The `lifespan` function will automatically check for and attempt to start the Ollama server before adding the LangServe routes.

    * **API URL:** `http://localhost:8000`
    * **Interactive Docs:** `http://localhost:8000/docs`
