# Multiflow CrewAI Project

This project demonstrates a multi-crew setup using CrewAI, where user intent is analyzed to route tasks to specialized crews:
- `DocSearchCrew`: For searching and retrieving information from documents.
- `AnalystCrew`: For performing data analysis tasks, potentially using SQL databases.

The application uses an intent analysis agent to determine the nature of a user's query and then dispatches the query to the appropriate crew.

## Project Structure

-   `multiflow/`
    -   `pyproject.toml`: Defines project metadata, dependencies, and build system.
    -   `.env`: (You need to create this) Stores API keys and environment-specific configurations.
    -   `src/`
        -   `multiflow/`
            -   `main.py`: Main entry point for the application. Handles intent analysis and routing.
            -   `crews/`
                -   `doc_search_crew/`: Contains the `DocSearchCrew` implementation.
                    -   `crew.py`: Defines the agents, tasks, and crew for document searching.
                    -   `config/`: YAML files for agent and task definitions for `DocSearchCrew`.
                    -   `tools/`: Tools used by `DocSearchCrew` (e.g., Pinecone tools).
                -   `analyst_crew/`: Contains the `AnalystCrew` implementation.
                    -   `crew.py`: Defines the agents, tasks, and crew for data analysis.
                    -   `config/`: YAML files for agent and task definitions for `AnalystCrew`.
                    -   `tools/`: Tools used by `AnalystCrew` (e.g., NL2SQLTool, file tools).
            -   `knowledge/`: Stores knowledge base files for `DocSearchCrew`.

## Setup

1.  **Python Version:**
    Ensure you have Python `>=3.10` and `<3.12` installed.

2.  **Create a Virtual Environment:**
    It's highly recommended to use a virtual environment.
    ```bash
    python -m venv .venv
    ```
    Activate the environment:
    -   Windows (PowerShell):
        ```powershell
        . .\venv\Scripts\Activate.ps1
        ```
    -   macOS/Linux:
        ```bash
        source .venv/bin/activate
        ```

3.  **Install Dependencies:**
    With the virtual environment activated, install the project dependencies from the root directory of the `multiflow` project:
    ```bash
    pip install .
    ```
    This command reads the `pyproject.toml` file and installs all necessary packages.

4.  **Set Up Environment Variables:**
    Create a `.env` file in the root of the `multiflow` project directory (e.g., `f:\dev\crewai_agent\multiflow\.env`).
    Add the following environment variables with your actual keys and settings:

    ```env
    OPENAI_API_KEY="your_openai_api_key_here"

    # For DocSearchCrew (Pinecone)
    PINECONE_API_KEY="your_pinecone_api_key_here"
    PINECONE_ENV="your_pinecone_environment_here" # e.g., "gcp-starter" or "us-west1-gcp" etc.
    PINECONE_INDEX_NAME="your_pinecone_index_name_here"

    # For AnalystCrew (Optional - if using a specific database for NL2SQLTool)
    # If not set, AnalystCrew will default to an in-memory SQLite database.
    # DB_URI="postgresql://user:password@host:port/database"
    # DB_URI="mysql+mysqlconnector://user:password@host:port/database"
    # DB_URI="sqlite:///path/to/your/database.db"
    ```

    **Note:** The `AnalystCrew` currently uses an in-memory SQLite database by default if `DB_URI` is not provided. Its `FileTool` and `DirectorySearchTool` operate on the local filesystem.

## Running the Application

With your virtual environment activated and the `.env` file configured, run the application from the `multiflow` project root:

```bash
python src/multiflow/main.py
```
or
```bash
python -m src.multiflow.main
```

The application will:
1.  Load environment variables from `.env`.
2.  Use a sample query defined in `src/multiflow/main.py`.
3.  Analyze the intent of the query.
4.  Route the query to the appropriate crew (`DocSearchCrew` or `AnalystCrew`).
5.  The selected crew will process the query and print its results to the console.

You can modify the `sample_query` in `src/multiflow/main.py` (within the `IntentAnalysisFlow` class, `get_user_query` method) to test different scenarios and routing paths.
