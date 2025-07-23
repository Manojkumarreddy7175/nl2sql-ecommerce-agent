# NL2SQL E-Commerce AI Agent

Hey everyone! I'm Manoj Kumar Reddy, a passionate coder who spends a lot of time solving algorithmic puzzles on LeetCode and optimizing Java code for efficiency. This project is my take on building an AI agent that turns natural language questions into SQL queries for e-commerce data analysis. I evaluated options like a custom Mistral setup via Ollama against RAG-based approaches (e.g., Vanna AI) and stuck with the custom one for its local, lightweight performance and full control. The agent queries datasets on ad sales, total sales, and eligibility, providing meaningful English summaries and actionable insights.

## Project Overview
This FastAPI application serves as an intelligent backend for e-commerce data querying. It uses a local LLM (Mistral via Ollama) to generate optimized SQLite SELECT queries from natural language inputs. No cloud dependencies—everything runs locally. I focused on making summaries readable and natural (e.g., "The total items are 264."), drawing from my experience in crafting efficient algorithms to ensure quick responses.

Key datasets:
- **Ad Sales**: Tracks ad_sales, impressions, ad_spend, clicks, etc.
- **Total Sales**: Covers total_sales and units_ordered.
- **Eligibility**: Includes eligibility status and messages.

It handles metrics like RoAS and CPC with built-in business logic.

## Features
- **NL2SQL Conversion**: Translates questions like "Count of total items" into safe, optimized SQL.
- **Meaningful Summaries**: Generates natural English responses (e.g., "The total sales are 1234.56.") for console output.
- **Chart Generation**: Creates bar charts for multi-row results (base64-encoded in responses).
- **Conversation Memory**: Retains context for follow-up questions (limited to 3 entries for efficiency).
- **Validation & Security**: Ensures only SELECT queries, prevents division by zero, and sanitizes outputs.
- **Performance Optimizations**: Prompt caching, regex extraction, and history limiting for fast processing.
- **Server-Side Console Output**: Prints human-readable answers directly in CMD for easy debugging.

## Architecture
Here's a visual represntation of the system's architecture, grouped by components (User, API, Ollama).

![AI-ecommerce -agent architecure ](https://github.com/user-attachments/assets/f6b95947-75e8-4e48-a4dd-3f3c7ce73a4b)

### User Processes
1. Formulate and send query via POST to /ask.
2. Receive JSON response and view results.

### API Processes (FastAPI Backend)
1. Validate request.
2. Build prompt with schema, examples, and history.
3. Generate SQL via Ollama.
4. Validate and execute SQL on SQLite DB.
5. Process results (summary, chart).
6. Update memory and print to console.
7. Return JSON.

### Ollama Processes (Local LLM)
1. Receive prompt from API.
2. Generate SQL response.
3. Return to API for processing.

Full flow: User → FastAPI (prompt build → Ollama → DB query → processing) → Response back to User + Console Print.

## Installation
1. **Clone the Repo**:
git clone https://github.com/Manojkumarreddy7175/nl2sql-ecommerce-agent.git
cd nl2sql-ecommerce-agent

text

2. **Install Dependencies**:

pip install  FastAPI, Uvicorn, OpenAI, Pandas, Matplotlib

3. **Set Up Ollama**:
- Install from [ollama.ai](https://ollama.ai).
- Pull Mistral: `ollama pull mistral`.
- Run Ollama server.

4. **Database**:
- Use the included `ecommerce.db` or create one matching the schema.
- (Optional) Populate with your CSV data.

## Usage
1. **Start the Server**:

uvicorn main:app --reload

text
- Runs on `http://localhost:8000`.

2. **Query the API**:

curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d '{"question": "Count of total items", "session_id": "default"}'

text

- JSON Response Example:
  ```
  {
    "question": "Count of total items",
    "sql": "SELECT COUNT(DISTINCT ads.item_id) AS total_items FROM ad_sales ads",
    "result": [{"total_items": 264}],
    "summary": "The total items are 264.",
    "chart_base64": null,
    "message": "Success"
  }
  ```

- Console Output (Server CMD):
  ```
  Question: Count of total items
  SQL Query: SELECT COUNT(DISTINCT ads.item_id) AS total_items FROM ad_sales ads
  Summary: The total items are 264.
  Message: Success
  ```

## Examples
- **Input**: "What is my total sales?"
- Summary: "The total sales are 1234.56."
- **Input**: "Calculate the RoAS (Return on Ad Spend)."
- Summary: "The roas is 7.92."
- **Input**: "Top 3 items by ad spend."
- Summary: "Top 5 results: 123: 500.00, 456: 450.00, 789: 400.00" (with chart if applicable).

## How It Works
- **Prompt Engineering**: Combines schema, rules, and examples for accurate SQL generation.
- **SQL Processing**: Validates, executes on SQLite, and formats results into natural language.

- **Why This Tech?**: Local Mistral for privacy/speed; FastAPI for lightweight API; no RAG to keep it simple and custom.





– Manoj Kumar Reddy
