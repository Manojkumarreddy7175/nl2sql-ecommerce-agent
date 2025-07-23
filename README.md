# NL2SQL E-Commerce AI Agent

A local, privacy-first AI backend that translates natural language questions into optimized SQLite queries for e-commerce analytics. Powered by FastAPI and a locally running [Mistral LLM via Ollama](https://ollama.ai), this agent enables secure, cloud-free querying and returns clean summaries and optional charts. The codebase emphasizes safety (only `SELECT` queries), performance, and natural result explanations.

---

## Features

- **Natural Language to SQL:** Converts user questions into secure, validated SQLite `SELECT` queries via a local Mistral LLM.
- **Human-Readable Summaries:** Uses custom logic to summarize query results, e.g. `"The total sales are 1234.56."`
- **Chart Generation:** For 2-column, multi-row results, returns a base64-encoded bar chart image.
- **Session Memory:** Remembers up to 3 previous queries per session for context-aware follow-ups.
- **Strict Validation & Security:** Only `SELECT` queries are allowed; prevents division by zero and cleans SQL output.
- **Server Console Logging:** Logs every request, SQL, summary, and chart status directly to the server console for visibility.
- **Performance:** Uses prompt caching, efficient regex, and memory limiting to ensure fast, resource-light operation.

---

## Code Structure

- `main.py` – FastAPI app, request model, `/ask` endpoint, SQL execution, summary and chart generation, server-side logging.
- `prompts.py` – Builds the LLM prompt (schema, business logic, examples, rules), with prompt caching for speed.
- `sql_generator.py` – Sends prompt and question to Mistral, aggressively extracts only the SQL from LLM output.
- `utils.py` – Natural language summary logic and matplotlib-based chart generator.
- `memory.py` – Session-limited conversation memory (max 3 entries) for context-aware LLM prompts.
- `config.py` – (Not shown here) Handles DB connection and OpenAI/Ollama client config.

---

## Database Schema

The agent expects a SQLite database (`ecommerce.db`) with the following tables:

- **ad_sales:**  
  `date TEXT (YYYY-MM-DD)`, `item_id INT`, `ad_sales FLOAT`, `impressions INT`, `ad_spend FLOAT`, `clicks INT`, `units_sold INT`
- **total_sales:**  
  `date TEXT (YYYY-MM-DD)`, `item_id INT`, `total_sales FLOAT`, `total_units_ordered INT`
- **eligibility:**  
  `eligibility_datetime_utc TEXT`, `item_id INT`, `eligibility INT`, `message TEXT`

**Business Metrics Handled in SQL:**
- RoAS = `SUM(ad_sales) / NULLIF(SUM(ad_spend), 0)`
- CPC = `SUM(ad_spend) / NULLIF(SUM(clicks), 0)`

---

## Architecture
Visual Representation 
![AI-ecommerce -agent architecture](https://github.com/user-attachments/assets/f6b95947-75e8-4e48-a4dd-3f3c7ce73a4b)
```
User (HTTP POST /ask)
      |
      v
FastAPI Backend (main.py)
  - Validates input
  - Builds prompt (prompts.py, memory.py)
  - Calls local LLM (Mistral via Ollama, sql_generator.py)
  - Validates/extracts SQL
  - Executes on SQLite (config.py)
  - Summarizes (utils.py)
  - Optionally charts (utils.py)
  - Logs to server console
      |
      v
User receives JSON result (summary, SQL, result, chart_base64, message)

```
---

## Installation

1. **Clone the Repo:**
   ```bash
   git clone https://github.com/Manojkumarreddy7175/nl2sql-ecommerce-agent.git
   cd nl2sql-ecommerce-agent
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *(Requires FastAPI, Uvicorn, OpenAI, pandas, matplotlib, etc.)*

3. **Set Up Ollama with Mistral:**
   - Download Ollama from [ollama.ai](https://ollama.ai)
   - Pull Mistral: `ollama pull mistral`
   - Start Ollama: `ollama run mistral` or background service

4. **Prepare the Database:**
   - Use provided `ecommerce.db` or create a matching file.
   - Optionally populate with your data.

---

## Usage

1. **Start the FastAPI server:**
   ```bash
   uvicorn main:app --reload
   # By default runs at http://localhost:8000
   ```

2. **Issue a Query:**
   ```bash
   curl -X POST http://localhost:8000/ask \
     -H "Content-Type: application/json" \
     -d '{"question": "Count of total items", "session_id": "default"}'
   ```

   **Sample JSON Response:**
   ```json
   {
     "question": "Count of total items",
     "sql": "SELECT COUNT(DISTINCT ads.item_id) AS total_items FROM ad_sales ads",
     "result": [{"total_items": 264}],
     "summary": "The total items are 264.",
     "chart_base64": null,
     "message": "Success"
   }
   ```

   **Sample Server Console:**
   ```
   Question: Count of total items
   SQL Query: SELECT COUNT(DISTINCT ads.item_id) AS total_items FROM ad_sales ads
   Summary: The total items are 264.
   Message: Success
   ```

---

## Example Questions & Summaries

- **Q:** "What is my total sales?"  
  **A:** "The total sales are 1234.56."
- **Q:** "Calculate the RoAS (Return on Ad Spend)."  
  **A:** "The roas is 7.92."
- **Q:** "Top 3 items by ad spend."  
  **A:** "Top 3 results: 123: 500.00, 456: 450.00, 789: 400.00" (plus chart)

---

## How It Works (Code-backed)

- The `/ask` endpoint receives a question and session ID, logs the request, and builds a prompt using schema, examples, and recent history.
- `sql_generator.py` sends the prompt and input to Mistral (via OpenAI-compatible API), extracting *only* the SQL with regex. Only safe `SELECT` queries are ever executed.
- SQLite executes the query; results are summarized in natural English via `utils.py` (with logic for single/multi-row and two-column results).
- If the result is multi-row and two-column, a matplotlib bar chart is generated and returned as a base64 string.
- All responses and summaries are logged to the server console for transparency and debugging.


*Built with code-level efficiency, security, and clarity in mind.*

– Manoj Kumar Reddy 
