# NL2SQL E-Commerce Agent

A FastAPI-powered backend that translates natural language questions into safe, optimized SQL queries for e-commerce analytics. It leverages a local [Ollama Mistral](https://ollama.com/library/mistral) large language model via the OpenAI-compatible API, and visualizes results with summaries and charts.

---

## Features

- **Natural Language to SQL**: Converts user questions into secure, valid SQLite `SELECT` queries.
- **LLM Integration**: Uses Ollama Mistral (OpenAI API compatible) for query generation.
- **Strict SQL Safety**: Only allows `SELECT` statements, with schema and business-rule awareness.
- **Session Memory**: Maintains limited conversation history for each user session.
- **Result Summaries**: Provides clear, human-readable insights for query results.
- **Chart Generation**: Automatically creates bar charts (as base64 images) for suitable results.

---

## Architecture

![NL2SQL Architecture](docs/nl2sql-architecture.gif)



![AI-ecommerce -agent architecure ](https://github.com/user-attachments/assets/f6b95947-75e8-4e48-a4dd-3f3c7ce73a4b)


<sup><em>If the GIF does not display, ensure you have a file at `docs/nl2sql-architecture.gif` or replace the path with your actual GIF location.</em></sup>

---

## Database Schema

Predefined tables in `ecommerce.db`:

- **ad_sales**  
  `date TEXT (YYYY-MM-DD)`, `item_id INT`, `ad_sales FLOAT`, `impressions INT`, `ad_spend FLOAT`, `clicks INT`, `units_sold INT`
- **total_sales**  
  `date TEXT (YYYY-MM-DD)`, `item_id INT`, `total_sales FLOAT`, `total_units_ordered INT`
- **eligibility**  
  `eligibility_datetime_utc TEXT`, `item_id INT`, `eligibility INT`, `message TEXT`

**Business Metrics Supported:**
- RoAS (Return on Ad Spend): `SUM(ad_sales) / NULLIF(SUM(ad_spend), 0)`
- CPC (Cost Per Click): `SUM(ad_spend) / NULLIF(SUM(clicks), 0)`

---

## Getting Started

### Prerequisites

- Python 3.7+
- [Ollama](https://ollama.com/) (with [Mistral model](https://ollama.com/library/mistral) running locally)
- `ecommerce.db` SQLite database with the schema above

### Install Dependencies

```bash
pip install fastapi openai matplotlib pandas uvicorn
```

### Start Ollama (Mistral)

```bash
ollama run mistral
```

### Run the FastAPI Server

```bash
uvicorn main:app --reload
```

The server will be available at:  
`http://localhost:8000`

---

## API Usage

### POST `/ask`

**Request Body:**
```json
{
  "question": "Which product had the highest CPC?",
  "session_id": "user123"
}
```

**Response Example:**
```json
{
  "question": "...",
  "sql": "...",
  "result": [...],
  "summary": "...",
  "chart_base64": "...",
  "message": "Success"
}
```

- `chart_base64` is a PNG image encoded as a base64 string (for two-column results).

---

## Customization

- **LLM Configuration**: Adjust `openai.api_base` and `openai.api_key` as needed for your Ollama instance.
- **Schema Extension**: Update the prompt and database schema in `main.py` to add new tables or metrics.

---

## Security Notes

- Only SELECT queries are generated and executed.
- All LLM outputs are filtered and validated before execution.
- Conversation memory is session-specific and non-persistent.

---

## License

MIT License

---

## Acknowledgments

- [Ollama](https://ollama.com/) for local LLM hosting.
- [FastAPI](https://fastapi.tiangolo.com/) for API framework.

---

<sup>For questions or contributions, please open an issue or pull request.</sup>![Uploading image.png…]()
