
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sqlite3
import openai
import matplotlib.pyplot as plt
import io
import base64
import pandas as pd
from typing import List, Dict
from functools import lru_cache
import re
import logging


# Set logging level to ERROR to minimize unnecessary logs in CMD
logging.basicConfig(level=logging.ERROR)


# Connect to local Ollama (Mistral)
openai.api_base = "http://localhost:11434/v1"
openai.api_key = "ollama"  # Dummy key, required by openai-python


# Connect to SQLite DB
conn = sqlite3.connect("ecommerce.db", check_same_thread=False)


# FastAPI app
app = FastAPI()


# Request model
class Query(BaseModel):
    question: str
    session_id: str = "default"


# Cached static prompt base to avoid rebuilding
@lru_cache(maxsize=1)
def get_base_prompt() -> str:
    return """
You are an expert AI assistant for generating clean, optimized, and valid SQLite SELECT queries from natural language questions. Follow these instructions precisely:


DATABASE SCHEMA:
- ad_sales (date TEXT (YYYY-MM-DD), item_id INT, ad_sales FLOAT, impressions INT, ad_spend FLOAT, clicks INT, units_sold INT)
- total_sales (date TEXT (YYYY-MM-DD), item_id INT, total_sales FLOAT, total_units_ordered INT)
- eligibility (eligibility_datetime_utc TEXT, item_id INT, eligibility INT, message TEXT)


BUSINESS LOGIC & METRICS:
- RoAS (Return on Ad Spend) = SUM(ad_sales) / NULLIF(SUM(ad_spend), 0)
- CPC (Cost Per Click) = SUM(ad_spend) / NULLIF(SUM(clicks), 0)
- Do NOT add ad_sales and total_sales together; total_sales already includes ad_sales.


BEST PRACTICES & RULES:
- Output ONLY the raw SQL query starting with SELECT. Do not include explanations, comments, markdown, or additional text.
- Only generate a single valid SELECT statement. Never use INSERT, UPDATE, DELETE, DROP, or any non-SELECT statement.
- Use only the tables and columns defined in the schema. Never invent or hallucinate columns or tables.
- Always qualify every column name with its table alias.
- Use JOINs with explicit ON conditions when combining tables, typically on item_id and date.
- Use proper aggregations (SUM, AVG, COUNT, MIN, MAX) and GROUP BY when aggregating.
- Use NULLIF in division to prevent division by zero.
- Use WHERE filters and date ranges ONLY as explicitly specified in the question. Do not add unsolicited filters.
- Use LIMIT N and ORDER BY for top/bottom or sorted results.
- If the question cannot be answered, return exactly: SELECT 'Not possible with current schema.' as message;
"""


# Few-shot examples with RoAS to handle the reported issue
FEW_SHOT_EXAMPLES = [
    {"user": "What is my total sales?", "sql": "SELECT SUM(ts.total_sales) AS total_sales FROM total_sales ts;"},
    {"user": "Which product had the highest CPC (Cost Per Click)?", "sql": "SELECT ads.item_id, (SUM(ads.ad_spend) / NULLIF(SUM(ads.clicks), 0)) AS cpc FROM ad_sales ads GROUP BY ads.item_id ORDER BY cpc DESC LIMIT 1;"},
    {"user": "Calculate the RoAS (Return on Ad Spend).", "sql": "SELECT SUM(ads.ad_sales) / NULLIF(SUM(ads.ad_spend), 0) AS roas FROM ad_sales ads;"},
    {"user": "Top 3 items by ad spend.", "sql": "SELECT ads.item_id, SUM(ads.ad_spend) AS total_ad_spend FROM ad_sales ads GROUP BY ads.item_id ORDER BY total_ad_spend DESC LIMIT 3;"}
]


conversation_memory: Dict[str, List[Dict[str, str]]] = {}


@lru_cache(maxsize=128)
def get_relevant_examples(question: str) -> List[Dict[str, str]]:
    """Select top 2-3 relevant examples for better coverage without slowing down."""
    keywords = set(question.lower().split())
    scored_examples = []
    for ex in FEW_SHOT_EXAMPLES:
        ex_keywords = set(ex["user"].lower().split())
        score = len(keywords.intersection(ex_keywords))
        if score > 0:
            scored_examples.append((score, ex))
    scored_examples.sort(key=lambda x: x[0], reverse=True)
    return [ex for _, ex in scored_examples[:3]]


def build_prompt(question: str, session_id: str) -> str:
    """Build a leaner prompt for faster processing."""
    relevant_examples = get_relevant_examples(question)
    examples_str = "\n".join([f"User: {ex['user']}\nSQL: {ex['sql']}" for ex in relevant_examples])
    history = conversation_memory.get(session_id, [])[-2:]  # Limit to last 2 for brevity
    history_str = "\n".join([f"Previous: {msg['question']} -> {msg['sql']}" for msg in history])
    return f"{get_base_prompt()}\nFew-Shot Examples:\n{examples_str if examples_str else 'None'}\nConversation History:\n{history_str if history_str else 'None'}\nGenerate SQL for: {question}\nOutput ONLY the SQL query."


def generate_sql(system_prompt: str, user_content: str) -> str:
    """Optimized SQL generation with improved post-processing to extract pure SQL."""
    try:
        response = openai.ChatCompletion.create(
            model="mistral",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}],
            temperature=0.0  # Lower temperature for deterministic, faster output
        )
        sql_output = response["choices"][0]["message"]["content"].strip()
        
        # Regex to aggressively extract SELECT statement, ignoring leading text
        match = re.search(r'(SELECT\s+.*?);?\s*$', sql_output, re.IGNORECASE | re.DOTALL | re.MULTILINE)
        if match:
            return match.group(1)
        
        
        cleaned_output = re.sub(r'^[^S]*SELECT', 'SELECT', sql_output, flags=re.IGNORECASE)
        match_fallback = re.search(r'(SELECT\s+.*?);?\s*$', cleaned_output, re.IGNORECASE | re.DOTALL)
        return match_fallback.group(1) if match_fallback else sql_output
    except Exception as e:
        logging.error(f"Error in generate_sql: {str(e)}")
        raise


@app.post("/ask")
async def ask_question(query: Query):
    try:
        advanced_prompt = build_prompt(query.question, query.session_id)
        
        initial_sql = generate_sql(advanced_prompt, query.question)
        
        if not initial_sql.upper().startswith("SELECT") or any(kw in initial_sql.upper() for kw in ["DROP", "INSERT", "UPDATE", "DELETE"]):
            fallback_prompt = f"{get_base_prompt()}\nStrictly output only the SQL for: {query.question}"
            initial_sql = generate_sql(fallback_prompt, query.question)
            if not initial_sql.upper().startswith("SELECT"):
                raise ValueError("Only SELECT queries allowed.")
        
        safe_sql = initial_sql.split(";")[0].strip()
        
        cursor = conn.cursor()
        cursor.execute(safe_sql)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        result = [dict(zip(columns, row)) for row in rows]
        
        if query.session_id not in conversation_memory:
            conversation_memory[query.session_id] = []
        conversation_memory[query.session_id].append({"question": query.question, "sql": safe_sql})
        conversation_memory[query.session_id] = conversation_memory[query.session_id][-3:]
        
        summary = generate_summary(result, columns, query.question)
        chart_base64 = generate_chart(result, columns, query.question) if len(result) > 1 and len(columns) == 2 else None
        
        response_data = {
            "question": query.question,
            "sql": safe_sql,
            "result": result,
            "summary": summary,
            "chart_base64": chart_base64,
            "message": "Success" if result else "No data found"
        }
        
        # Print human-readable answer to CMD (server console)
        print(f"Question: {query.question}")
        print(f"SQL Query: {safe_sql}")
        print(f"Summary: {summary}")
        print(f"Message: {response_data['message']}")
        if chart_base64:
            print("Chart generated (base64 available in full response).")
        
        return response_data
    
    except openai.OpenAIError as oe:
        logging.error(f"LLM error: {str(oe)}")
        raise HTTPException(status_code=500, detail=f"LLM error: {str(oe)}")
    except sqlite3.Error as se:
        logging.error(f"Database error: {str(se)}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(se)}")
    except ValueError as ve:
        logging.error(f"Value error: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


def generate_summary(result: List[Dict], columns: List[str], question: str) -> str:
    if not result:
        return "No data found for this question."
    if len(result) == 1:
        row = result[0]
        if len(row) == 1:
            key = list(row.keys())[0]
            value = row[key]
            key_phrase = key.replace('_', ' ').lower()
            if value is None:
                return f"No {key_phrase} data available."
            else:
                formatted_value = f"{value:,.2f}" if isinstance(value, float) else f"{value:,}"
                verb = "is" if value == 1 else "are"
                return f"The {key_phrase} {verb} {formatted_value}."
        keys = list(row.keys())
        id_val, metric_val = row.get(keys[0]), row.get(keys[1])
        if metric_val is None:
            return f"No valid {keys[1].replace('_', ' ')} found for {keys[0].replace('_', ' ')} {id_val}."
        else:
            formatted_metric = f"{metric_val:,.2f}" if isinstance(metric_val, float) else f"{metric_val:,}"
            return f"Product {id_val} had the highest {keys[1].replace('_', ' ').upper()} of {formatted_metric}."
    if len(result) > 1 and len(columns) == 2:
        return "Top 5 results: " + ", ".join(f"{row[columns[0]]}: {row[columns[1]]:,.2f}" for row in result[:5])
    return f"Returned {len(result)} row(s) with {len(columns)} column(s)."


def generate_chart(result: List[Dict], columns: List[str], question: str) -> str:
    try:
        df = pd.DataFrame(result[:10])  # Limit to 10 rows for speed
        plt.figure(figsize=(8, 4))
        plt.bar(df[columns[0]].astype(str), df[columns[1]])
        plt.title(question[:50])  # Truncate title for brevity
        plt.xlabel(columns[0])
        plt.ylabel(columns[1])
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        chart_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        plt.close()
        return chart_base64
    except Exception as e:
        logging.error(f"Chart generation error: {str(e)}")
        return None
