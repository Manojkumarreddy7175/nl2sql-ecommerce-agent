## 


from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import sqlite3
import openai
import logging

from config import conn
from prompts import build_prompt
from sql_generator import generate_sql
from utils import generate_summary, generate_chart
from memory import conversation_memory


# Set logging level to ERROR to minimize unnecessary logs in CMD
logging.basicConfig(level=logging.ERROR)


# Connect to local Ollama (Mistral)
openai.api_base = "http://localhost:11434/v1"
openai.api_key = "ollama"  # Dummy key, required by openai-python


# FastAPI app
app = FastAPI()


# Request model
class Query(BaseModel):
    question: str
    session_id: str = "default"


@app.post("/ask")
async def ask_question(query: Query):
    try:
        advanced_prompt = build_prompt(query.question, query.session_id)
        
        initial_sql = generate_sql(advanced_prompt, query.question)
        
        if not initial_sql.upper().startswith("SELECT") or any(kw in initial_sql.upper() for kw in ["DROP", "INSERT", "UPDATE", "DELETE"]):
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
