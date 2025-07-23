from functools import lru_cache
from typing import List, Dict
from memory import conversation_memory

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

FEW_SHOT_EXAMPLES = [
    {"user": "What is my total sales?", "sql": "SELECT SUM(ts.total_sales) AS total_sales FROM total_sales ts;"},
    {"user": "Which product had the highest CPC (Cost Per Click)?", "sql": "SELECT ads.item_id, (SUM(ads.ad_spend) / NULLIF(SUM(ads.clicks), 0)) AS cpc FROM ad_sales ads GROUP BY ads.item_id ORDER BY cpc DESC LIMIT 1;"},
    {"user": "Calculate the RoAS (Return on Ad Spend).", "sql": "SELECT SUM(ads.ad_sales) / NULLIF(SUM(ads.ad_spend), 0) AS roas FROM ad_sales ads;"},
    {"user": "Top 3 items by ad spend.", "sql": "SELECT ads.item_id, SUM(ads.ad_spend) AS total_ad_spend FROM ad_sales ads GROUP BY ads.item_id ORDER BY total_ad_spend DESC LIMIT 3;"}
]

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