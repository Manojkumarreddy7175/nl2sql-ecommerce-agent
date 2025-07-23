import re
from config import openai
import logging

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
