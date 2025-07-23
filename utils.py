import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from typing import List, Dict
import logging

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
