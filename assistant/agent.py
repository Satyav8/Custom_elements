"""
Agent loop using Groq API with llama-3.3-70b-versatile.
Includes retry logic for intermittent Groq tool_use_failed errors.
"""

import os
import json
import time
from groq import Groq
from dotenv import load_dotenv

from assistant.retrieval import search
from assistant.tools import TOOL_MAP, TOOL_DECLARATIONS
from assistant.prompts import SYSTEM_PROMPT

load_dotenv()

_CLIENT = Groq(api_key=os.environ["GROQ_API_KEY"])
_MODEL = "llama-3.3-70b-versatile"
_MAX_RETRIES = 3


def _call_with_retry(messages):
    """Call Groq with retries for intermittent tool_use_failed errors."""
    for attempt in range(_MAX_RETRIES):
        try:
            return _CLIENT.chat.completions.create(
                model=_MODEL,
                messages=messages,
                tools=TOOL_DECLARATIONS,
                tool_choice="auto",
                max_tokens=1024,
            )
        except Exception as e:
            if "tool_use_failed" in str(e) and attempt < _MAX_RETRIES - 1:
                time.sleep(1)
                continue
            raise


class DealerAgent:
    def __init__(self):
        self._history: list[dict] = []

    def chat(self, user_message: str) -> str:
        # Retrieve relevant catalogue context
        retrieved = search(user_message, n_results=8)
        catalogue_context = "\n".join(
            f"- {p['sku']} | {p['name']} | INR {p['price_inr']} | Stock: {p['stock']} | Fits: {p['vehicle_fitment']}"
            for p in retrieved
        )

        # System prompt stays concise; catalogue context goes in user message
        system = SYSTEM_PROMPT
        user_with_context = (
            f"[Relevant catalogue items:\n{catalogue_context}]\n\n{user_message}"
        )

        self._history.append({"role": "user", "content": user_with_context})
        messages = [{"role": "system", "content": system}] + self._history

        # Agentic loop
        while True:
            response = _call_with_retry(messages)
            msg = response.choices[0].message
            messages.append({"role": "assistant", "content": msg.content, "tool_calls": msg.tool_calls})

            if not msg.tool_calls:
                break

            for tc in msg.tool_calls:
                fn_name = tc.function.name
                fn_args = json.loads(tc.function.arguments)
                fn = TOOL_MAP.get(fn_name)
                result = fn(**fn_args) if fn else {"error": f"Unknown tool: {fn_name}"}
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result),
                })

        final_text = msg.content or ""
        # Store clean history (without injected catalogue context)
        self._history[-1] = {"role": "user", "content": user_message}
        self._history.append({"role": "assistant", "content": final_text})

        return final_text.strip()

    def reset(self):
        self._history = []
