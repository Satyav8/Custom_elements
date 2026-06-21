"""
Agent loop using Groq API with llama-3.3-70b-versatile.
LangSmith tracing via @traceable decorator.
"""

import os
import json
import time
from groq import Groq
from dotenv import load_dotenv
from langsmith import traceable
from langsmith.wrappers import wrap_openai

from assistant.retrieval import search
from assistant.tools import TOOL_MAP, TOOL_DECLARATIONS
from assistant.prompts import SYSTEM_PROMPT

load_dotenv()

_CLIENT = Groq(api_key=os.environ["GROQ_API_KEY"])
_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
_MAX_RETRIES = 3


@traceable(name="groq_llm_call", run_type="llm")
def _llm_call(messages: list, model: str = _MODEL) -> dict:
    """Traced Groq API call."""
    response = _CLIENT.chat.completions.create(
        model=model,
        messages=messages,
        tools=TOOL_DECLARATIONS,
        tool_choice="auto",
        max_tokens=1024,
    )
    msg = response.choices[0].message
    return {
        "content": msg.content,
        "tool_calls": [
            {"name": tc.function.name, "arguments": tc.function.arguments}
            for tc in (msg.tool_calls or [])
        ],
        "_raw": msg,  # keep raw for agent loop
    }


@traceable(name="tool_call", run_type="tool")
def _run_tool(name: str, args: dict) -> dict:
    """Traced tool execution."""
    fn = TOOL_MAP.get(name)
    return fn(**args) if fn else {"error": f"Unknown tool: {name}"}


def _call_with_retry(messages):
    for attempt in range(_MAX_RETRIES):
        try:
            result = _llm_call(messages)
            return result["_raw"], result
        except Exception as e:
            if "tool_use_failed" in str(e) and attempt < _MAX_RETRIES - 1:
                time.sleep(1)
                continue
            raise


class DealerAgent:
    def __init__(self):
        self._history: list[dict] = []

    @traceable(name="dealer_agent_chat", run_type="chain")
    def chat(self, user_message: str) -> str:
        retrieved = search(user_message, n_results=8)
        catalogue_context = "\n".join(
            f"- {p['sku']} | {p['name']} | INR {p['price_inr']} | Stock: {p['stock']} | Fits: {p['vehicle_fitment']}"
            for p in retrieved
        )

        user_with_context = f"[Relevant catalogue items:\n{catalogue_context}]\n\n{user_message}"
        self._history.append({"role": "user", "content": user_with_context})

        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + self._history

        while True:
            raw_msg, _ = _call_with_retry(messages)
            messages.append({
                "role": "assistant",
                "content": raw_msg.content,
                "tool_calls": raw_msg.tool_calls,
            })

            if not raw_msg.tool_calls:
                break

            for tc in raw_msg.tool_calls:
                fn_name = tc.function.name
                fn_args = json.loads(tc.function.arguments)
                result = _run_tool(fn_name, fn_args)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result),
                })

        final_text = raw_msg.content or ""
        self._history[-1] = {"role": "user", "content": user_message}
        self._history.append({"role": "assistant", "content": final_text})
        return final_text.strip()

    def reset(self):
        self._history = []
