# VIKMO Dealer Assistant

An AI-powered conversational assistant for auto-parts dealers — find parts, check stock, and place orders via natural language.

## What's implemented

- **Part A (core):** RAG-based dealer assistant with function calling, multi-turn conversation, and guardrails.
- **Part B (bonus):** Per-SKU demand forecasting with Prophet vs. seasonal-naive baseline.

## Tech Stack

| Component | Choice |
|---|---|
| LLM | Llama 3.3 70B via Groq |
| Embeddings | `sentence-transformers` (all-MiniLM-L6-v2) |
| Vector DB | ChromaDB (local, persistent) |
| Framework | Gemini Python SDK (direct) |
| UI | Streamlit |
| Forecasting | Prophet + seasonal-naive baseline |

## Setup

```bash
# 1. Clone and install
pip install -r requirements.txt

# 2. Set your API key
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY

# 3. Run the Streamlit app
streamlit run app.py

# 4. Run the eval suite
python -m eval.run_eval

# 5. Run demand forecasting (optional)
python -m forecasting.forecast
```

The first run will build the ChromaDB vector index (~30 seconds). Subsequent runs load it from disk.

## Example Interactions

```
User: Do you have brake pads for a Bajaj Pulsar 150?
Bot:  [calls find_parts_by_vehicle] Yes! Found 3 options:
      - BRK-1042 | Brake Pad Set — Bajaj Pulsar 150 | ₹450 | Stock: 32

User: Place an order for 5 units of BRK-1042 for ABC Motors
Bot:  [calls create_order] Order confirmed!
      Order ID: ORD-A1B2C3D4 | Total: ₹2,250 | Status: confirmed

User: What's the weather today?
Bot:  I can only help with auto-parts queries. How can I assist you with parts?
```

## Assumptions

- Stock is not decremented on order placement (read-only catalogue).
- Vehicle fitment matching uses substring search + semantic fallback.
- ChromaDB index is stored in `data/chroma_db/` and persists across runs.

See [DESIGN.md](DESIGN.md) for architecture decisions and [eval/results.json](eval/results.json) for evaluation output.
