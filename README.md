# VIKMO Dealer Assistant

An AI-powered conversational assistant for auto-parts dealers — find parts, check stock, and place orders via natural language.

**GitHub:** https://github.com/Satyav8/Custom_elements | **Author:** Prabhas

## What's implemented

- **Part A (core):** RAG-based dealer assistant with function calling, multi-turn conversation, and guardrails.
- **Part B (bonus):** Per-SKU demand forecasting with Prophet vs. seasonal-naive baseline.

## Tech Stack

| Component | Choice |
|---|---|
| LLM | Llama 4 Scout 17B via Groq (primary); Llama 3.3 70B + 3.1 8B as fallbacks |
| Embeddings | `sentence-transformers` (all-MiniLM-L6-v2) |
| Vector DB | ChromaDB (local, persistent) |
| Tracing | LangSmith (`@traceable` on all LLM/tool/agent calls) |
| UI | Streamlit |
| Forecasting | Prophet + seasonal-naive baseline |

## Setup

```bash
# 1. Clone and install
pip install -r requirements.txt

# 2. Set your API key
cp .env.example .env
# Edit .env and add your GROQ_API_KEY (and optionally LANGCHAIN_API_KEY for tracing)

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
Bot:  [calls find_parts_by_vehicle] Yes! BRK-1002 | Brake Pad Set — Bajaj Pulsar 150 | INR 1460 | Stock: 136

User: Place an order for 2 units of BRK-1042 for ABC Motors
Bot:  [calls create_order] Order placed! ORD-94AFDE60 | Total: INR 4940 | Status: confirmed

User: What's the weather today?
Bot:  I can only help with auto-parts queries.
```

## Assumptions

- Stock is not decremented on order placement (read-only catalogue).
- Vehicle fitment matching uses substring search + semantic fallback.
- ChromaDB index is stored in `data/chroma_db/` and persists across runs.

## Forecasting Results (Part B)

| Model | Overall MAE | SKUs beat baseline |
|---|---|---|
| Seasonal Naive (baseline) | 9.98 | — |
| Prophet | 7.15 | 25/30 (83%) |

Holdout: last 4 weeks per SKU. No data leakage — strict temporal split.

## Eval Results (Part A)

| Category | Score |
|---|---|
| happy_path | 5/5 |
| ambiguous | 3/3 |
| out_of_scope | 3/3 |
| tricky | 4/4 |
| **Total** | **15/15 (100%)** |

See [DESIGN.md](DESIGN.md) for architecture decisions and [eval/results.json](eval/results.json) for full evaluation output.

---

*Built as part of the VIKMO AI/ML Internship take-home assignment. All evaluation results are reproducible — run `python -m eval.run_eval` to verify.*
