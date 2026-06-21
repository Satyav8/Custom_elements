# DESIGN.md — VIKMO Dealer Assistant

## Part A — Dealer Assistant

### 1. Retrieval Approach (RAG)

**Choice:** `sentence-transformers` (all-MiniLM-L6-v2) + ChromaDB with cosine similarity.

**Why this model:** all-MiniLM-L6-v2 is a compact (80MB), fast, well-tested model for semantic similarity. It runs locally with no API cost and handles domain-general text well. For an auto-parts domain, the terminology is close enough to general English that a general-purpose embedding model suffices.

**Chunking:** One document per SKU. Each document concatenates: `name | category | brand | vehicle_fitment | description`. No sub-chunking is needed because each SKU is already a short, self-contained record. Chunking within a SKU would lose cross-field context (e.g. brand + fitment together).

**Indexing:** ChromaDB with HNSW index and cosine distance. HNSW gives O(log n) approximate nearest-neighbour search, adequate for 600 SKUs (and scales to millions without code changes).

**Retrieval at query time:** Top-8 results are retrieved and injected into the system prompt as a formatted list. This keeps context window usage low while covering the most relevant SKUs. The LLM then reasons over these candidates.

**Alternative considered:** BM25 keyword search. Rejected because: (a) it misses semantic variations ("engine oil" vs "lubricant"), (b) it doesn't handle misspellings or partial vehicle names well.

### 2. Tool Design & Function Calling

Three tools, chosen to cover the three core dealer actions:

| Tool | When called | Output |
|---|---|---|
| `check_stock` | User asks about availability of a specific SKU | Stock count + price |
| `find_parts_by_vehicle` | User names a vehicle without a specific SKU | Filtered part list |
| `create_order` | User wants to place an order | Structured `OrderConfirmation` JSON |

**How the model decides:** The Gemini model sees tool declarations with descriptions. The system prompt instructs it to always use tools for data lookups, never to invent values. Gemini's function-calling API returns a `FunctionCall` part when it decides a tool is needed; the agent loop executes the tool and feeds the result back as a `FunctionResponse`.

**Structured output for orders:** `create_order` uses Pydantic models (`OrderRequest`, `OrderLineItem`, `OrderConfirmation`) to validate input and serialise output. The LLM never sees free text for orders — it always gets a validated JSON dict.

### 3. Prompt Design & Guardrails

**System prompt includes:**
- Role definition scoped to auto-parts only
- Explicit instruction to never invent data
- Instruction to ask one clarifying question when vehicle/context is missing
- Hard rule: decline off-topic queries with a polite fixed message

**Guardrails:**
- The LLM is told to respond to off-topic queries with: "I can only help with auto-parts queries."
- Catalogue context is injected per turn, so the model always has grounded data to work from
- Tool results include explicit error messages when SKUs don't exist, preventing the model from hallucinating details

**Known limitation:** Prompt-level guardrails are soft — a sufficiently adversarial prompt can bypass them. Production hardening would add a classifier layer before the LLM.

### 4. Conversation Handling

History is stored as a list of `{"role", "parts"}` dicts and passed to Gemini on every turn. The agent loop:
1. Retrieves fresh catalogue context for each user message
2. Appends user message to history
3. Calls Gemini → handles tool calls → gets final text
4. Appends model response to history

Multi-turn context is maintained this way across the session. A "New Conversation" button in the UI resets history.

### 5. Evaluation Methodology

**Eval set (15 cases):** covers 4 types:
- `happy_path` (5): standard flows that must work correctly
- `ambiguous` (3): underspecified queries where the bot must ask a clarifying question
- `tricky` (4): edge cases — nonexistent SKUs, over-stock orders, universal fitment
- `out_of_scope` (3): off-topic queries that must be declined

**Scoring:** Each case checks `must_contain` and `must_not_contain` string lists against the response. Simple but fast and deterministic.

**Failure modes observed:**
- The model sometimes searches semantically before asking for clarification on ambiguous vehicle queries — retrieval is too eager.
- `must_not_contain` checks on partial strings can produce false negatives (e.g. "confirmed" appearing in a polite decline).
- Tool selection is occasionally incorrect when query mentions a vehicle name that looks like a SKU.

**What I'd change:** Use an LLM-as-judge scorer instead of string matching for richer evaluation. Add multi-turn eval cases (turn 1 is ambiguous, turn 2 provides vehicle, turn 3 orders).

---

## Part B — Demand Forecasting

### Model Choice

**Prophet** (Meta's time-series library) vs. **Seasonal Naive** baseline.

**Why Prophet:**
- Handles yearly seasonality automatically (important for festive Oct–Nov lift in the data)
- Handles trend changes robustly with `changepoint_prior_scale`
- Optional regressor support lets us include `promo_flag` directly
- Interpretable: decomposable into trend + seasonality + holiday components

**Why seasonal naive as baseline:**
- "Last year same week" is the natural benchmark for seasonal data
- It respects the weekly cadence and annual pattern without any fitting
- Any model that can't beat it isn't adding value

### Validation Scheme

**Strict temporal holdout:** last 4 weeks of each SKU's series are the test set. Training uses only data before the cutoff. No shuffling, no cross-validation across time — that would leak future information.

**Why 4 weeks:** matches the "near-term" framing in the brief. Long enough to measure seasonal effects but short enough to be actionable for inventory decisions.

**Leakage prevention:**
- The cutoff date is computed per-SKU from its last observed date
- Prophet's `make_future_dataframe` generates strictly future dates
- Promo flags for the forecast horizon default to 0 (conservatively assume no promo)

### Metrics

**MAE** (Mean Absolute Error): preferred over RMSE because it's robust to the occasional large spike from promotions. MAPE is also reported but excluded from primary ranking because it's undefined when true values are 0.

### Limitations

- With only 78 weeks (< 2 full years), Prophet has limited data to estimate yearly seasonality. More data would improve accuracy significantly.
- Per-SKU models don't share information across similar SKUs. A hierarchical model (e.g. grouped Prophet or LightGBM with shared lag features) would be the next step.
