# NutriMind Meal Planner — Sprint 3

**Phase 1 of the NutriMind AI Wellness Coach platform.**  
A personalised meal planning agent built with LangGraph, backed by real nutritional data, and designed to grow into a full multi-agent wellness system in the Capstone sprint.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Setup](#3-setup)
4. [How to Use](#4-how-to-use)
5. [Technical Decisions](#5-technical-decisions)
6. [Sprint Requirements Mapping](#6-sprint-requirements-mapping)
7. [Optional Tasks Implemented](#7-optional-tasks-implemented)
8. [Improvement Over Sprint 2](#8-improvement-over-sprint-2)
9. [Capstone Roadmap](#9-capstone-roadmap)
10. [Known Limitations](#10-known-limitations)

---

## 1. Project Overview

### What it does

NutriMind Meal Planner is an AI agent that generates personalised, nutritionally validated meal plans. Given a user's health goals, dietary restrictions, calorie target, and allergies, it plans full days of meals — breakfast, lunch, and dinner — backed by real food composition data and a curated nutrition knowledge base. It also produces a deduplicated shopping list for every plan it generates.

### Who it's for

People who want evidence-based meal planning without the manual effort: those managing weight, following specific diets (vegan, gluten-free, halal, etc.), or dealing with food allergies who need plans that are both personalised and safe.

### The problem it solves

Generic meal plan generators either ignore individual health profiles or invent nutritional data. This agent uses the official CIQUAL 2025 French food composition database (3,484 foods) for real macros, an allergen checker grounded in EU allergen law, and a ChromaDB-backed knowledge base of food safety research. Every meal is validated against the user's profile before it is finalised.

### Phase 1 of a larger platform

This project is deliberately scoped as Phase 1. The architecture is designed so that future agents — supplement guidance, weekly check-ins, multi-agent orchestration — can be added on top of the same core without reworking what exists here. See the [Capstone Roadmap](#9-capstone-roadmap) for what comes next.

---

## 2. Architecture

```
User
  │
  ▼
Streamlit UI (app.py)
  │  sidebar: health profile form → save_profile() → SQLite
  │  tab 1:  meal request text → invoke meal_agent
  │  tab 2:  shopping list display
  │
  ▼
load_profile node
  │  reads user_id from state, queries SQLite, injects profile
  │
  ▼
MealPlanningAgent — LangGraph ReAct loop
  │
  ├── check_allergens          scan ingredients against user's allergy list
  ├── lookup_nutrition         CIQUAL 2025 CSV via pandas (3,484 foods)
  ├── score_meal_health        1–10 score against calorie target & health goals
  └── validate_meal_safety     ChromaDB knowledge base (Sprint 2 asset)
                                 └── query translation (MultiQuery)
                                       └── 2-3 LLM-generated search phrasings
  │
  ▼
format_output node
  │  LLM parses agent's final message into structured JSON
  │  → meal_plan dict  +  shopping_list[]
  │
  ▼
Streamlit renders results
  │
  └── LangSmith (auto-instrumented via env vars — no code changes)
        traces every run under project: meal-planning-agent-sprint3
```

### Graph nodes

| Node | Responsibility |
|------|---------------|
| `load_profile` | Entry node — reads SQLite, injects user profile into state |
| `agent` | ReAct reasoning — calls LLM with tool-bound model, loops until no tool calls remain |
| `tools` | `ToolNode` executing whichever tool the agent selected; retries up to 3× on rate limit |
| `format_output` | Post-processing — extracts `meal_plan` and `shopping_list` from the agent's final message |

### State (`core/state.py`)

```python
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]  # accumulates across all nodes
    user_profile: dict                        # injected by load_profile node
    meal_plan: dict                           # written by format_output node
    shopping_list: list[str]                  # written by format_output node
    current_step: str
    user_id: str
    error: str | None
```

---

## 3. Setup

### Prerequisites

- Python 3.11+
- OpenAI API key
- Sprint 2 ChromaDB knowledge base (optional — the RAG tool degrades gracefully without it)
- LangSmith API key (optional — tracing is off when the key is absent)

### Steps

**1. Clone the repository**
```bash
git clone <repo-url>
cd NutriMind_Meal_Planner
```

**2. Create and activate a virtual environment**
```bash
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Configure environment variables**
```bash
cp .env.example .env
```

Open `.env` and fill in your keys:

```env
OPENAI_API_KEY=sk-...

# LangSmith (optional but recommended)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls__...
LANGCHAIN_PROJECT=meal-planning-agent-sprint3

# SQLite path
MEMORY_DB_PATH=data/user_profiles.db
```

**5. CIQUAL nutritional data**

Already included in the repository at `data/ciqual/ciqual_cleaned.csv` — no action needed.

**6. Add the ChromaDB knowledge base (Sprint 2)**

Copy your Sprint 2 ChromaDB directory to:
```
knowledge_base/data/chroma_db/
```

Collection name expected: `nutrition_kb`

If the directory is absent, the RAG validator returns a graceful fallback message — the rest of the agent still works.

**7. Run the app**
```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

**8. Run the tests**
```bash
# Unit tests (no API key needed, ~2 seconds)
pytest tests/test_tools.py -v

# End-to-end test (requires OPENAI_API_KEY, ~2–3 minutes)
pytest tests/test_e2e.py -v -s
```

---

## 4. How to Use

### Step 1 — Fill in your health profile (sidebar)

Open the sidebar on the left. Enter:

| Field | Example |
|-------|---------|
| Name | Alex |
| Age | 32 |
| Weight | 78 kg |
| Height | 175 cm |
| Daily calorie target | 1 800 kcal |
| Health goals | Weight loss, Heart health |
| Dietary restrictions | Gluten-free |
| Allergies | peanuts, sesame |

Click **Save Profile**. Your profile is stored in SQLite and pre-filled on every future visit.

### Step 2 — Generate a meal plan (Tab 1)

Type a request in the text box:

```
Plan 3 days of healthy meals for me
```

Other example requests:
```
Give me a high-protein meal plan for muscle gain this week
Plan 5 days of vegan meals, around 1600 kcal/day
I need gluten-free meals for 2 days with at least 30g protein per meal
```

Click **Generate Plan**. A spinner shows while the agent runs (typically 2–4 minutes for a 3-day plan — the agent calls all 4 tools for each meal).

The result appears as expandable day-by-day sections showing meal names, ingredients, and macros (calories, protein, carbs, fat).

### Step 3 — Check the shopping list (Tab 2)

Switch to the **Shopping List** tab. The list is deduplicated and alphabetically sorted — ready to copy into your grocery app.

### Expected output (example)

```
Day 1
  Breakfast: Greek Yogurt Parfait
    Ingredients: Greek yogurt, strawberries, blueberries, granola
    206 kcal | P: 6g  C: 7.5g  F: 16.3g

  Lunch: Grilled Chicken Salad
    Ingredients: chicken breast, mixed greens, feta cheese, olive oil
    348 kcal | P: 45g  C: 4g  F: 14g

  Dinner: Quinoa Bowl with Roasted Vegetables
    Ingredients: quinoa, courgette, cherry tomatoes, olive oil
    420 kcal | P: 12g  C: 58g  F: 8g
...

Shopping List (24 items)
  - blueberries
  - chicken breast
  - cherry tomatoes
  ...
```

---

## 5. Technical Decisions

### Why LangGraph ReAct over a simple chain

A fixed chain would require pre-defining the exact sequence of tool calls. Meal planning is inherently variable — a 1-day plan needs ~12 tool calls, a 5-day plan needs ~60. The ReAct loop lets the agent decide dynamically which tool to call next based on the output of the previous call, retry with corrected queries when a food isn't found, and stop when it has enough data. A chain cannot do this.

### Why Agentic RAG instead of embedding CIQUAL into ChromaDB

CIQUAL is structured tabular data: 3,484 foods, each with exactly defined numeric columns. Vector search on this data would be lossy — embeddings of "chicken breast, 100g, 165 kcal" do not reliably retrieve precise numeric values. A pandas lookup with substring matching returns exact, reproducible figures.

The ChromaDB knowledge base, by contrast, holds unstructured research text about food additives, allergens, and ingredient safety. This is what vector search excels at. Each tool is matched to the retrieval method that suits its data structure.

### Why query translation in the RAG tool

Sprint 2 reviewer feedback identified that single-query RAG retrieval missed relevant documents when the user's phrasing didn't match the knowledge base's terminology. Query translation (generating 2–3 alternative phrasings via an LLM before querying ChromaDB) significantly improves recall at the cost of 1–2 extra LLM calls per tool invocation — a worthwhile trade-off for a safety-critical retrieval step.

### Why SQLite for long-term memory

User health profiles must persist across sessions. SQLite is zero-config, ships with Python, and requires no separate database service. For a Sprint 3 prototype with a single user, it is the right level of complexity. The schema (`core/memory.py`) uses an upsert pattern so profiles survive app restarts.

### Why CIQUAL as a pandas tool instead of hardcoded data

Hardcoded nutrition data becomes stale, covers only a few hundred foods, and is impossible to audit. CIQUAL is published annually by the French food safety agency (Anses), covers 3,484 foods with known provenance, and is directly downloadable. The pandas lookup tool scales to any number of foods without code changes.

### Why string-only parameters on all tools

LangGraph tools are invoked via function calling. When tool parameters are complex types (dicts, lists), the LLM must generate valid JSON for nested structures — and it frequently produces malformed arguments, causing tool call failures. Flattening all inputs to simple strings (comma-separated lists, plain text) removes this failure mode entirely. Each tool parses structured data internally.

### Why LangSmith for observability

LangSmith auto-instruments LangGraph with zero code changes — just two environment variables. It provides a visual trace of every node, every tool call, every LLM invocation, input/output at each step, and latency data. For a ReAct agent that can make 60+ tool calls in a single run, this is essential for debugging and iterating on prompts. It is the only observability tool added in Sprint 3.

---

## 6. Sprint Requirements Mapping

| Requirement | Implementation |
|------------|---------------|
| Define a clear agent purpose | Personalised meal planning with nutritional validation — documented in this README and in `CLAUDE.md` |
| Core functionality | ReAct agent plans meals, validates against profile, returns structured meal plan + shopping list |
| User interface | Streamlit app (`app.py`) — sidebar profile form, two-tab layout, spinner feedback |
| Appropriate tools and libraries | LangGraph, LangChain, OpenAI, ChromaDB, pandas, Streamlit, SQLite, LangSmith |
| Error handling | `try/except` on all agent invocations; RAG tool degrades gracefully; ToolNode retries on rate limit |
| Real-world usage | Profile persistence, allergen safety checks, real CIQUAL data, ChromaDB knowledge base |
| Documentation | This README + inline docstrings on every tool (required by LangChain tool-calling) |
| Examples of common use cases | Section 4 of this README |
| Technical decisions explained | Section 5 of this README |

---

## 7. Optional Tasks Implemented

### Medium: Long-term memory (SQLite user profiles)
`core/memory.py` persists user health profiles across sessions. The `load_profile` graph node automatically injects the profile before the ReAct loop on every invocation. Profile fields: name, age, weight, height, calorie target, health goals, dietary restrictions, allergies.

### Medium: External data tool (CIQUAL pandas lookup)
`tools/nutrition_lookup.py` queries the CIQUAL 2025 CSV (3,484 foods) using pandas substring matching with difflib fuzzy fallback. Returns exact, portion-scaled macros from an authoritative external dataset.

### Medium: Retry logic
The `ToolNode` is configured with `RetryPolicy(max_attempts=3)`, retrying failed tool calls up to twice on transient errors. The `agent` node independently retries up to 3 times on `RateLimitError` with a 2-second backoff. A `recursion_limit` of 80 iterations prevents infinite loops.

### Hard: Agentic RAG with query translation
`tools/rag_validator.py` queries a ChromaDB collection of food safety research. Before querying, it calls an LLM to generate 2–3 alternative search phrasings (query translation), then deduplicates results by SHA-256 content hash. This improves recall over single-query RAG and directly addresses Sprint 2 reviewer feedback.

### Hard: LangSmith observability
LangSmith is auto-instrumented via environment variables. Every run is tagged with `run_name="meal_plan_generation"` and `metadata={"user_id": ..., "sprint": "sprint3"}` for easy filtering in the dashboard. A startup check prints whether tracing is enabled or disabled.

---

## 8. Improvement Over Sprint 2

### Query translation added to RAG pipeline

Sprint 2 used single-query retrieval: one embedding, one ChromaDB query. The reviewer noted that this missed relevant documents when the user's phrasing didn't match the KB's vocabulary.

Sprint 3 generates 2–3 alternative queries via an LLM call before hitting ChromaDB, then deduplicates results by content hash. This significantly improves recall for safety-critical lookups — for example, "does this meal contain additives?" now also searches for "preservatives", "emulsifiers", and "food colouring risks" in the same call.

### String-only tool parameters for reliability

Sprint 2 tools accepted dict and list parameters. During development, the LLM frequently generated malformed JSON for nested arguments, causing silent tool failures. All Sprint 3 tools accept only `str`, `int`, or `float` — the tool parses structured data internally. This eliminated an entire category of agent failures.

### Modular architecture enables Capstone reuse

`core/`, `tools/`, and `prompts/` are independently importable. No business logic in `app.py`. Each tool file is independently testable. The graph is compiled once at module load and reused across requests. This design allows the Capstone to add new agents alongside this one without touching the existing code.

---

## 9. Capstone Roadmap

This project is Phase 1 of the NutriMind AI Wellness Coach. The following will be built in the Capstone sprint:

| Feature | Description |
|---------|-------------|
| **Supplement guidance agent** | Separate LangGraph graph; recommends supplements based on dietary gaps identified in meal plans |
| **Weekly check-in agent** | Tracks meal adherence over sessions; adjusts future plans based on feedback |
| **Multi-agent orchestration** | LangGraph supervisor routing between meal planner, supplement advisor, and check-in agents |
| **FastAPI backend** | Wraps `core/` in a REST API; enables the Flutter mobile frontend |
| **Flutter mobile UI** | Production-quality cross-platform app replacing the Streamlit prototype |
| **Open Food Facts integration** | Adds product barcode lookup for branded/packaged food validation |
| **User authentication** | Multi-user support replacing the hardcoded `default_user` |
| **Progress tracking** | Persistent tracking of weight, calorie adherence, and goal progress over weeks |

---

## 10. Known Limitations

**CIQUAL coverage gaps**  
CIQUAL is a French database optimised for ingredients available in France. Foods common in other cuisines (black beans, tortillas, miso) may not be found by name. The agent handles this gracefully — it retries with alternative names — but may fall back to approximate values for some ingredients.

**Agent runtime**  
A 3-day meal plan requires approximately 36–60 tool calls (allergen check + nutrition lookup + health score + RAG validation per meal). On `gpt-4o-mini`, this takes 2–4 minutes. Streamlit shows a spinner for the full duration. A streaming output mode (not implemented in Sprint 3) would improve perceived performance.

**Single-user SQLite**  
The current implementation uses a hardcoded `user_id = "default_user"`. Multi-user support requires authentication — deferred to Capstone.

**No ChromaDB on Streamlit Cloud**  
The RAG tool uses `/tmp/chroma_db` on Streamlit Cloud (detected via `/mount/src`). Without a pre-built KB at that path, the tool returns a graceful fallback. Building the KB at deploy time is not implemented in Sprint 3.

**Format sensitivity**  
The `format_output` node uses an LLM to extract structured JSON from the agent's final message. If the agent produces an unusually formatted response, the JSON parse may fail — the error is captured and surfaced in `state["error"]` rather than crashing the app.

---

## Project Structure

```
NutriMind_Meal_Planner/
├── app.py                      # Streamlit UI — no business logic
├── core/
│   ├── graph.py                # LangGraph graph: nodes, edges, compilation
│   ├── state.py                # AgentState TypedDict
│   └── memory.py               # SQLite persistence for user profiles
├── tools/
│   ├── nutrition_lookup.py     # CIQUAL 2025 pandas lookup
│   ├── allergen_checker.py     # EU allergen mapping + direct matching
│   ├── health_scorer.py        # 1–10 meal quality score
│   └── rag_validator.py        # ChromaDB RAG with query translation
├── prompts/
│   └── system_prompts.py       # All prompt strings (never inline in logic)
├── data/
│   ├── ciqual/
│   │   └── ciqual_cleaned.csv  # CIQUAL 2025 food composition table
│   └── user_profiles.db        # SQLite (created at runtime)
├── knowledge_base/
│   └── data/chroma_db/         # Sprint 2 ChromaDB knowledge base
├── tests/
│   ├── conftest.py             # sys.path setup
│   ├── test_tools.py           # 8 unit tests (no API key required)
│   └── test_e2e.py             # Full agent invocation test
├── .env.example                # Environment variable template
├── requirements.txt
└── CLAUDE.md                   # Claude Code project instructions
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `langgraph` | Agent graph orchestration (ReAct loop, nodes, edges) |
| `langchain` + `langchain-openai` | LLM bindings, tool definitions, message types |
| `chromadb` | Vector store for Sprint 2 nutrition knowledge base |
| `pandas` + `openpyxl` | CIQUAL CSV lookup |
| `streamlit` | Web UI |
| `python-dotenv` | Environment variable management |
| `langsmith` | LLM observability and tracing |
| `pydantic` | Data validation (used by LangChain internals) |

---

*Built as Sprint 3 of the Turing College AI Engineering programme.*  
*Phase 1 of the NutriMind AI Wellness Coach — Capstone continuation planned.*
