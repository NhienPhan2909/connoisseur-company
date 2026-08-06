# 🍽️ Connoisseur Companion

**A Retrieval-Augmented, tool-using conversational agent for California restaurant discovery.**

🔗 **Live demo:** https://connoisseur-company.onrender.com/
*(Free-tier hosting — the app sleeps after ~15 min idle and takes 30–60s to wake on first visit.)*

Ask about restaurants by name, cuisine, or vibe — e.g. *"Find me a moody spot in DTLA"* or *"Tell me about Sakura Garden"*. The UI includes a live **agent reasoning trace** panel so you can see exactly which tools the agent calls, with what arguments, and what data it gets back, before it drafts its final answer.

---

## Table of contents

1. [Why this project](#why-this-project)
2. [System architecture](#system-architecture)
3. [Agentic design: from single agent to a specialized multi-agent system](#agentic-design-from-single-agent-to-a-specialized-multi-agent-system)
4. [Tasks defined for the multi-agent pipeline](#tasks-defined-for-the-multi-agent-pipeline)
5. [Workflow orchestration (sequential + parallel phases)](#workflow-orchestration-sequential--parallel-phases)
6. [Production chatbot: tool-calling with MCP](#production-chatbot-tool-calling-with-mcp)
7. [Design decisions & trade-offs](#design-decisions--trade-offs)
8. [Tech stack](#tech-stack)
9. [Project structure](#project-structure)
10. [Running locally](#running-locally)
11. [Deployment](#deployment)

---

## Why this project

Most "chatbot" demos are a single LLM call with a system prompt. This project instead explores the **full agentic AI lifecycle**: preparing multimodal data for retrieval, designing specialized agents with distinct roles/goals, orchestrating them through a stateful workflow with sequential and parallel phases, and finally exposing the whole thing as a **production tool-calling chatbot** over a standardized protocol (MCP) that any LLM host can talk to.

It's built around a single domain — California restaurants — using:
- Unstructured text (`California-Culinary-Map.txt`) describing restaurants in prose,
- Structured JSON (`structured_restaurant_data.json`) extracted from that text via an LLM,
- User reviews with images (`augmented_user_review.json`) with LLM-generated image captions.

## System architecture

```
┌─────────────────────────────┐
│   Module 1 — Data Prep      │  Extract structured JSON + image captions
│   (multimodal LLM pipeline) │  from raw restaurant text & review photos
└──────────────┬──────────────┘
               │
┌──────────────▼───────────────┐
│   Module 2 — RAG             │  Chunk + embed restaurant/recipe data,
│   (vector store + retrieval) │  build a queryable vector database
└──────────────┬───────────────┘
               │
┌──────────────▼───────────────┐
│   Module 3 — Agentic Layer   │  Design 6 specialized agents (L1) → wire them
│   (design → orchestrate →    │  into a stateful multi-agent workflow with
│   productionize)             │  parallel phases (L2) → expose as a real
│                              │  tool-calling chatbot over MCP (L3)
└──────────────┬───────────────┘
               │
┌──────────────▼───────────────┐
│   Deployed App (this repo)   │  Gradio host (app.py) ⇄ MCP tool server
│                              │  (server.py) ⇄ Gemini LLM, hosted on Render
└──────────────────────────────┘
```

The deployed app (`app.py` + `server.py`) is a **simplified, production-ready distillation** of the ideas explored in the Module 3 notebooks: instead of a LangGraph multi-agent workflow with 6 specialized agents, it uses a single ReAct agent that dynamically chooses from 3 well-scoped tools — a deliberate trade-off explained below.

## Agentic design: from single agent to a specialized multi-agent system

`M3L1-Design-Specialized-Agents-v1.ipynb` defines **six specialized agents**, each with a `role`, `goal`, and `backstory` (the classic agent-definition pattern popularized by frameworks like CrewAI, reimplemented here with plain system prompts + an LLM client):

| # | Agent | Responsibility | Why a separate agent? |
|---|-------|-----------------|------------------------|
| 1 | **User Profile Generator** | Extracts preferences, dietary restrictions, and dining patterns from a user's visit history and social posts | Personalization needs a dedicated "read the user" step before any retrieval happens — mixing this into retrieval logic would make prompts unfocused |
| 2 | **RAG Retriever** | Queries the Module 2 vector database for restaurants/recipes relevant to the user profile | Retrieval quality benefits from a narrow, single-purpose prompt tuned for query formulation and relevance filtering, not recommendation writing |
| 3 | **Food Trend Analyst** | Identifies current food trends and culinary movements relevant to the candidates | Trend awareness is a distinct "lens" on the same data — isolating it prevents trend commentary from diluting nutrition or style analysis |
| 4 | **Food Style Expert** | Analyzes cuisine type, regional variation, cooking method, and flavor profile of candidates | Requires deep, specific domain knowledge (culinary anthropology) that shouldn't be conflated with health/dietary reasoning |
| 5 | **Nutrition Expert** | Evaluates nutritional content, allergens, and dietary-restriction fit | Nutrition claims need conservative, specialized reasoning — kept separate so it can't be "talked over" by more creative agents |
| 6 | **Recommendation Expert** | Synthesizes all upstream analyses into a final, well-reasoned recommendation | A dedicated synthesis step avoids any single upstream agent over-indexing its own specialty in the final answer |

**Design rationale:** splitting a single "restaurant chatbot" prompt into 6 specialized agents follows the same principle as microservices — each agent has one job, one clear success criterion, and can be tested/improved independently. It also mirrors how a real restaurant recommendation team might work (a trend scout, a nutritionist, a cuisine expert, etc., all feeding a lead recommender), which is a natural, explainable decomposition of the problem.

## Tasks defined for the multi-agent pipeline

Each agent above is paired with a **task** (`description` + `expected_output`) — this two-part definition (agent = *who*, task = *what*) is what allows the same agent role to be reused across different pipelines or inputs. **Six tasks** are defined in total, one per agent, chained in dependency order:

1. **Generate the user profile** → feeds into retrieval
2. **Retrieve relevant restaurants and recipes** → feeds into all three analysis tasks
3. **Analyze food trends** (parallel)
4. **Analyze food styles** (parallel)
5. **Evaluate nutrition and dietary fit** (parallel)
6. **Generate final recommendations** → synthesizes outputs of tasks 3–5

Separating *task* from *agent* config makes the pipeline declarative: swapping in a different LLM, prompt, or even a different agent implementation for "Food Style Expert" doesn't require touching what the task expects as output — a clean separation of concerns borrowed from workflow-orchestration best practice.

## Workflow orchestration (sequential + parallel phases)

`M3L2-Implement-Multi-Agent-Systems-v1.ipynb` wires the six agents into a **hybrid workflow** — sequential where there's a hard data dependency, parallel where there isn't:

| Phase | Type | Agents | Why |
|-------|------|--------|-----|
| 1. User Analysis | Sequential | User Profile Generator | Must complete before retrieval can be personalized |
| 2. Data Retrieval | Sequential | RAG Retriever | Downstream analysis needs candidate restaurants/recipes first |
| 3. Analysis | **Parallel** | Food Trend Analyst, Food Style Expert, Nutrition Expert | These three agents read the *same* retrieved candidates but analyze independent dimensions (trends / style / nutrition) — running them concurrently (via `ThreadPoolExecutor`, mirroring what LangGraph does under the hood) cuts wall-clock latency roughly 3x versus running them one after another |
| 4. Synthesis | Sequential | Recommendation Expert | Must wait for *all* of Phase 3's outputs before it can synthesize a final answer |

A shared `AgentState` (a `TypedDict`) flows through every node — each node reads what it needs and writes its own output field, plus a `workflow_step` field that's updated at every stage purely for **observability**: it lets the graph (and a UI, in principle) show users/developers exactly where execution currently is, without needing to inspect the full state object.

**Why Phase 3 dominates runtime:** even in parallel, Phase 3 is bottlenecked by its *slowest* of the three agents (each is a separate LLM call with its own latency), whereas Phases 1, 2, and 4 are single LLM calls each — so Phase 3 is the phase most worth optimizing (e.g., using faster/cheaper models for the parallel agents) if latency becomes a concern.

## Production chatbot: tool-calling with MCP

`M3L3-Build-Chatbot-Interface-v1.ipynb` — and this repo's deployed `app.py`/`server.py` — take a different, more *production-realistic* approach than the full 6-agent workflow: a **single ReAct agent equipped with tools exposed over the Model Context Protocol (MCP)**.

**Why MCP + a single agent instead of the full 6-agent LangGraph pipeline for the live demo?**
- **Latency & cost**: A user chatting expects a fast reply. Running 4 sequential LLM calls plus 3 parallel ones per turn (as the full multi-agent workflow does) is appropriate for an offline batch-recommendation job, but too slow/expensive for an interactive chat turn.
- **MCP decouples "what tools exist" from "which LLM/host uses them"**: `server.py` exposes 3 tools (`get_restaurant_info`, `recommend_by_vibe`, `get_review`) as a standalone process. Any MCP-compatible client (Claude Desktop, this Gradio app, a future different LLM) can plug into the same server without modification — this is the same protocol real production AI assistants use to integrate external data sources.
- **A single ReAct loop is enough when tools are well-scoped**: because each tool has a narrow, clearly-documented purpose, one LLM can reliably decide *which* tool to call and *when* — the specialization that the 6-agent design achieves through separate prompts is instead achieved here through separate *tools*, which is cheaper (one LLM call per reasoning step instead of one call per specialist) while keeping the same "single responsibility per unit" principle.

**Live agent trace:** the deployed UI surfaces every tool call, its arguments, and a preview of its result in a dedicated panel — a lightweight window into the same ReAct reasoning loop that the full multi-agent workflow demonstrates conceptually, made visible for anyone (including a hiring manager) inspecting the live demo.

**Resilience:** the app detects Gemini rate-limit (HTTP 429) errors and, instead of crashing, parses Google's `RetryInfo.retryDelay` from the error response to tell the user exactly when (in hours/minutes, and a wall-clock time) they can try again.

## Design decisions & trade-offs

| Decision | Reasoning |
|---|---|
| 6 specialized agents (design) vs. 1 ReAct agent + 3 tools (production) | Multi-agent decomposition is valuable for *designing* and *reasoning about* a complex recommendation pipeline; a single tool-using agent is more latency/cost-appropriate for a real-time chat product. Both patterns are demonstrated intentionally. |
| MCP for tool exposure | Standardizes tool access so the same `server.py` works with any MCP client, not just this specific Gradio app — a realistic integration pattern rather than hard-coding function calls inline. |
| Gemini (`gemini-flash-lite-latest`) over IBM watsonx | The original course used watsonx; this deployment swaps in Gemini's free tier so the live demo requires no paid credentials to run. |
| Render (free tier) over Hugging Face Spaces | Avoided newly-introduced HF Spaces restrictions (ZeroGPU-only free hardware, account-level quota bugs) — Render's free web-service tier needs no GPU and no payment method. |
| Visible reasoning trace in the UI | Makes the agent's tool-use decisions inspectable in real time, rather than requiring a reader to trust an opaque final answer — valuable both for debugging and for demonstrating the underlying agentic mechanics. |

## Tech stack

- **LLM**: Google Gemini (`gemini-flash-lite-latest`) via `langchain-google-genai`
- **Agent orchestration**: LangGraph (`M3L2` workflow) · custom ReAct loop (production `app.py`)
- **Tool protocol**: MCP (Model Context Protocol) via `fastmcp`
- **UI**: Gradio (custom CSS for a responsive, mobile-friendly layout)
- **Data**: LLM-extracted structured JSON + multimodal (image-captioned) review data
- **Hosting**: Render (free tier, auto-deploys from GitHub on every push)

## Project structure

```
app.py                          # Gradio host: ReAct loop, rate-limit handling, agent trace UI
server.py                       # MCP tool server: get_restaurant_info, recommend_by_vibe, get_review
client.py / test.py             # Standalone MCP client scripts for local testing (not used by app.py)
requirements.txt                # Pinned deploy dependencies
structured_restaurant_data.json # LLM-extracted structured restaurant records
augmented_user_review.json      # User reviews with LLM-generated image captions
California-Culinary-Map.txt     # Raw source text describing California restaurants
M2L1-v1.ipynb, M2L2-v1.ipynb, M2L3-v1.ipynb        # Module 2: RAG / vector database construction
M3L1-Design-Specialized-Agents-v1.ipynb            # Module 3.1: define 6 agents' roles/goals/backstories + 6 tasks
M3L2-Implement-Multi-Agent-Systems-v1.ipynb         # Module 3.2: LangGraph workflow, sequential + parallel phases
M3L3-Build-Chatbot-Interface-v1.ipynb               # Module 3.3: intent classification, preference extraction, chatbot loop
```

## Running locally

```bash
python -m venv venv
venv\Scripts\activate           # (Windows) or: source venv/bin/activate
pip install -r requirements.txt
echo GEMINI_API_KEY=your_key_here > .env
python app.py
```

## Deployment

Hosted on [Render](https://render.com) (free tier): connected directly to this GitHub repo with auto-deploy on push to `main`. Build command: `pip install -r requirements.txt`; start command: `python app.py`. `GEMINI_API_KEY` is set as a Render environment secret, never committed to source control.