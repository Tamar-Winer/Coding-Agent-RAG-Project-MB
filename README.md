<div align="center">

<!-- Animated Title Banner -->
<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=200&section=header&text=Master%20Agentic%20RAG&fontSize=52&fontColor=ffffff&animation=fadeIn&fontAlignY=38&desc=Self-Correcting%20%7C%20Event-Driven%20%7C%20Multilingual&descAlignY=60&descSize=18" />

<!-- Badges -->
<p>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/LlamaIndex-0.14-7C3AED?style=for-the-badge&logo=llama&logoColor=white" />
  <img src="https://img.shields.io/badge/Gradio-6.x-F97316?style=for-the-badge&logo=gradio&logoColor=white" />
  <img src="https://img.shields.io/badge/Cohere-command--r%2B-coral?style=for-the-badge&logo=cohere&logoColor=white" />
  <img src="https://img.shields.io/badge/Pinecone-Vector%20DB-00B4B4?style=for-the-badge&logo=pinecone&logoColor=white" />
</p>

<p>
  <img src="https://img.shields.io/badge/Multilingual-HE%20%7C%20EN%20%7C%20ES-10B981?style=flat-square" />
  <img src="https://img.shields.io/badge/Self--Correction-Enabled-F59E0B?style=flat-square" />
  <img src="https://img.shields.io/badge/Agentic%20Workflow-Event--Driven-818CF8?style=flat-square" />
  <img src="https://img.shields.io/badge/Phase-A%20%2B%20B%20%2B%20C-F97316?style=flat-square" />
  <img src="https://img.shields.io/badge/License-MIT-64748B?style=flat-square" />
</p>

<br/>

> **An advanced Retrieval-Augmented Generation (RAG) system** built on an Event-Driven Workflow architecture.
> It scans technical documentation, indexes it semantically, and delivers precise answers with built-in
> self-validation and automatic error correction — in both Hebrew and English.

<br/>

</div>
<img width="1915" height="905" alt="image" src="https://github.com/user-attachments/assets/1b1b2ace-729d-438a-8eed-3fa1cc27438d" />
<img width="1918" height="903" alt="image" src="https://github.com/user-attachments/assets/422c2c93-9899-463e-b068-30e9e21aea6a" />
---

## Table of Contents

- [Overview](#-overview)
- [Live Demo](#-live-demo)
- [Architecture](#-architecture)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [How It Works](#-how-it-works)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Usage](#-usage)
- [The Three Phases](#-the-three-phases)
- [Data Schema](#-data-schema)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)

---

## Overview

**Master Agentic RAG** is not a simple "question and answer" bot. It is a multi-step **agentic pipeline** that thinks, validates, and self-corrects before returning any response.

When you ask a question, the system:

1. **Routes** the query intelligently — structured data request or semantic search?
2. **Retrieves** the top-3 most relevant document chunks from a cloud vector database
3. **Validates** the confidence score of every result — if the score is below `0.65`, it retries automatically
4. **Synthesizes** a clear, coherent answer from the validated context

This is built as a three-phase evolutionary project, where each phase adds a new layer of intelligence on top of the previous one.

---

## Live Demo

```
┌─────────────────────────────────────────────────────────────────┐
│  🧠  Master Agentic RAG                        ● System Online  │
│  Intelligent Document Q&A · Self-Correction · Hybrid Routing    │
├─────────────────────────────────────┬───────────────────────────┤
│  ⚙️ Agentic Workflow                │                           │
│  ┌──────────────────────────────┐  │   [Chat messages appear   │
│  │ 1 Smart Router               │  │    here with beautiful    │
│  │ 2 Vector Retrieval           │  │    dark-themed bubbles]   │
│  │ 3 Confidence Check ≥ 0.65   │  │                           │
│  │ 4 Synthesis                  │  │                           │
│  └──────────────────────────────┘  │                           │
│                                     │                           │
│  💡 Try asking                      │  [Ask in Hebrew or       │
│  "What database is used?"           │   English…]  [Send →]    │
│  "רשימה של כל ההחלטות"             │                           │
│  "What languages does UI support?"  │                           │
└─────────────────────────────────────┴───────────────────────────┘
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE (Gradio)                      │
│              Dark Navy Theme · RTL Support · Multilingual           │
└────────────────────────────┬────────────────────────────────────────┘
                             │  query
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   MasterAgenticWorkflow (LlamaIndex)                │
│                                                                     │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────────┐   │
│  │  router_step │────▶│retrieve_step │────▶│  validate_step   │   │
│  │              │     │              │     │  (score ≥ 0.65?) │   │
│  │  Structured  │     │  Pinecone    │     │                  │   │
│  │  keywords?   │     │  Vector DB   │◀────│  NO → retry (×2) │   │
│  │      │       │     │  top-k=3     │     └────────┬─────────┘   │
│  │      │ YES   │     └──────────────┘              │ YES          │
│  │      ▼       │                                   ▼              │
│  │  JSON file   │                        ┌──────────────────┐     │
│  │  (structured │                        │ synthesis_step   │     │
│  │   data)      │                        │ Cohere R+ → text │     │
│  └──────────────┘                        └────────┬─────────┘     │
└───────────────────────────────────────────────────│─────────────────┘
                                                    │  answer
                                                    ▼
                                            ┌───────────────┐
                                            │  User receives │
                                            │  final answer  │
                                            └───────────────┘
```

### Component Flow

```
Documents (Markdown)
       │
       ▼
┌─────────────┐    embed-multilingual-v3.0    ┌───────────────┐
│  LlamaIndex │ ─────────────────────────────▶│   Pinecone    │
│  Ingestion  │                               │  Vector Store │
└─────────────┘                               └───────┬───────┘
                                                      │  semantic similarity
                                                      ▼
                                             ┌─────────────────┐
                                             │ NodeWithScore[] │
                                             │ + metadata      │
                                             └────────┬────────┘
                                                      │
                                                      ▼
                                             ┌─────────────────┐
                                             │ Cohere R+       │
                                             │ (Response       │
                                             │  Synthesizer)   │
                                             └─────────────────┘
```

---

## Features

<table>
<tr>
<td width="50%">

### Core Capabilities

- **Smart Router** — Automatically distinguishes between structured data queries (`"list all decisions"`) and free-form semantic questions
- **Self-Correction Loop** — Up to 2 automatic retries when the confidence score falls below `0.65`
- **Hybrid Indexing** — Vector embeddings enriched with metadata (Tool, File, Project)
- **Multilingual Embeddings** — Cohere `embed-multilingual-v3.0` supports Hebrew, English, and Spanish natively
- **Compact Synthesis** — LlamaIndex `response_mode="compact"` for clean, concise answers

</td>
<td width="50%">

### Developer Experience

- **Event-Driven** — Clean step-based workflow with typed events (`TextRetrievedEvent`, `ContentValidatedEvent`)
- **Async Pipeline** — Fully `async/await` for non-blocking execution
- **Pydantic Schemas** — Structured data extraction with type-safe models (`Decision`, `TechnicalRule`, `Warning`)
- **Graceful Degradation** — System reports meaningful errors if the engine fails to initialize
- **Streaming UI** — Real-time typing indicator while the agent processes

</td>
</tr>
</table>

### UI Highlights

| Feature | Details |
|---|---|
| **Theme** | Dark Navy (`#0d1526`) with Orange (`#f97316`) accent — professional & modern |
| **Font** | Inter (English) + Assistant (Hebrew) via Google Fonts |
| **Sidebar** | Live workflow visualization, system stats, clickable sample questions |
| **Chat Bubbles** | Gradient orange for user · Glass-morphism for assistant |
| **Animations** | Fade-in entrance animations, pulsing status indicator, hover effects |
| **RTL Support** | Full Hebrew right-to-left text rendering |
| **Responsive** | Adapts gracefully across screen sizes |

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Orchestration** | [LlamaIndex Workflows](https://docs.llamaindex.ai/en/stable/module_guides/workflow/) | Event-driven agentic pipeline |
| **LLM** | [Cohere `command-r-plus`](https://cohere.com/command) | Response synthesis |
| **Embeddings** | Cohere `embed-multilingual-v3.0` | Semantic vector creation |
| **Vector DB** | [Pinecone](https://pinecone.io) | Similarity search at scale |
| **UI Framework** | [Gradio 6.x](https://gradio.app) | Interactive web interface |
| **Data Validation** | [Pydantic v2](https://docs.pydantic.dev) | Schema enforcement |
| **Env Management** | `python-dotenv` | Secure API key loading |
| **Language** | Python 3.10+ | Async/await, type hints, match statements |

---

## Project Structure

```
project3-rag/
│
├── app.py                    # Main application — workflow + UI
│
├── data/                     # Source knowledge base documents
│   ├── project_info.md       # General project information
│   ├── tech_spec.md          # Technical specifications
│   ├── tasks.md              # Backlog and task tracking
│   └── ui_style_guide.md     # Design system guidelines
│
├── project_knowledge.json    # Structured data (decisions, rules, warnings)
│                             # Used by the router for structured queries
│
├── .env                      # API keys (not committed)
├── .env.example              # Example env file (safe to commit)
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

---

## How It Works

### The Agentic Loop in Detail

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  STEP 1 · router_step                               │
│                                                     │
│  Does the query contain keywords like:              │
│  "רשימה", "אזהרות", "חוקים", "סטטיסטיקה"?         │
│                                                     │
│  YES ──▶ Load project_knowledge.json ──▶ STOP       │
│  NO  ──▶ Pass to retrieve_step                      │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  STEP 2 · retrieve_step                             │
│                                                     │
│  Call Pinecone with the query embedding             │
│  Get top-3 NodeWithScore results                    │
│                                                     │
│  If 0 results AND retries < 2:                      │
│    retries += 1 ──▶ loop back (StartEvent)          │
│  If still 0 after 2 retries ──▶ STOP (no info)     │
│  If results found ──▶ TextRetrievedEvent            │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  STEP 3 · validate_step                             │
│                                                     │
│  top_score = nodes[0].score                         │
│                                                     │
│  score < 0.65 AND retries < 2:                      │
│    retries += 1 ──▶ loop back (improve query)       │
│  score < 0.65 AND retries = 2:                      │
│    STOP ("relevance too low")                       │
│  score ≥ 0.65:                                      │
│    ──▶ ContentValidatedEvent                        │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  STEP 4 · synthesis_step                            │
│                                                     │
│  synthesizer.synthesize(query, validated_nodes)     │
│  response_mode = "compact"                          │
│  ──▶ StopEvent(result=final_answer)                 │
└─────────────────────────────────────────────────────┘
    │
    ▼
 User sees the answer
```

### Confidence Threshold Explained

The `validate_step` checks `nodes[0].score` — the cosine similarity between the query embedding and the top retrieved document chunk.

| Score Range | Action |
|---|---|
| `≥ 0.65` | Accept — proceed to synthesis |
| `0.40 – 0.64` | Reject — retry with broader query (up to 2×) |
| `< 0.40` or `0 nodes` | Reject — return "no relevant info found" |

---

## Installation

### Prerequisites

- Python **3.10** or higher
- A [Pinecone](https://app.pinecone.io) account with an index named `coding-agent-rag`
- A [Cohere](https://dashboard.cohere.com) API key

### Step 1 — Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/master-agentic-rag.git
cd master-agentic-rag
```

### Step 2 — Create a virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

Key packages installed:

```
llama-index-core
llama-index-embeddings-cohere
llama-index-llms-cohere
llama-index-vector-stores-pinecone
llama-index-workflows
gradio>=6.0
pinecone-client
cohere
pydantic>=2.0
python-dotenv
```

### Step 4 — Configure environment variables

```bash
cp .env.example .env
# Then edit .env with your actual keys
```

### Step 5 — Run the app

```bash
python app.py
```

Open your browser at `http://localhost:7860`

---

## Configuration

Create a `.env` file in the project root:

```env
# ── Required ──────────────────────────────────────────
COHERE_API_KEY=your_cohere_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here

# ── Optional ───────────────────────────────────────────
# PINECONE_INDEX_NAME=coding-agent-rag   # default
# SIMILARITY_TOP_K=3                     # number of chunks to retrieve
# CONFIDENCE_THRESHOLD=0.65              # minimum score to accept
# MAX_RETRIES=2                          # self-correction retry limit
# WORKFLOW_TIMEOUT=60                    # seconds before timeout
```

### Pinecone Index Setup

Your Pinecone index must be configured with:

| Setting | Value |
|---|---|
| **Index Name** | `coding-agent-rag` |
| **Dimensions** | `1024` (Cohere multilingual-v3.0) |
| **Metric** | `cosine` |
| **Cloud** | Any (AWS, GCP, Azure) |

To ingest your documents into Pinecone, run your indexing script that reads from the `data/` directory.

---

## Usage

### Running the UI

```bash
python app.py
# → Gradio server starts at http://0.0.0.0:7860
```

### Example Queries

**Hebrew (structured routing)**
```
רשימה של כל ההחלטות הטכניות
מה האזהרות הקיימות במערכת?
כל החוקים הטכניים
```

**English (semantic search)**
```
What database is used in the project?
What languages does the UI support?
Which cloud provider stores the documents?
What is the production database restriction?
```

**Mixed language**
```
What is the צבע נבחר for the UI?
Tell me about the RTL הנחיות
```

---

## The Three Phases

This project was built incrementally through three evolutionary phases:

### Phase A — RAG MVP

> *"Get something working end-to-end"*

- Established the core pipeline: document loading → embedding → Pinecone indexing → query → answer
- Used LlamaIndex `VectorStoreIndex` with basic `as_query_engine()`
- Connected Cohere multilingual embeddings for Hebrew/English support
- Basic Gradio interface

### Phase B — Event-Driven Workflow

> *"Make it intelligent and observable"*

- Rewrote the pipeline as a `LlamaIndex Workflow` with typed `Event` classes
- Introduced explicit `@step` decorators for each processing stage
- Added `Context` (shared state) across steps for retry counting
- Added the **self-correction loop**: if confidence is low, the agent retries automatically
- The workflow is now observable — each step prints its action and score

### Phase C — Structured Routing + Data Extraction

> *"Give the agent two brains: one for fuzzy, one for exact"*

- Added the `router_step` that intercepts structured queries (lists, statistics, rules)
- Structured queries bypass Pinecone and read from a local `project_knowledge.json`
- Defined Pydantic schemas (`Decision`, `TechnicalRule`, `Warning`) for typed extraction
- The agent now "thinks" about which retrieval path to take before doing any work

---

## Data Schema

The `project_knowledge.json` file stores structured project data used by the router:

```json
{
  "decisions": [
    {
      "id": "DEC-001",
      "title": "Database Selection",
      "summary": "Chose PostgreSQL for complex relational queries",
      "observed_at": "2026-01-15"
    }
  ],
  "rules": [
    {
      "id": "RULE-001",
      "rule": "Do not use SQLite in production",
      "scope": "Infrastructure"
    }
  ],
  "warnings": [
    {
      "id": "WARN-001",
      "area": "Chatbot CSS",
      "message": "Do not modify chatbot CSS without consulting the product designer",
      "severity": "high"
    }
  ]
}
```

### Pydantic Models

```python
class Decision(BaseModel):
    id: str
    title: str          # Technical decision title
    summary: str        # Decision summary
    observed_at: str    # YYYY-MM-DD

class TechnicalRule(BaseModel):
    id: str
    rule: str           # The rule or guideline
    scope: str          # Impact area (UI, Security, API, Infrastructure)

class Warning(BaseModel):
    id: str
    area: str           # Sensitive system area
    message: str        # Warning content
    severity: str       # "high" | "medium" | "low"
```

---

## Troubleshooting

<details>
<summary><b>System shows "offline" on startup</b></summary>

The engine failed to initialize. Check:
1. `.env` file exists and contains valid keys
2. Your Pinecone index `coding-agent-rag` exists and has data
3. Network connectivity (Pinecone and Cohere are cloud services)
4. Run `python app.py` in terminal to see the full error message

</details>

<details>
<summary><b>"No relevant information found" for every query</b></summary>

Your Pinecone index may be empty or the embeddings mismatched:
1. Verify the index has vectors: check the Pinecone dashboard
2. Make sure you indexed using the same model: `embed-multilingual-v3.0`
3. The index dimension must be `1024` (not 1536 which is OpenAI)
4. Try lowering `CONFIDENCE_THRESHOLD` to `0.5` in `.env` for testing

</details>

<details>
<summary><b>SSL errors on Windows</b></summary>

The app disables SSL verification by default (`ssl._create_unverified_context`). This is already handled in the code for environments like university networks with certificate inspection.

</details>

<details>
<summary><b>Gradio "asyncio" error</b></summary>

The respond function uses `async def`. Gradio 6+ handles async generators natively. If you see an error, ensure you have `gradio>=6.0`:

```bash
pip install --upgrade gradio
```

</details>

<details>
<summary><b>Hebrew text displays incorrectly</b></summary>

Ensure your terminal and browser support UTF-8. All JSON files are read with `encoding="utf-8"`. The Gradio UI loads the `Assistant` font from Google Fonts for proper Hebrew rendering.

</details>

---

## Contributing

Contributions are welcome! Here's how to get started:

```bash
# 1. Fork the repository
# 2. Create a feature branch
git checkout -b feature/your-feature-name

# 3. Make your changes
# 4. Test thoroughly
python app.py

# 5. Commit with a descriptive message
git commit -m "feat: add streaming response support"

# 6. Push and open a Pull Request
git push origin feature/your-feature-name
```

### Ideas for Contribution

- [ ] Add document ingestion pipeline (drag & drop PDF/Markdown)
- [ ] Add streaming token output from Cohere
- [ ] Implement query history with session persistence
- [ ] Add a confidence score display in the chat UI
- [ ] Support additional file types (DOCX, HTML, CSV)
- [ ] Add Spanish language routing keywords
- [ ] Write unit tests for each workflow step

---

## License

This project is licensed under the **MIT License**.

```
MIT License · Copyright (c) 2026

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, subject to the following conditions...
```

---

<div align="center">

<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=100&section=footer&animation=fadeIn" />

**Built with precision, curiosity, and a lot of async/await**

[![LlamaIndex](https://img.shields.io/badge/LlamaIndex-Docs-7C3AED?style=flat-square)](https://docs.llamaindex.ai)
[![Cohere](https://img.shields.io/badge/Cohere-Docs-coral?style=flat-square)](https://docs.cohere.com)
[![Pinecone](https://img.shields.io/badge/Pinecone-Docs-00B4B4?style=flat-square)](https://docs.pinecone.io)
[![Gradio](https://img.shields.io/badge/Gradio-Docs-F97316?style=flat-square)](https://www.gradio.app/docs)

</div>
