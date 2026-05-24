import os
import ssl
import urllib3
import json
import gradio as gr
from typing import List
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from pinecone import Pinecone

from llama_index.core import VectorStoreIndex, get_response_synthesizer, Settings
from llama_index.core.workflow import Event, Workflow, step, StartEvent, StopEvent, Context
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.llms.cohere import Cohere
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import NodeWithScore

ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

load_dotenv()

Settings.llm = Cohere(api_key=os.getenv("COHERE_API_KEY"), model="command-r-plus")
Settings.embed_model = CohereEmbedding(
    api_key=os.getenv("COHERE_API_KEY"), model_name="embed-multilingual-v3.0"
)


# ── Data schemas ──────────────────────────────────────────────────────────────

class Decision(BaseModel):
    id: str
    title: str = Field(description="כותרת ההחלטה הטכנית")
    summary: str = Field(description="תקציר ההחלטה")
    observed_at: str = Field(description="תאריך זיהוי (YYYY-MM-DD)")

class TechnicalRule(BaseModel):
    id: str
    rule: str = Field(description="הכלל או ההנחיה")
    scope: str = Field(description="תחום ההשפעה (UI, Security, API)")

class Warning(BaseModel):
    id: str
    area: str = Field(description="האזור הרגיש במערכת")
    message: str = Field(description="תוכן האזהרה")
    severity: str = Field(description="רמת חומרה (high/medium/low)")

class ProjectData(BaseModel):
    decisions: List[Decision]
    rules: List[TechnicalRule]
    warnings: List[Warning]


# ── Workflow events ───────────────────────────────────────────────────────────

class TextRetrievedEvent(Event):
    nodes: list[NodeWithScore]
    query: str

class ContentValidatedEvent(Event):
    nodes: list[NodeWithScore]
    query: str


# ── Agentic workflow ──────────────────────────────────────────────────────────

class MasterAgenticWorkflow(Workflow):
    def __init__(self, engine_tools, **kwargs):
        super().__init__(**kwargs)
        self.retriever = engine_tools["retriever"]
        self.synthesizer = engine_tools["synthesizer"]

    @step
    async def router_step(self, ctx: Context, ev: StartEvent) -> TextRetrievedEvent | StopEvent:
        query = ev.get("query").lower()
        ctx.data["original_query"] = query

        structured_keywords = ["רשימה", "כל ההחלטות", "אזהרות", "חוקים", "מה השתנה", "סטטיסטיקה"]

        if any(word in query for word in structured_keywords):
            try:
                with open("project_knowledge.json", "r", encoding="utf-8") as f:
                    data = json.load(f)
                response = (
                    "להלן הנתונים המובנים שמצאתי בתיעוד הפרויקט:\n"
                    + json.dumps(data, indent=2, ensure_ascii=False)
                )
                return StopEvent(result=response)
            except FileNotFoundError:
                pass

        return await self.retrieve_step(ctx, ev)

    @step
    async def retrieve_step(self, ctx: Context, ev: StartEvent) -> TextRetrievedEvent | StopEvent:
        query = ev.get("query")
        retries = await ctx.data.get("retries", 0)

        nodes = self.retriever.retrieve(query)

        if not nodes:
            if retries < 2:
                ctx.data["retries"] = retries + 1
                return StartEvent(query=query)
            return StopEvent(result="מצטער, לא נמצא מידע רלוונטי בתיעוד.")

        return TextRetrievedEvent(nodes=nodes, query=query)

    @step
    async def validate_step(self, ctx: Context, ev: TextRetrievedEvent) -> ContentValidatedEvent | StartEvent:
        top_score = ev.nodes[0].score

        if top_score < 0.65:
            retries = await ctx.data.get("retries", 0)
            if retries < 2:
                ctx.data["retries"] = retries + 1
                return StartEvent(query=ev.query)
            return StopEvent(result="מצאתי מידע, אך רמת הרלוונטיות שלו נמוכה מדי.")

        return ContentValidatedEvent(nodes=ev.nodes, query=ev.query)

    @step
    async def synthesis_step(self, ev: ContentValidatedEvent) -> StopEvent:
        response = self.synthesizer.synthesize(ev.query, ev.nodes)
        return StopEvent(result=str(response))


# ── Engine setup ──────────────────────────────────────────────────────────────

def setup_engine():
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"), ssl_verify=False)
    pinecone_index = pc.Index("coding-agent-rag")
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    index = VectorStoreIndex.from_vector_store(vector_store, embed_model=Settings.embed_model)
    return {
        "retriever": VectorIndexRetriever(index=index, similarity_top_k=3),
        "synthesizer": get_response_synthesizer(response_mode="compact"),
    }

try:
    engine_config = setup_engine()
    agent = MasterAgenticWorkflow(engine_config, timeout=60)
    SYSTEM_ONLINE = True
except Exception as e:
    print(f"Initialization error: {e}")
    agent = None
    SYSTEM_ONLINE = False


# ── Chat logic ────────────────────────────────────────────────────────────────

async def chat_process(message: str, history: list) -> str:
    if not agent:
        return "System is offline. Please check your API keys and connection."
    result = await agent.run(query=message)
    return str(result)


# ── UI Styles & HTML ──────────────────────────────────────────────────────────

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Assistant:wght@300;400;600;700&display=swap');

/* ── Variables ── */
:root {
    --bg-deep:      #070c18;
    --bg-base:      #0d1526;
    --bg-card:      #111d35;
    --bg-glass:     rgba(255,255,255,0.04);
    --border:       rgba(255,255,255,0.07);
    --border-hover: rgba(249,115,22,0.45);
    --orange:       #f97316;
    --orange-light: #fb923c;
    --orange-glow:  rgba(249,115,22,0.25);
    --green:        #10b981;
    --yellow:       #f59e0b;
    --purple:       #818cf8;
    --text-1:       #e2e8f0;
    --text-2:       #94a3b8;
    --text-3:       #475569;
    --radius-lg:    16px;
    --radius-md:    12px;
    --radius-sm:    8px;
}

/* ── Reset & base ── */
*, *::before, *::after { box-sizing: border-box; }

body,
.gradio-container {
    background: var(--bg-deep) !important;
    font-family: 'Inter', 'Assistant', sans-serif !important;
    color: var(--text-1) !important;
    min-height: 100vh;
}

.gradio-container {
    max-width: 1280px !important;
    margin: 0 auto !important;
    padding: 20px !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--bg-card); border-radius: 99px; }
::-webkit-scrollbar-thumb:hover { background: var(--orange); }

/* ── Hero header ── */
#hero-header {
    background: linear-gradient(135deg, var(--bg-card) 0%, #162040 100%);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 28px 36px 24px;
    margin-bottom: 20px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 24px 64px rgba(0,0,0,0.5);
}

#hero-header::before {
    content: '';
    position: absolute;
    inset: 0;
    background:
        radial-gradient(ellipse 60% 80% at 10% 60%, rgba(249,115,22,0.07) 0%, transparent 70%),
        radial-gradient(ellipse 40% 60% at 90% 20%, rgba(129,140,248,0.05) 0%, transparent 70%);
    pointer-events: none;
}

.hero-title {
    font-size: 1.75rem;
    font-weight: 700;
    color: var(--text-1);
    letter-spacing: -0.5px;
    margin: 0;
    line-height: 1.2;
}

.hero-sub {
    font-size: 0.875rem;
    color: var(--text-2);
    font-weight: 300;
    margin: 4px 0 0 0;
}

.logo-box {
    width: 52px; height: 52px;
    background: linear-gradient(135deg, var(--orange), var(--orange-light));
    border-radius: 14px;
    display: flex; align-items: center; justify-content: center;
    font-size: 26px;
    box-shadow: 0 6px 20px var(--orange-glow), 0 0 0 1px rgba(249,115,22,0.3);
    flex-shrink: 0;
}

.badge-row {
    display: flex; gap: 8px; flex-wrap: wrap; margin-top: 18px;
    position: relative; z-index: 1;
}

.badge {
    display: inline-flex; align-items: center; gap: 5px;
    background: rgba(255,255,255,0.05);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.72rem;
    color: var(--text-2);
    font-weight: 500;
    transition: all 0.25s ease;
}

.badge:hover {
    border-color: var(--orange);
    color: var(--orange-light);
    box-shadow: 0 0 12px var(--orange-glow);
}

/* ── Status pill ── */
.status-pill {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(16,185,129,0.1);
    border: 1px solid rgba(16,185,129,0.25);
    border-radius: 20px;
    padding: 4px 10px;
    font-size: 0.72rem; font-weight: 600;
    color: var(--green);
}

.status-pill.offline {
    background: rgba(239,68,68,0.1);
    border-color: rgba(239,68,68,0.25);
    color: #ef4444;
}

.dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: currentColor;
    animation: blink 2s ease-in-out infinite;
}

@keyframes blink {
    0%,100% { opacity: 1; transform: scale(1); }
    50%      { opacity: 0.4; transform: scale(0.7); }
}

/* ── Sidebar cards ── */
.side-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 18px 20px;
    margin-bottom: 14px;
    transition: border-color 0.25s ease, box-shadow 0.25s ease;
}

.side-card:hover {
    border-color: rgba(249,115,22,0.2);
    box-shadow: 0 0 24px rgba(249,115,22,0.06);
}

.card-title {
    font-size: 0.7rem;
    font-weight: 600;
    color: var(--text-3);
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin: 0 0 14px 0;
    display: flex; align-items: center; gap: 6px;
}

/* ── Workflow steps ── */
.wf-step {
    display: flex; align-items: flex-start; gap: 10px;
    padding: 9px 0;
    border-bottom: 1px solid var(--border);
}
.wf-step:last-child { border-bottom: none; padding-bottom: 0; }
.wf-step:first-child { padding-top: 0; }

.step-num {
    width: 26px; height: 26px; border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.7rem; font-weight: 700;
    flex-shrink: 0; margin-top: 1px;
}

.step-label { font-size: 0.8rem; font-weight: 500; color: var(--text-1); }
.step-desc  { font-size: 0.7rem; color: var(--text-3); margin-top: 1px; }

/* ── Stat rows ── */
.stat-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 5px 0;
    border-bottom: 1px solid var(--border);
    font-size: 0.77rem;
}
.stat-row:last-child { border-bottom: none; }
.stat-key { color: var(--text-3); }
.stat-val { font-weight: 500; }

/* ── Sample question chips ── */
.q-chip {
    background: rgba(249,115,22,0.06);
    border: 1px solid rgba(249,115,22,0.18);
    border-radius: var(--radius-sm);
    padding: 8px 11px;
    font-size: 0.75rem;
    color: var(--text-2);
    cursor: pointer;
    transition: all 0.2s ease;
    line-height: 1.4;
}
.q-chip:hover {
    background: rgba(249,115,22,0.12);
    border-color: rgba(249,115,22,0.4);
    color: var(--orange-light);
}

/* ── Chatbot panel ── */
#chatbot-wrap .gradio-chatbot,
.chatbot-panel {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-lg) !important;
    box-shadow: 0 8px 32px rgba(0,0,0,0.35) !important;
    overflow: hidden !important;
}

/* Chat bubbles */
.message.user .bubble-wrap {
    background: linear-gradient(135deg, var(--orange), var(--orange-light)) !important;
    border-radius: 18px 18px 4px 18px !important;
    box-shadow: 0 4px 14px var(--orange-glow) !important;
    color: #fff !important;
}

.message.bot .bubble-wrap,
.message.assistant .bubble-wrap {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid var(--border) !important;
    border-radius: 18px 18px 18px 4px !important;
    color: var(--text-1) !important;
}

/* ── Input area ── */
#input-row {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 10px 12px;
    margin-top: 12px;
    display: flex;
    gap: 10px;
    align-items: flex-end;
    transition: border-color 0.25s ease, box-shadow 0.25s ease;
}

#input-row:focus-within {
    border-color: rgba(249,115,22,0.4);
    box-shadow: 0 0 0 3px rgba(249,115,22,0.08);
}

#msg-input textarea,
#msg-input input {
    background: transparent !important;
    color: var(--text-1) !important;
    border: none !important;
    outline: none !important;
    font-family: 'Inter', 'Assistant', sans-serif !important;
    font-size: 0.92rem !important;
    resize: none !important;
    box-shadow: none !important;
}

#msg-input textarea::placeholder { color: var(--text-3) !important; }
#msg-input label { display: none !important; }

/* ── Buttons ── */
#send-btn {
    background: linear-gradient(135deg, var(--orange), var(--orange-light)) !important;
    border: none !important;
    border-radius: 10px !important;
    color: #fff !important;
    font-weight: 600 !important;
    font-size: 0.875rem !important;
    padding: 10px 20px !important;
    cursor: pointer !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 4px 14px var(--orange-glow) !important;
    min-width: 80px !important;
}

#send-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px var(--orange-glow) !important;
}

#send-btn:active { transform: translateY(0) !important; }

#clear-btn {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text-2) !important;
    font-weight: 500 !important;
    font-size: 0.875rem !important;
    padding: 10px 16px !important;
    cursor: pointer !important;
    transition: all 0.25s ease !important;
}

#clear-btn:hover {
    background: rgba(239,68,68,0.1) !important;
    border-color: rgba(239,68,68,0.3) !important;
    color: #ef4444 !important;
}

/* ── Footer ── */
#footer-bar {
    background: var(--bg-glass);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 10px 20px;
    margin-top: 14px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.72rem;
    color: var(--text-3);
}

/* ── Animations ── */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}

.animate-in { animation: fadeInUp 0.55s cubic-bezier(.22,.68,0,1.1) both; }
.delay-1 { animation-delay: 0.08s; }
.delay-2 { animation-delay: 0.16s; }
.delay-3 { animation-delay: 0.24s; }

/* ── Gradio overrides ── */
.gr-group, .gr-box { border: none !important; background: transparent !important; }
footer { display: none !important; }

/* Hide default Gradio labels we don't want */
.chatbot > .label { display: none !important; }
"""

STATUS_CLASS = "status-pill" if SYSTEM_ONLINE else "status-pill offline"
STATUS_ICON  = "●" if SYSTEM_ONLINE else "●"
STATUS_TEXT  = "System Online" if SYSTEM_ONLINE else "System Offline"

HEADER_HTML = f"""
<div id="hero-header" class="animate-in">
  <div style="display:flex; justify-content:space-between; align-items:flex-start; position:relative; z-index:1;">
    <div style="display:flex; align-items:center; gap:14px;">
      <div class="logo-box">🧠</div>
      <div>
        <h1 class="hero-title">Master Agentic RAG</h1>
        <p class="hero-sub">Intelligent Document Q&amp;A · Self-Correction · Hybrid Routing</p>
      </div>
    </div>
    <div style="display:flex; flex-direction:column; align-items:flex-end; gap:6px;">
      <div class="{STATUS_CLASS}">
        <span class="dot"></span>
        {STATUS_TEXT}
      </div>
      <span style="font-size:0.68rem; color:var(--text-3);">v2.0 · LlamaIndex Workflow</span>
    </div>
  </div>

  <div class="badge-row">
    <span class="badge">🔍 Semantic Search</span>
    <span class="badge">🔄 Self-Correction</span>
    <span class="badge">📊 Smart Router</span>
    <span class="badge">🌐 Multilingual (HE / EN)</span>
    <span class="badge">⚡ Pinecone Vector DB</span>
    <span class="badge">🤖 Cohere command-r+</span>
    <span class="badge">🏗️ Event-Driven</span>
  </div>
</div>
"""

SIDEBAR_HTML = """
<div style="height:100%;">

  <!-- Workflow Steps -->
  <div class="side-card animate-in delay-1">
    <p class="card-title">⚙️ Agentic Workflow</p>

    <div class="wf-step">
      <div class="step-num" style="background:rgba(249,115,22,0.15); color:#f97316;">1</div>
      <div>
        <div class="step-label">Smart Router</div>
        <div class="step-desc">Detects structured vs. semantic queries</div>
      </div>
    </div>

    <div class="wf-step">
      <div class="step-num" style="background:rgba(16,185,129,0.15); color:#10b981;">2</div>
      <div>
        <div class="step-label">Vector Retrieval</div>
        <div class="step-desc">Top-3 semantic search in Pinecone</div>
      </div>
    </div>

    <div class="wf-step">
      <div class="step-num" style="background:rgba(245,158,11,0.15); color:#f59e0b;">3</div>
      <div>
        <div class="step-label">Confidence Check</div>
        <div class="step-desc">Validates score ≥ 0.65, retries if low</div>
      </div>
    </div>

    <div class="wf-step">
      <div class="step-num" style="background:rgba(129,140,248,0.15); color:#818cf8;">4</div>
      <div>
        <div class="step-label">Synthesis</div>
        <div class="step-desc">Composes a coherent final answer</div>
      </div>
    </div>
  </div>

  <!-- Sample Questions -->
  <div class="side-card animate-in delay-2">
    <p class="card-title">💡 Try asking</p>
    <div style="display:flex; flex-direction:column; gap:7px;">
      <div class="q-chip">"What database is used in the project?"</div>
      <div class="q-chip">"רשימה של כל ההחלטות הטכניות"</div>
      <div class="q-chip">"What languages does the UI support?"</div>
      <div class="q-chip">"מה האזהרות הקיימות במערכת?"</div>
    </div>
  </div>

  <!-- System Stats -->
  <div class="side-card animate-in delay-3">
    <p class="card-title">📈 System Config</p>
    <div style="display:flex; flex-direction:column; gap:1px;">
      <div class="stat-row">
        <span class="stat-key">Vector Store</span>
        <span class="stat-val" style="color:#10b981;">Pinecone</span>
      </div>
      <div class="stat-row">
        <span class="stat-key">LLM</span>
        <span class="stat-val" style="color:#10b981;">command-r-plus</span>
      </div>
      <div class="stat-row">
        <span class="stat-key">Embeddings</span>
        <span class="stat-val" style="color:#10b981;">multilingual-v3.0</span>
      </div>
      <div class="stat-row">
        <span class="stat-key">Top-K</span>
        <span class="stat-val" style="color:#818cf8;">3 results</span>
      </div>
      <div class="stat-row">
        <span class="stat-key">Min. Confidence</span>
        <span class="stat-val" style="color:#f59e0b;">0.65</span>
      </div>
      <div class="stat-row">
        <span class="stat-key">Max Retries</span>
        <span class="stat-val" style="color:#f97316;">2</span>
      </div>
    </div>
  </div>

</div>
"""

PLACEHOLDER_HTML = """
<div style="display:flex; flex-direction:column; align-items:center; justify-content:center;
            height:100%; padding:48px 24px; text-align:center; gap:14px;">
  <div style="font-size:3.5rem; filter:drop-shadow(0 0 20px rgba(249,115,22,0.3));">🧠</div>
  <div style="font-size:1.05rem; font-weight:600; color:#94a3b8;">
    Ask me anything about the project
  </div>
  <div style="font-size:0.82rem; color:#475569; max-width:320px; line-height:1.6;">
    I'll search the documentation, validate the results, and return a precise answer
    — in Hebrew or English.
  </div>
  <div style="display:flex; gap:8px; flex-wrap:wrap; justify-content:center; margin-top:6px;">
    <span style="background:rgba(249,115,22,0.08); border:1px solid rgba(249,115,22,0.2);
                 border-radius:20px; padding:4px 12px; font-size:0.7rem; color:#94a3b8;">
      Semantic Search
    </span>
    <span style="background:rgba(16,185,129,0.08); border:1px solid rgba(16,185,129,0.2);
                 border-radius:20px; padding:4px 12px; font-size:0.7rem; color:#94a3b8;">
      Self-Correcting
    </span>
    <span style="background:rgba(129,140,248,0.08); border:1px solid rgba(129,140,248,0.2);
                 border-radius:20px; padding:4px 12px; font-size:0.7rem; color:#94a3b8;">
      Multilingual
    </span>
  </div>
</div>
"""

FOOTER_HTML = """
<div id="footer-bar">
  <div>
    Built with&nbsp;
    <span style="color:#f97316; font-weight:500;">LlamaIndex</span>&nbsp;·&nbsp;
    <span style="color:#f97316; font-weight:500;">Gradio</span>&nbsp;·&nbsp;
    <span style="color:#f97316; font-weight:500;">Cohere</span>&nbsp;·&nbsp;
    <span style="color:#f97316; font-weight:500;">Pinecone</span>
  </div>
  <div>Phase A + B + C · Agentic Self-Correcting RAG Pipeline · 2026</div>
</div>
"""


# ── Gradio app ────────────────────────────────────────────────────────────────

theme = gr.themes.Base(
    primary_hue=gr.themes.colors.orange,
    secondary_hue=gr.themes.colors.blue,
    neutral_hue=gr.themes.colors.slate,
    font=gr.themes.GoogleFont("Inter"),
).set(
    body_background_fill="transparent",
    background_fill_primary="transparent",
    background_fill_secondary="transparent",
    border_color_primary="rgba(255,255,255,0.07)",
    color_accent="#f97316",
    button_primary_background_fill="linear-gradient(135deg,#f97316,#fb923c)",
    button_primary_text_color="#ffffff",
    button_primary_shadow="0 4px 14px rgba(249,115,22,0.35)",
    input_background_fill="transparent",
    input_border_color="transparent",
    chatbot_text_size="md",
)

with gr.Blocks(title="Master Agentic RAG") as demo:

    # Header
    gr.HTML(HEADER_HTML)

    with gr.Row(equal_height=True):

        # ── Sidebar ──────────────────────────────────────────────────────────
        with gr.Column(scale=1, min_width=260):
            gr.HTML(SIDEBAR_HTML)

        # ── Chat panel ───────────────────────────────────────────────────────
        with gr.Column(scale=3, elem_id="chatbot-wrap"):

            chatbot = gr.Chatbot(
                value=[],
                height=530,
                show_label=False,
                placeholder=PLACEHOLDER_HTML,
                elem_classes=["chatbot-panel"],
            )

            with gr.Row(elem_id="input-row"):
                msg_input = gr.Textbox(
                    placeholder="Ask in Hebrew or English…  (Shift+Enter for new line)",
                    show_label=False,
                    scale=7,
                    container=False,
                    lines=1,
                    max_lines=5,
                    elem_id="msg-input",
                    autofocus=True,
                )
                send_btn = gr.Button(
                    "Send →",
                    variant="primary",
                    scale=1,
                    min_width=80,
                    elem_id="send-btn",
                )
                clear_btn = gr.Button(
                    "Clear",
                    variant="secondary",
                    scale=1,
                    min_width=60,
                    elem_id="clear-btn",
                )

    # Footer
    gr.HTML(FOOTER_HTML)

    # ── Event handlers ────────────────────────────────────────────────────────

    async def respond(message: str, history: list):
        if not message.strip():
            yield "", history
            return

        history = history + [[message, "⏳ Searching documentation…"]]
        yield "", history

        response = await chat_process(message, [])
        history[-1] = [message, response]
        yield "", history

    send_btn.click(
        fn=respond,
        inputs=[msg_input, chatbot],
        outputs=[msg_input, chatbot],
    )

    msg_input.submit(
        fn=respond,
        inputs=[msg_input, chatbot],
        outputs=[msg_input, chatbot],
    )

    clear_btn.click(
        fn=lambda: ([], ""),
        outputs=[chatbot, msg_input],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        css=CUSTOM_CSS,
        theme=theme,
    )
