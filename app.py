import os
import ssl
import urllib3
import json
import gradio as gr
from typing import List, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from pinecone import Pinecone

# ייבויי LlamaIndex
from llama_index.core import VectorStoreIndex, get_response_synthesizer
from llama_index.core.workflow import Event, Workflow, step, StartEvent, StopEvent, Context
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.core.program import LLMTextCompletionProgram
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.tools import QueryEngineTool

# --- הגדרות אבטחה לנטפרי ---
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

load_dotenv()

# --- 1. הגדרת סכמת הנתונים (שלב ג' - Data Extraction) ---
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

# --- 2. הגדרת אירועים (שלב ב' - Event Driven) ---
class TextRetrievedEvent(Event):
    nodes: list[NodeWithScore]
    query: str

class ContentValidatedEvent(Event):
    nodes: list[NodeWithScore]
    query: str

# --- 3. ה-Workflow המרכזי (המוח של הסוכן) ---
class MasterAgenticWorkflow(Workflow):
    def __init__(self, engine_tools, **kwargs):
        super().__init__(**kwargs)
        self.retriever = engine_tools['retriever']
        self.synthesizer = engine_tools['synthesizer']
        
    @step
    async def router_step(self, ctx: Context, ev: StartEvent) -> TextRetrievedEvent | StopEvent:
        """שלב ג' - ניתוב חכם בין חיפוש סמנטי לנתונים מובנים"""
        query = ev.get("query").lower()
        await ctx.set("original_query", query)
        
        # מילות מפתח לניתוב ל-JSON
        structured_keywords = ["רשימה", "כל ההחלטות", "אזהרות", "חוקים", "מה השתנה", "סטטיסטיקה"]
        
        if any(word in query for word in structured_keywords):
            print("📊 Router: מנתב לשליפה מהנתונים המובנים (JSON)...")
            try:
                with open("project_knowledge.json", "r", encoding="utf-8") as f:
                    data = json.load(f)
                # ה-LLM מעבד את ה-JSON לתשובה קריאה
                response = f"להלן הנתונים המובנים שמצאתי בתיעוד הפרויקט:\n{json.dumps(data, indent=2, ensure_ascii=False)}"
                return StopEvent(result=response)
            except FileNotFoundError:
                print("⚠️ קובץ JSON חסר, עובר לחיפוש סמנטי...")

        # אם לא מובנה - עובר לחיפוש סמנטי (שלב א')
        return await self.retrieve_step(ctx, ev)

    @step
    async def retrieve_step(self, ctx: Context, ev: StartEvent) -> TextRetrievedEvent | StopEvent:
        """שלב א' + ב': שליפה מה-Vector Store עם לולאת תיקון"""
        query = ev.get("query")
        retries = await ctx.get("retries", default=0)
        
        print(f"🔎 ניסיון שליפה #{retries + 1} עבור: {query}")
        nodes = self.retriever.retrieve(query)
        
        if not nodes:
            if retries < 2:
                await ctx.set("retries", retries + 1)
                return StartEvent(query=query) # ניסיון חוזר
            return StopEvent(result="מצטער, לא נמצא מידע רלוונטי בתיעוד.")
            
        return TextRetrievedEvent(nodes=nodes, query=query)

    @step
    async def validate_step(self, ctx: Context, ev: TextRetrievedEvent) -> ContentValidatedEvent | StartEvent:
        """שלב ב' - ולידציה של רמת ביטחון (Confidence)"""
        top_score = ev.nodes[0].score
        print(f"📊 Confidence Score: {top_score:.2f}")
        
        if top_score < 0.65:
            retries = await ctx.get("retries", default=0)
            if retries < 2:
                print("🔄 ביטחון נמוך, מנסה לשפר חיפוש...")
                await ctx.set("retries", retries + 1)
                return StartEvent(query=ev.query)
            return StopEvent(result="מצאתי מידע, אך רמת הרלוונטיות שלו נמוכה מדי.")
            
        return ContentValidatedEvent(nodes=ev.nodes, query=ev.query)

    @step
    async def synthesis_step(self, ev: ContentValidatedEvent) -> StopEvent:
        """שלב א' - יצירת תשובה סופית"""
        print("✍️ מנסח תשובה סופית...")
        response = self.synthesizer.synthesize(ev.query, ev.nodes)
        return StopEvent(result=str(response))

# --- 4. פונקציות עזר ואתחול ---
def setup_engine():
    embed_model = CohereEmbedding(api_key=os.getenv("COHERE_API_KEY"), model_name="embed-multilingual-v3.0")
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"), ssl_verify=False)
    pinecone_index = pc.Index("coding-agent-rag")
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)
    
    return {
        'retriever': VectorIndexRetriever(index=index, similarity_top_k=3),
        'synthesizer': get_response_synthesizer(response_mode="compact")
    }

# יצירת הסוכן
try:
    engine_config = setup_engine()
    agent = MasterAgenticWorkflow(engine_config, timeout=60)
except Exception as e:
    print(f"Error initializing: {e}")
    agent = None

# --- 5. ממשק GRADIO מעוצב ---
async def chat_process(message, history):
    if not agent:
        return "המערכת לא אותחלה. בדקי חיבור ו-API Keys."
    result = await agent.run(query=message)
    return str(result)

with gr.Blocks(theme=gr.themes.Soft(), title="Master Agentic RAG") as demo:
    gr.Markdown("""
    # 🤖 Master Agentic RAG Assistant
    ### מערכת תשאול חכמה: Routing | Self-Correction | Data Extraction
    ---
    """)
    
    with gr.Row():
        with gr.Column(scale=4):
            chatbot = gr.ChatInterface(
                fn=chat_process,
                textbox=gr.Textbox(placeholder="שאלי אותי כל דבר על הפרויקט...", container=False, scale=7),
                submit_btn="שלח 🚀",
                retry_btn="נסה שוב 🔄",
                undo_btn="מחק ↩️",
                clear_btn="נקה הכל 🗑️",
            )
        
    gr.Markdown("""
    <div style='text-align: center; font-size: 0.9em; color: #555;'>
    <b>שלב א'</b>: RAG סמנטי | <b>שלב ב'</b>: Workflow & Validation | <b>שלב ג'</b>: Structured Routing
    </div>
    """)

if __name__ == "__main__":
    demo.launch()