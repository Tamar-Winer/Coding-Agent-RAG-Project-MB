import os
import ssl
import urllib3
import gradio as gr
from dotenv import load_dotenv

# --- הגדרות אבטחה לנטפרי (יוסרו אוטומטית כשהחסימה תשתחרר) ---
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from pinecone import Pinecone
from llama_index.core import (
    SimpleDirectoryReader, 
    StorageContext, 
    VectorStoreIndex, 
    get_response_synthesizer
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.ingestion import IngestionPipeline

# 1. טעינת משתני סביבה
load_dotenv()

def initialize_rag():
    print("🔄 מתחיל תהליך סנכרון נתונים...")
    
    # א. טעינת מסמכים עם Metadata
    documents = SimpleDirectoryReader("./data").load_data()
    for doc in documents:
        doc.metadata["ai_tool"] = "Cursor/Claude Code"
        doc.metadata["project_phase"] = "MVP"

    # ב. הגדרת מודלים
    embed_model = CohereEmbedding(
        api_key=os.getenv("COHERE_API_KEY"), 
        model_name="embed-multilingual-v3.0"
    )
    parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)

    # ג. חיבור ל-Pinecone
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"), ssl_verify=False)
    pinecone_index = pc.Index("coding-agent-rag")
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    
    # ד. מנגנון סנכרון (Ingestion Pipeline)
    # הפייפליין דואג לעבד רק מסמכים חדשים או כאלו שהשתנו
    pipeline = IngestionPipeline(
        transformations=[parser, embed_model],
        vector_store=vector_store,
    )
    
    # הרצת הסנכרון
    print("🚀 מסנכרן קבצים מול Pinecone (מעלה רק שינויים)...")
    pipeline.run(documents=documents)

    # ה. בניית ה-Query Engine (תשאול והסקה)
    index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)
    
    retriever = VectorIndexRetriever(index=index, similarity_top_k=3)
    postprocessor = SimilarityPostprocessor(similarity_cutoff=0.7)
    response_synthesizer = get_response_synthesizer(response_mode="compact")
    
    return RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
        node_postprocessors=[postprocessor]
    )

# אתחול המנוע
try:
    query_engine = initialize_rag()
    print("✅ המערכת מוכנה לעבודה!")
except Exception as e:
    print(f"⚠️ שגיאה באתחול (ייתכן בגלל חסימת נטפרי): {e}")
    query_engine = None

# 2. ממשק Gradio מעוצב
def predict(message, history):
    if query_engine is None:
        return "שגיאה: המערכת לא הצליחה להתחבר לבסיס הנתונים. וודא שנטפרי אישרו את הכתובת."
    
    response = query_engine.query(message)
    return str(response)

# עיצוב הממשק בקו נקי
with gr.Blocks(theme=gr.themes.Soft(), css="footer {visibility: hidden}") as demo:
    gr.Markdown("""
    # 🤖 Coding Agent RAG
    ### עוזר אישי לניהול ותשאול תיעוד הפרויקט
    """)
    
    gr.ChatInterface(
        fn=predict,
        textbox=gr.Textbox(
            placeholder="שאל אותי על התיעוד של Cursor או Claude Code...", 
            container=False, 
            scale=7
        ),
        submit_btn="שלח",
        retry_btn="נסה שוב 🔄",
        undo_btn="בטל ↩️",
        clear_btn="נקה צ'אט 🗑️",
    )

if __name__ == "__main__":
    demo.launch()