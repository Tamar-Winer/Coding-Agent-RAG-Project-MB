# 🤖 Agentic RAG Assistant - Project 3

מערכת RAG (Retrieval-Augmented Generation) מתקדמת המבוססת על ארכיטקטורת **Event-Driven Workflows**. המערכת מסוגלת לסרוק תיעוד טכני, לאנדקס אותו ולספק תשובות מדויקות תוך ביצוע ולידציה עצמית ותיקון שגיאות בזמן אמת.

## 🚀 תכונות מרכזיות
- **Agentic Workflow**: שימוש ב-LlamaIndex Workflows לניהול זרימת מידע מבוססת אירועים.
- **Self-Correction**: מנגנון תיקון עצמי המבצע חיפוש חוזר אם רמת הביטחון (Confidence Score) של המידע נמוכה.
- **Hybrid Indexing**: אינדוקס וקטורי ב-Pinecone עם העשרה של Metadata לשמירת הקשר (Tool, File, Project).
- **Semantic Search**: שימוש ב-Embeddings רב-לשוניים של Cohere לתמיכה מלאה בעברית ואנגלית.
- **Professional UI**: ממשק צ'אט מודרני ונקי המבוסס על Gradio.

## 🏗️ ארכיטקטורה
הפרויקט בנוי משלושה שלבים אבולוציוניים:
1. **שלב א' (MVP)**: הקמת ה-Pipeline הבסיסי, חיבור ל-Pinecone וביצוע חיפוש סמנטי.
2. **שלב ב' (Event-Driven)**: שכתוב המערכת למבנה של Steps ו-Events לניהול לוגיקה מורכבת.
3. **שלב ג' (Reflection)**: הוספת לולאות משוב (Feedback Loops) המאפשרות לסוכן "לחשוב" ולתקן את עצמו לפני הצגת התשובה.



## 🛠️ טכנולוגיות
- **Framework**: [LlamaIndex](https://www.llamaindex.ai/)
- **Vector DB**: [Pinecone](https://www.pinecone.io/)
- **Embedding Model**: `cohere.embed-multilingual-v3.0`
- **UI Framework**: [Gradio](https://gradio.app/)
- **Language**: Python 3.x

## 📋 התקנה והרצה

1. שכפלי את המאגר:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/coding-agent-rag.git](https://github.com/YOUR_USERNAME/coding-agent-rag.git)
   cd coding-agent-rag