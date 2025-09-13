# NewsBot-AI-Powered-News-Analysis
This project implements an AI-powered news analysis system with two versions: one using PySpark for parallel article processing and one using plain Python for sequential execution. 

There are **two implementations**:  
- **With PySpark** → Uses Spark for **parallel downloading and parsing** of news articles (faster for large batches).  
- **Without PySpark** → Runs sequentially in Python (simpler, fewer dependencies).  

---

##  How It Works  

1. **Fetch Articles**  
   - Uses the [GNews API](https://gnews.io/) to fetch top URLs for a given query.  

2. **Extract Content**  
   - Downloads and parses articles using [newspaper3k](https://pypi.org/project/newspaper3k/).  
   - In the PySpark version, this step runs in parallel across Spark workers.  

3. **Summarization & Sentiment Analysis**  
   - Summarize articles into 3–4 sentences.  
   - Detect overall sentiment (**positive / negative / neutral**).  
   - Powered by [DSPy](https://github.com/stanfordnlp/dspy) and Groq’s `gemma2-9b-it` model.  

4. **Semantic Search & Q&A**  
   - Articles are embedded with [Sentence Transformers](https://www.sbert.net/).  
   - A retriever finds the most relevant articles.  
   - DSPy generates answers using retrieved context.  

5. **Interactive UI**  
   - Built with [Streamlit](https://streamlit.io/).  
   - Lets you fetch, explore summaries, view sentiment stats, and ask questions about the news.  

---

##  Features  

-  Search any news topic  
-  Extract and summarize full articles  
-  Sentiment classification  
-  Semantic retrieval + Q&A  
-  Dashboard showing sentiment stats and summaries  
-  Parallel processing with PySpark (optional)  

