
"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"
# Configuration
GNEWS_API_KEY = " "
NUM_RESULTS = 10
OUTPUT_JSON_FILE = "news_knowledge.json"

import os
import requests
from newspaper import Article
import json
import dspy
from dspy import Module
from dspy import Signature, ChainOfThought
import time
import streamlit as st
from sentence_transformers import SentenceTransformer
import pprint

from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType

spark = SparkSession.builder \
    .appName("NewsBotSpark") \
    .master("local[*]") \
    .getOrCreate()


model = SentenceTransformer('all-MiniLM-L6-v2')

def embedder_fn(texts):
    return model.encode(texts, show_progress_bar=True)

# Initialize DSPy
@st.cache_resource
def initialize_dspy():
    os.environ['GROQ_API_KEY'] = " "
    lm = dspy.LM('groq/gemma2-9b-it', api_key=os.environ['GROQ_API_KEY'])
    dspy.configure(lm=lm)
    return lm


class NewsRetriever:
    def __init__(self, api_key):
        self.api_key = api_key

    def get_urls(self, query):
        url = f"https://gnews.io/api/v4/search?q={query}&lang=en&max={NUM_RESULTS}&apikey={self.api_key}"
        response = requests.get(url)
        if response.status_code != 200:
            st.error(f"API Error: {response.status_code}")
            return []
        data = response.json()
        return [article['url'] for article in data.get('articles', [])]

def extract_text_from_url(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        return f"Error retrieving article: {e}"

class SummarizeSignature(Signature):
    """Summarize the article content into 3-4 sentences"""
    article: str = dspy.InputField()
    summary: str = dspy.OutputField()

class SentimentSignature(Signature):
    """Analyze the overall sentiment of the article. Return only one word: positive, negative, or neutral"""
    article: str = dspy.InputField()
    sentiment: str = dspy.OutputField(desc="One word only: positive, negative, or neutral") 

class QAWithContextSignature(Signature):
    """Answer a question using the given context"""
    context: str = dspy.InputField()
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


def process_articles(query):
    """Process articles: Spark for parallel download/parse, DSPy in main process for summarization/sentiment."""
    lm = initialize_dspy()  # make sure LM is loaded in driver
    # Initialize modules in driver
    summarizer = ChainOfThought(SummarizeSignature)
    sentiment_analyzer = ChainOfThought(SentimentSignature)

    news_retriever = NewsRetriever(GNEWS_API_KEY)
    article_urls = news_retriever.get_urls(query)

    if not article_urls:
        st.error("No articles found. Please try a different query.")
        return []

    # --- Spark UDF only for extracting article text (I/O + parsing) ---
    def _extract_text_local(url):
        try:
            art = Article(url)
            art.download()
            art.parse()
            return art.text
        except Exception:
            return None

    extract_udf = udf(_extract_text_local, StringType())

    # Build Spark DataFrame containing urls
    df = spark.createDataFrame([(u,) for u in article_urls], ["url"])
    df = df.withColumn("article", extract_udf(df.url))

    # Force the download/parse step and get results back to driver
    # Use collect() but guard it with count() first to see progress / trigger job
    try:
        total = df.count()  # triggers the Spark job that runs extract_udf in workers
        st.write(f"Downloaded/parsed {total} article slots (some may be None).")
        rows = df.select("url", "article").collect()
    except Exception as e:
        st.error(f"Spark extraction job failed: {e}")
        return []

    # --- Now run DSPy summarizer + sentiment in the driver process ---
    kb_list = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, row in enumerate(rows, start=1):
        url = row["url"]
        article_text = row["article"]
        status_text.text(f"Processing article {i}/{len(rows)}: {url}")
        progress_bar.progress(i / max(1, len(rows)))

        if not article_text:
            st.warning(f"Skipped (no text) — {url}")
            continue

        # Run summarizer & sentiment in main process (driver) — safe because LM is initialized here
        try:
            summary = summarizer(article=article_text).summary.strip()
        except Exception as e:
            summary = ""
            st.warning(f"Summarizer failed for article {i}: {e}")

        try:
            sentiment = sentiment_analyzer(article=article_text).sentiment.strip()
        except Exception as e:
            sentiment = "unknown"
            st.warning(f"Sentiment analyzer failed for article {i}: {e}")

        kb_list.append({
            "url": url,
            "article": article_text,
            "summary": summary,
            "sentiment": sentiment
        })

    progress_bar.empty()
    status_text.empty()

    # Save JSON
    try:
        with open(OUTPUT_JSON_FILE, "w", encoding="utf-8") as f:
            json.dump(kb_list, f, indent=2)
    except Exception as e:
        st.warning(f"Failed to write knowledge file: {e}")

    return kb_list



def answer_question_with_context(context, question):
    """Answer question using context"""
    lm = initialize_dspy()
    qa_module = ChainOfThought(QAWithContextSignature)
    return qa_module(context=context, question=question).answer.strip()


def main():
    st.set_page_config(page_title="NewsBot", page_icon="📰", layout="wide")
    
    st.title("📰 NewsBot - AI-Powered News Analysis")

    
    # Sidebar for settings
    with st.sidebar:
        
        st.markdown("---")
        st.header("How to use:")
        st.markdown("""
        1. Enter a news topic
        2. Click 'Fetch & Analyze News'
        3. Wait for processing
        4. Ask questions about the articles
        """)
    
    # Initialize session state
    if 'knowledge_base' not in st.session_state:
        st.session_state.knowledge_base = []
    if 'retrieval_system' not in st.session_state:
        st.session_state.retrieval_system = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        query = st.text_input(" ", 
                             placeholder="Enter a topic to search for news articles:")
    
    with col2:
        st.write("")  # Spacing
        fetch_button = st.button("🔍 Fetch & Analyze News", type="primary")
    
    # Fetch and process articles
    if fetch_button and query:
        with st.spinner("Fetching and analyzing news articles..."):
            knowledge_base = process_articles(query)
            
            if knowledge_base:
                st.session_state.knowledge_base = knowledge_base
                # embedder = dspy.Embedder('openai/text-embedding-3-small', dimensions=512)
                # st.session_state.embedder = embedder
                # st.session_state.retrieval_system = dspy.retrievers.Embeddings(embedder=embedder, corpus=knowledge_base, k=3)
                pprint.pprint([doc['article'] for doc in knowledge_base])
                st.session_state.retrieval_system = dspy.retrievers.Embeddings(embedder=embedder_fn, corpus=[doc['article'] for doc in knowledge_base], k=3)
                st.session_state.chat_history = []
                st.success(f"Successfully processed {len(knowledge_base)} articles!")
            else:
                st.error("No articles were successfully processed.")
    
    # Display knowledge base summary
    if st.session_state.knowledge_base:
        st.markdown("---")
        st.header("📊 Article Summary")
        
        col1, col2, col3 ,col4 = st.columns(4)
        with col1:
            st.metric("Total Articles", len(st.session_state.knowledge_base))
        with col2:
            sentiments = [doc['sentiment'] for doc in st.session_state.knowledge_base]
            positive_count = sum(1 for s in sentiments if 'positive' in s.lower())
            st.metric("Positive Sentiment", positive_count)
        with col3:
            negative_count = sum(1 for s in sentiments if 'negative' in s.lower())
            st.metric("Negative Sentiment", negative_count)
        with col4:
            neutral_count = sum(1 for s in sentiments if 'neutral' in s.lower())
            st.metric("Neutral Sentiment", neutral_count)

        
        # Show articles in expandable sections
        with st.expander("📄 View All Articles"):
            for i, doc in enumerate(st.session_state.knowledge_base, 1):
                st.markdown(f"**Article {i}:** {doc['url']}")
                st.markdown(f"**Summary:** {doc['summary']}")
                st.markdown(f"**Sentiment:** {doc['sentiment']}")
                st.markdown("---")
    
    # Chat interface
    if st.session_state.knowledge_base and st.session_state.retrieval_system:
        st.markdown("---")
        st.header("💬 Ask Questions")
        
        # Display chat history
        if st.session_state.chat_history:
            st.subheader("Chat History")
            for i, (question, answer) in enumerate(st.session_state.chat_history):
                with st.container():
                    st.markdown(f"**Q{i+1}:** {question}")
                    st.markdown(f"**A{i+1}:** {answer}")
                    st.markdown("---")
        
        # Question input
        # question = st.text_input("Ask Questions About the Articles", 
        #                        placeholder="e.g., What are the main points discussed?")
        question = st.text_input(" ", placeholder="Ask a question about the articles:")

        
        col1, col2 = st.columns([1, 5])
        with col1:
            ask_button = st.button("❓ Ask Question", type="primary")
        with col2:
            if st.button("🗑️ Clear Chat History"):
                st.session_state.chat_history = []
                st.rerun()
        
        if ask_button and question:
            with st.spinner("Generating answer..."):
                try:
                    # Retrieve relevant documents
                    relevant_docs = st.session_state.retrieval_system(question).passages
                    context = " ".join(relevant_docs)
                    
                    # Generate answer
                    answer = answer_question_with_context(context, question)
                    
                    # Add to chat history
                    st.session_state.chat_history.append((question, answer))
                    
                    # Display the new answer
                    st.success("Answer generated!")
                    st.markdown(f"**Question:** {question}")
                    st.markdown(f"**Answer:** {answer}")
                    
                except Exception as e:
                    st.error(f"Error generating answer: {e}")
    
    # elif not st.session_state.knowledge_base:
    #     st.info("👆 Enter a topic and click 'Fetch & Analyze News' to get started!")

if __name__ == "__main__":
    main()


