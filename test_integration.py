import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.nlp.sentiment_analyzer import analyze_sentiment
from services.nlp.summarizer import ExtractiveSummarizer
from services.rag.embeddings import EmbeddingEngine
from services.rag.vector_store import VectorStore
from services.llm.chat_engine import ChatEngine
from services.visualization.flowchart_generator import FlowchartGenerator
from services.audio.tts_engine import TTSEngine

def main():
    print("----- Phase 8: System Integration Test -----")
    
    print("\n1. Testing NLP Sentiment...")
    s = analyze_sentiment("This software architecture is incredibly fast and brilliantly constructed!")
    print(f"Sentiment: {s['label']} (polarity: {s['polarity_score']})")
    
    print("\n2. Testing NLP Summarization...")
    text = (
        "CloudInsight provides a deeply integrated workflow for enterprise knowledge."
        " It handles dense, unstructured data like legal PDFs. "
        " Furthermore it can digest CSV configurations. "
        " Security is maintained locally with offline-first fallbacks."
    )
    summ = ExtractiveSummarizer(num_sentences=1).summarize(text)
    print(f"Summary: {summ}")

    print("\n3. Testing RAG Core...")
    corpus = [
        "CloudInsight uses a modular ingestion architecture.",
        "The RAG store stores embeddings for retrieval.",
        "Security is maintained locally completely offline."
    ]
    embed_engine = EmbeddingEngine()
    embed_engine.fit(corpus)
    vectors = embed_engine.embed(corpus)
    
    store = VectorStore()
    store.add_documents(vectors, [{"text": c} for c in corpus])
    
    q_vec = embed_engine.embed(["How is security handled?"])[0]
    results = store.search(q_vec, top_k=1)
    print(f"Top Semantic Match: {results[0]['text']} (Score: {results[0]['similarity_score']:.2f})")
    
    print("\n4. Testing LLM Chat Engine Router...")
    chat = ChatEngine()
    response = chat.generate_response("How is security handled?", results)
    print("Chat Engine Output:", response)
    
    print("\n5. Testing Flowchart Builder...")
    flow = FlowchartGenerator().generate_hierarchy(["Ingestion", "Embedding", "Retrieval"])
    print(flow.replace('\n', '  '))
    
    print("\n6. Testing TTS Engine Simulation...")
    # Just checking it doesn't crash on invocation
    result_audio = TTSEngine.synthesize("Integration successful.")
    print(f"Audio payload ready at: {result_audio}")
    
    print("\nIntegration Test Completed Successfully.")

if __name__ == "__main__":
    main()
