from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# Sample documents
docs = [
    Document(
        page_content="Farmers in India are facing severe drought conditions.",
        metadata={"source": "news_article_1"}
    ),
    Document(
        page_content="The Indian Premier League is one of the most popular cricket tournaments.",
        metadata={"source": "news_article_2"}
    ),
    Document(
        page_content="Bollywood is the Hindi-language film industry of India.",
        metadata={"source": "news_article_3"}
    )
]


# 1. Create embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 2. Create vector store and store embeddings
vector_store = Chroma.from_documents(
    embedding=embeddings,
    collection_name="rag_collection",
    persist_directory="./chroma_db"
)

vector_store.add_documents(docs)
print("Documents embedded and stored successfully!")

vector_store.get(include=["metadatas", "documents", "embeddings"])

# 3. Similarity search
query = "What is happening to Indian farmers?"

results = vector_store.similarity_search(query, k=2)

# 4. Display results
for result in results:
    print("-----")
    print(result.page_content)