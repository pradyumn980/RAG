from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

text = """Farmers in India are facing a severe drought this year, which has led to a significant decrease in crop yields. The government has announced relief measures to support affected farmers, including financial aid and subsidies for water conservation techniques. Experts suggest that adopting sustainable farming practices and investing in irrigation infrastructure can help mitigate the impact of future droughts.

Cricket is a game quite popular in India, and the Indian Premier League (IPL) is one of the most-watched cricket tournaments in the world. The IPL features top players from around the globe and has a massive fan following. The tournament not only provides entertainment but also contributes significantly to the economy through sponsorships, broadcasting rights, and tourism.

The Indian film industry, commonly known as Bollywood, produces a large number of films each year. It is renowned for its vibrant music, dance sequences, and colorful storytelling. Bollywood films have a global audience and have influenced fashion, culture, and entertainment worldwide. The industry continues to evolve with new technologies and storytelling techniques, attracting talent from various backgrounds."""

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

text_splitter = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=80
)

docs = text_splitter.create_documents([text])

print(len(docs))

for doc in docs:
    print("-----")
    print(doc.page_content)