import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import SpiderLoader
from constants import *

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY2")
pinecone_api_key = os.getenv("PINECONE_API_KEY")


# list of URLs to load

urls = [
    "https://blog.sendmea.io/viral-marketing-strategies-boost-your-brands-buzz/",
    "https://blog.sendmea.io/harness-the-power-of-influencer-testimonials/",
    "https://blog.sendmea.io/social-media-endorsements-boost-your-brand/",
    "https://blog.sendmea.io/unlocking-trust-social-proof-in-marketing-strategy/",
    "https://blog.sendmea.io/unveil-key-consumer-insights-for-business-growth/",
    "https://blog.sendmea.io/effective-feedback-collection-strategies-tips/",
    "https://blog.sendmea.io/unlock-insights-top-user-ratings-revealed/",
    "https://blog.sendmea.io/boosting-customer-satisfaction/",
    "https://blog.sendmea.io/consumer-feedback-unlock-valuable-insights/",
    "https://blog.sendmea.io/gauging-social-proof-in-video-campaigns-effect/",
    "https://blog.sendmea.io/leverage-social-proof-in-video-marketing-succes/",
    "https://blog.sendmea.io/boost-your-brand-with-successful-video-marketing-campaigns/",
    "https://blog.sendmea.io/leveraging-social-proof-in-video-marketing-strategy/",
    "https://blog.sendmea.io/understanding-legalities-of-user-generated-content/",
    "https://blog.sendmea.io/video-testimonials-impact-on-reputation-management/",
    "https://blog.sendmea.io/boost-engagement-encourage-user-generated-content/"
]

# initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)

# initialize embeddings
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

# initialize Pinecone vector store
vector_store = PineconeVectorStore(index_name=PINECONE_INDEX, embedding=embeddings, pinecone_api_key=pinecone_api_key)

# loop each url; load, split, process the documents
for url in urls:
    loader = WebBaseLoader(url)
    # loader = SpiderLoader(url)

    docs = loader.load()

    documents = text_splitter.split_documents(docs)
    print(f"Loaded {len(documents)} documents to Pinecone from {url}")

    vector_store.add_documents(documents)

print('documents added to Pinecone')