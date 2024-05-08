import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from constants import *

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY2")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

prompt = 'Give me 3 ways to use social proof in marketing?'

embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

document_vectorstore = PineconeVectorStore(index_name=PINECONE_INDEX, embedding=embeddings)
retriever = document_vectorstore.as_retriever()
context= retriever.get_relevant_documents(prompt)

for doc in context:
    print(f"Source: {doc.metadata['source']}\nContent: {doc.page_content}\n\n")
print('--------------------------')

template = PromptTemplate(template="{query} Context: {context}", input_variables=["query", "context"])
prompt_with_context = template.invoke({"query": prompt, "context": context})

llm = ChatOpenAI(temperature=0.5)
results = llm.invoke(prompt_with_context)

print(results.content)