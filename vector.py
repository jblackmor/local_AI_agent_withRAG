# Vector store database setup (embedded our documents & looking them up, aka 'vectorizing')
### vector search is a database hosted locally using chromaDB allowing us to
### quickly look up relevant information that we can then pass to our model,
### who can then use that data to give contextually relevant replies

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import pandas as pd
import os

df = pd.read_csv('realistic_restaurant_reviews.csv')
embeddings = OllamaEmbeddings(model='mxbai-embed-large')

db_location = './chroma_langchain_db'
add_documents = not os.path.exists(db_location)  # perform vectorization, if not not done already

# one-time process of converting csv into db (if it need be done)
if add_documents:
    documents = []
    ids = []

    for i, row in df.iterrows():
        document = Document(
            page_content = row['Title'] + " " + row['Review'], # stuff you want to look up
            metadata = {'rating': row['Rating'], 'date': row['Date']}, # addtl data to pull (but not query)
            id = str(i)
        )
        ids.append(str(i))
        documents.append(document)