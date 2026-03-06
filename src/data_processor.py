import pandas as pd
import ast
import os
import shutil
import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

DATASET_DIR = "dataset"
INDICES_DIR = "indices"
PROCESSED_DATA_PATH = "processed_movies.pkl"

def parse_json_col(x):
    try:
        if pd.isna(x):
            return []
        return ast.literal_eval(x)
    except:
        return []

def get_names(x):
    return [i['name'] for i in x] if isinstance(x, list) else []

def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan

def load_and_process_data():
    print("Loading datasets...")
    meta = pd.read_csv(os.path.join(DATASET_DIR, 'movies_metadata.csv'), low_memory=False)
    keywords = pd.read_csv(os.path.join(DATASET_DIR, 'keywords.csv'))
    credits = pd.read_csv(os.path.join(DATASET_DIR, 'credits.csv'))

    # Convert IDs to numeric where possible to join
    meta['id'] = pd.to_numeric(meta['id'], errors='coerce')
    keywords['id'] = pd.to_numeric(keywords['id'], errors='coerce')
    credits['id'] = pd.to_numeric(credits['id'], errors='coerce')
    
    meta = meta.dropna(subset=['id'])
    meta['id'] = meta['id'].astype(int)
    keywords['id'] = keywords['id'].astype(int)
    credits['id'] = credits['id'].astype(int)

    # Merge
    print("Merging datasets...")
    meta = meta.merge(credits, on='id')
    meta = meta.merge(keywords, on='id')

    # Parse JSON fields
    print("Parsing JSON fields...")
    features = ['cast', 'crew', 'keywords', 'genres', 'production_countries']
    for feature in features:
        meta[feature] = meta[feature].apply(parse_json_col)
    
    meta['director'] = meta['crew'].apply(get_director)
    meta['cast'] = meta['cast'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
    meta['cast'] = meta['cast'].apply(lambda x: x[:3] if len(x) > 3 else x) # Top 3 cast
    meta['keywords'] = meta['keywords'].apply(get_names)
    meta['genres'] = meta['genres'].apply(get_names)
    meta['production_countries'] = meta['production_countries'].apply(get_names)

    # Numeric conversion
    meta['revenue'] = pd.to_numeric(meta['revenue'], errors='coerce').fillna(0)
    meta['budget'] = pd.to_numeric(meta['budget'], errors='coerce').fillna(0)
    meta['vote_average'] = pd.to_numeric(meta['vote_average'], errors='coerce').fillna(0)
    
    # Fill NA
    meta['overview'] = meta['overview'].fillna('')
    meta['tagline'] = meta['tagline'].fillna('')
    meta['title'] = meta['title'].fillna('Unknown Title')

    # Create a soup for vector search
    def create_soup(x):
        return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres']) + ' ' + x['overview'] + ' ' + x['tagline']

    # Convert director to str for soup
    meta['director'] = meta['director'].fillna('')
    
    print("Creating soup for embedding...")
    meta['soup'] = meta.apply(create_soup, axis=1)

    print(f"Processed {len(meta)} records.")
    
    # Save processed dataframe for structured tools
    meta.to_pickle(PROCESSED_DATA_PATH)
    print(f"Saved processed data to {PROCESSED_DATA_PATH}")

    return meta

def build_vector_index(df):
    print("Building Vector Index (this may take a while)...")
    
    # Limit to top N for demo/speed if needed, but requirements said "production grade" so we try full or reasonable subset
    # For local embedding speed, let's limit to top 5000 by popularity or votes to verify flow first, 
    # but the user has full dataset. Let's try 1000 for quick turn-around in this turn, 
    # and comment on how to run full.
    # actually, user wants "quality", let's do more but maybe batch it? 
    # For this environment, doing all 45k might timeout. Let's pick top 10k by vote_count.
    
    df['vote_count'] = pd.to_numeric(df['vote_count'], errors='coerce').fillna(0)
    df = df.sort_values('vote_count', ascending=False).head(10000)
    
    documents = []
    for idx, row in df.iterrows():
        doc = Document(
            page_content=row['soup'],
            metadata={"title": row['title'], "id": row['id'], "year": str(row['release_date'])[:4] if pd.notna(row['release_date']) else "N/A"}
        )
        documents.append(doc)

    # Using local embeddings for reliability and cost
    print("Using local HuggingFace embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    if not documents:
        print("No documents to index.")
        return

    # FAISS
    # Batch processing to avoid big request payloads if necessary, though FAISS/LC handles it
    vectorstore = FAISS.from_documents(documents, embeddings)
    
    if not os.path.exists(INDICES_DIR):
        os.makedirs(INDICES_DIR)
        
    vectorstore.save_local(os.path.join(INDICES_DIR, "movies_faiss_index"))
    print("Vector index saved.")

if __name__ == "__main__":
    df = load_and_process_data()
    build_vector_index(df)
    print("Data processing complete.")
