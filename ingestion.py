import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader

# Text Splitters allow us to split the text into chunks to be consumed by the model
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()
if __name__ == "__main__":
    print("Ingesting data...")

    loader = TextLoader(
        "/Users/samuelsilva/Projects/Samuel/langchain-course/intro-to-vector-dbs/mediumblog1.txt"
    )
    document = loader.load()

    print("Splitting documents...")
    # Chunk size need to be big enough to have meaningful information but not too big to cause tokenization issues with the model
    # Chunk overlap is the amount of overlap between chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)

    print(f"Splitted into {len(texts)} chunks")

    print("Embedding and vectorizing...")
    embeddings = OpenAIEmbeddings()

    PineconeVectorStore.from_documents(
        texts,
        embeddings,
        index_name=os.getenv("INDEX_NAME"),
    )

    print("Done!")
