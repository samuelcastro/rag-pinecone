# Simple RAG implementation using LangChain + Pinecone and OpenAI
import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore

from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

load_dotenv()

if __name__ == "__main__":
    print("Retrieving data...")

    embeddings = OpenAIEmbeddings()
    llm = ChatOpenAI()

    query = "What is Pinecone in machine learning?"
    chain = PromptTemplate.from_template(template=query) | llm

    # result = chain.invoke(input={})
    # print("Answer without context:")
    # print(result.content)

    vectorstore = PineconeVectorStore(
        index_name=os.getenv("INDEX_NAME"),
        embedding=embeddings,
    )

    # Retrieval QA Chat Prompt: This prompt is used to answer questions based on the context.
    # link: https://smith.langchain.com/hub/langchain-ai/retrieval-qa-chat
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    # create_stuff_documents_chain: This chain takes a list of documents and formats them all into a prompt, then
    # passes that prompt to the LLM. It passes ALL documents, so you should make sure
    # it fits within the context window of the LLM.
    combine_docs_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=retrieval_qa_chat_prompt,
    )

    retrieval_chain = create_retrieval_chain(
        retriever=vectorstore.as_retriever(),
        combine_docs_chain=combine_docs_chain,
    )

    result = retrieval_chain.invoke(input={"input": query})
    print("Answer with context:")
    print(result)

    print("Done!")
