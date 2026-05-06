import os
from dotenv import load_dotenv

from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

load_dotenv()


def get_rag_chain(vector_db):
    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama3-8b-8192"
    )

    retriever = vector_db.as_retriever()

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    return qa_chain
