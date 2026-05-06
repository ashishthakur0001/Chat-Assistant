import chainlit as cl
from utils.pdf_loader import load_pdf
from utils.vector_store import create_vector_store
from utils.rag_chain import get_rag_chain

vector_db = None
chain = None

@cl.on_chat_start
async def start():
    await cl.Message(
        content="Upload a PDF file to start chatting with your documents."
    ).send()

@cl.on_message
async def main(message: cl.Message):
    global vector_db, chain

    if message.elements:
        for element in message.elements:
            if element.mime == "application/pdf":
                pdf_path = element.path

                docs = load_pdf(pdf_path)
                vector_db = create_vector_store(docs)
                chain = get_rag_chain(vector_db)

                await cl.Message(
                    content="PDF uploaded and processed successfully."
                ).send()
                return

    if chain is None:
        await cl.Message(
            content="Please upload a PDF first."
        ).send()
        return

    response = chain.invoke({"query": message.content})

    await cl.Message(content=response["result"]).send()
