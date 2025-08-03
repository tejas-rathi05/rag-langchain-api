import asyncio
from pinecone import Pinecone
from dotenv import load_dotenv
from pydantic import SecretStr
import os
import pdfplumber
from sentence_transformers import SentenceTransformer


from agent import QueueCallbackHandler, agent_executor
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import UploadFile, File
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

from dotenv import load_dotenv
import os
from pydantic import SecretStr

load_dotenv()  # ✅ Loads from .env locally

load_dotenv()

# ✅ Read keys from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is missing.")
if not SERPAPI_API_KEY:
    raise ValueError("SERPAPI_API_KEY is missing.")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is missing.")
if not PINECONE_INDEX:
    raise ValueError("PINECONE_INDEX is missing.")


# initilizing our application
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Your frontend URL
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# streaming function
async def token_generator(content: str, streamer: QueueCallbackHandler):
    task = asyncio.create_task(agent_executor.invoke(
        input=content,
        streamer=streamer,
        verbose=True  # set to True to see verbose output in console
    ))
    # initialize various components to stream
    async for token in streamer:
        try:
            if token == "<<STEP_END>>":
                # send end of step token
                yield "</step>"
            elif tool_calls := token.message.additional_kwargs.get("tool_calls"):
                if tool_name := tool_calls[0]["function"]["name"]:
                    # send start of step token followed by step name tokens
                    yield f"<step><step_name>{tool_name}</step_name>"
                if tool_args := tool_calls[0]["function"]["arguments"]:
                    # tool args are streamed directly, ensure it's properly encoded
                    yield tool_args
        except Exception as e:
            print(f"Error streaming token: {e}")
            continue
    await task

# invoke function
@app.post("/invoke")
async def invoke(content: str):
    queue: asyncio.Queue = asyncio.Queue()
    streamer = QueueCallbackHandler(queue)
    # return the streaming response
    return StreamingResponse(
        token_generator(content, streamer),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

@app.post("/upload_document")
async def upload_document(file: UploadFile = File(...)):
    content = await file.read()
    text = ""
    if file.filename.endswith(".txt"):
        text = content.decode("utf-8", errors="ignore")
    elif file.filename.endswith(".pdf"):
        with open(f"temp_{file.filename}", "wb") as f:
            f.write(content)
        with pdfplumber.open(f"temp_{file.filename}") as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        os.remove(f"temp_{file.filename}")
    else:
        return {"error": "Unsupported file type. Please upload a .txt or .pdf file."}

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.create_documents([text])

    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(os.getenv("PINECONE_INDEX"))
    model = SentenceTransformer("all-MiniLM-L6-v2")  # Hugging Face model

    for i, doc in enumerate(docs):
        embedding = model.encode(doc.page_content)
        index.upsert([(f"{file.filename}-{i}", embedding.tolist(), {"text": doc.page_content})])
    return {"status": "uploaded", "chunks": len(docs)}
