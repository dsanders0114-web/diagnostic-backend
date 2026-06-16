from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import base64
import tempfile
from dotenv import load_dotenv
from openai import OpenAI
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# 1. Initialize Data Connections
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

embeddings = OpenAIEmbeddings(openai_api_key=api_key)
vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# 2. Initialize the Server
app = FastAPI(title="Diagnostic Copilot Engine")

# 3. Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 4. Define the Multimodal Data Structure
class ChatRequest(BaseModel):
    message: str | None = None
    image: str | None = None
    audio: str | None = None

# 5. Build the Diagnostic Endpoint
@app.post("/api/diagnose")
async def process_diagnostic(request: ChatRequest):
    try:
        user_text = request.message or ""

        # Process Audio Data if Present
        if request.audio:
            audio_bytes = base64.b64decode(request.audio.split(",")[1] if "," in request.audio else request.audio)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
                tmp_audio.write(audio_bytes)
                tmp_audio_path = tmp_audio.name
            
            with open(tmp_audio_path, "rb") as audio_file:
                transcript = client.audio.transcriptions.create(
                    model="whisper-1", 
                    file=audio_file
                )
            
            if user_text:
                user_text = f"{user_text}\nVoice Note: {transcript.text}"
            else:
                user_text = transcript.text
            
            os.remove(tmp_audio_path)

        if not user_text and not request.image:
            raise HTTPException(status_code=400, detail="No input provided.")

        # Search ChromaDB Memory using the final text query
        search_query = user_text if user_text else "Visual diagnostic request"
        docs = vector_db.similarity_search(search_query, k=3)
        db_context = "\n\n".join([doc.page_content for doc in docs])
        
        # Construct the Core Logic Prompt
        system_prompt = (
            "You are a Tier-3 industrial diagnostic AI for client info. "
            "You must format all troubleshooting steps using bullet points. "
            "You must never use an em dash in your responses. "
            "Provide highly technical, logical, and concise engineering analysis.\n\n"
            f"Use the following equipment manual excerpts to strictly answer the user query. Do not guess outside of this context:\n\n{db_context}"
        )
        
        messages = [{"role": "system", "content": system_prompt}]
        
        # Construct the User Message
        if request.image:
            image_data = request.image
            if not image_data.startswith("data:image"):
                image_data = f"data:image/jpeg;base64,{image_data}"
                
            messages.append({
                "role": "user", 
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": image_data}}
                ]
            })
        else:
            messages.append({"role": "user", "content": user_text})
        
        # Generate Response from OpenAI
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )
        
        ai_reply = response.choices[0].message.content
        sources = list(set([doc.metadata.get('source', 'Unknown Manual') for doc in docs]))
        
        return {
            "reply": ai_reply,
            "sources": sources,
            "transcription": user_text if request.audio else None
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))