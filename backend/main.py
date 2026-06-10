import os
import re
import time
import json
import requests
import fitz
import google.generativeai as genai
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import Optional, List
from datetime import datetime
from sqlalchemy.orm import Session
from youtube_transcript_api import YouTubeTranscriptApi
from backend.database import get_db, Summary, Flashcard, Quiz, Chat
import whisper

from pathlib import Path

# Load environment variables from a .env file in the root folder relative to this file
backend_dir = Path(__file__).resolve().parent
project_root = backend_dir.parent
load_dotenv(dotenv_path=project_root / ".env")

# Initialize database tables
from backend.database import init_db
init_db()

# Load Whisper model once at startup (uses "small" model version)
try:
    print("[Whisper] Loading Whisper model ('small')... This may take a moment.")
    whisper_model = whisper.load_model("small")
    print("[Whisper] Whisper model loaded successfully!")
except Exception as e:
    print(f"[Whisper ERROR] Failed to load Whisper model: {e}")
    whisper_model = None


app = FastAPI(
    title="Webpage Summariser API",
    description="A backend API for generating text summaries using Gemini.",
    version="0.2.0"
)

# Enable CORS for Chrome Extension requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for extension interaction
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SummarizeRequest(BaseModel):
    text: str
    title: Optional[str] = None
    url: Optional[str] = None

class YouTubeRequest(BaseModel):
    url: str

class SummarizeResponse(BaseModel):
    summary: str
    id: int

class SummaryItem(BaseModel):
    id: int
    title: Optional[str] = None
    url: Optional[str] = None
    summary: str
    created_at: datetime

    class Config:
        from_attributes = True

class FlashcardRequest(BaseModel):
    summary_id: int

class FlashcardItem(BaseModel):
    id: int
    summary_id: int
    front: str
    back: str
    created_at: datetime

    class Config:
        from_attributes = True

class QuizRequest(BaseModel):
    summary_id: int

class QuizItem(BaseModel):
    id: int
    summary_id: int
    question: str
    options: List[str]
    answer: str
    created_at: datetime

    class Config:
        from_attributes = True

class ChatRequest(BaseModel):
    message: str
    summary_id: Optional[int] = None

class GeneralChatRequest(BaseModel):
    message: str

class ChatItem(BaseModel):
    id: int
    summary_id: int
    role: str
    message: str
    created_at: datetime

    class Config:
        from_attributes = True

class ChatResponse(BaseModel):
    message: str
    chat_id: int


# Retrieve the API key from environment variable
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    print("\n" + "="*80)
    print("WARNING: GEMINI_API_KEY environment variable is not set.")
    print("Please set it in your environment or create a '.env' file in the root folder.")
    print("="*80 + "\n")
else:
    genai.configure(api_key=api_key)

def generate_content_with_retry(model, prompt, max_retries=3, initial_delay=5):
    """Generates content using the Gemini model with exponential backoff on rate limit exceptions."""
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            return model.generate_content(prompt)
        except Exception as e:
            err_str = str(e).lower()
            is_rate_limit = "429" in err_str or "quota" in err_str or "rate_limit" in err_str or "rate limit" in err_str
            
            if is_rate_limit and attempt < max_retries - 1:
                print(f"Gemini API rate limit hit. Retrying attempt {attempt + 1} of {max_retries} in {delay}s...")
                time.sleep(delay)
                delay *= 2
            else:
                raise e

def parse_json_response(text: str):
    """Safely extracts and parses JSON array from markdown-wrapped text."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        # Remove opening ```json or ```
        cleaned = re.sub(r"^```(?:json)?\n", "", cleaned)
        # Remove closing ```
        cleaned = re.sub(r"\n```$", "", cleaned)
        cleaned = cleaned.strip()
    return json.loads(cleaned)

def call_groq_api(prompt: str, json_mode: bool = False) -> str:
    """Dispatches completions request to Groq Cloud API directly using requests."""
    groq_key = os.environ.get("GROQ_API_KEY")
    if not groq_key:
        raise ValueError("GROQ_API_KEY environment variable is not configured.")

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {groq_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2
    }
    
    if json_mode:
        data["response_format"] = {"type": "json_object"}
        
    response = requests.post(url, headers=headers, json=data, timeout=30)
    if response.status_code != 200:
        raise Exception(f"Groq API status code {response.status_code}: {response.text}")
        
    res_json = response.json()
    return res_json["choices"][0]["message"]["content"]

def generate_local_mock(fallback_type: str, text_preview: str = "") -> str:
    """Generates high-quality mock data locally for summaries, flashcards, or quizzes."""
    if fallback_type == "text":
        preview = text_preview[:200].strip() if text_preview else "the study material"
        return (
            "### Summary (Mock Fallback)\n"
            "This summary was generated locally because your AI API keys are currently offline or out of quota.\n\n"
            f"* **Primary Focus**: The studied text primarily describes: *{preview}...*\n"
            "* **Detailed Analysis**: The document highlights structural methodologies, procedural guidelines, and foundational theories.\n"
            "* **Key Takeaway**: Continuous review of study guides, practice flashcards, and quizzes is recommended for optimal retention of this topic."
        )
    elif fallback_type == "flashcards":
        return json.dumps([
            {"front": "What is the primary focus of the studied material?", "back": "Refer to the summary section to review the core focus."},
            {"front": "Key Concept 1: Define the primary topic described.", "back": "The concept represents the main structural process outlined in the text."},
            {"front": "Why is this subject important for college students?", "back": "It provides crucial context and foundational knowledge for related coursework."},
            {"front": "How can you apply this concepts in practice?", "back": "By analyzing the core components, methodologies, and rules outlined in the study guide."},
            {"front": "What is the key takeaway of this summary?", "back": "Developing a comprehensive understanding of the structural details and applications."}
        ])
    elif fallback_type == "quiz":
        return json.dumps([
            {
                "question": "What is the main focus of the studied material?",
                "options": ["A. The primary concept outlined in the text", "B. An unrelated side-topic", "C. A historical anecdote", "D. None of the above"],
                "answer": "A"
            },
            {
                "question": "Which of the following best describes the core principle?",
                "options": ["A. Option A", "B. The main concept explained in the summary", "C. Option C", "D. Option D"],
                "answer": "B"
            },
            {
                "question": "Why is this topic significant?",
                "options": ["A. It has no relevance", "B. It has no practical application", "C. It provides critical insights for students", "D. It is optional"],
                "answer": "C"
            },
            {
                "question": "What is the recommended approach to study this material?",
                "options": ["A. Skim it quickly", "B. Ignore key definitions", "C. Review summaries, flashcards, and practice quizzes", "D. Memorize without context"],
                "answer": "C"
            },
            {
                "question": "Which option represents a key takeaway?",
                "options": ["A. Option A", "B. Option B", "C. Option C", "D. Understanding the structural connections between ideas"],
                "answer": "D"
            }
        ])
    return ""

def call_groq_chat(system_prompt: str, history: List[Chat], new_message: str) -> str:
    """Dispatches chat completions to Groq API including system prompt and history."""
    groq_key = os.environ.get("GROQ_API_KEY")
    if not groq_key:
        raise ValueError("GROQ_API_KEY environment variable is not configured.")

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {groq_key}",
        "Content-Type": "application/json"
    }

    messages = [{"role": "system", "content": system_prompt}]
    for msg in history:
        messages.append({"role": msg.role, "content": msg.message})
    messages.append({"role": "user", "content": new_message})

    data = {
        "model": "llama-3.3-70b-versatile",
        "messages": messages,
        "temperature": 0.5
    }

    response = requests.post(url, headers=headers, json=data, timeout=30)
    if response.status_code != 200:
        raise Exception(f"Groq API status code {response.status_code}: {response.text}")

    res_json = response.json()
    return res_json["choices"][0]["message"]["content"]

def call_gemini_chat(system_prompt: str, history: List[Chat], new_message: str) -> str:
    """Dispatches chat completions to Gemini API including system instruction and history."""
    gemini_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_key:
        raise ValueError("GEMINI_API_KEY environment variable is not configured.")

    genai.configure(api_key=gemini_key)
    model = genai.GenerativeModel("gemini-2.5-flash", system_instruction=system_prompt)

    contents = []
    for msg in history:
        role = "user" if msg.role == "user" else "model"
        contents.append({"parts": [{"text": msg.message}], "role": role})
    contents.append({"parts": [{"text": new_message}], "role": "user"})

    response = generate_content_with_retry(model, contents)
    if response.text:
        return response.text
    raise ValueError("Empty response from Gemini API.")

def escape_markdown(text: str) -> str:
    """Escapes basic markdown characters to prevent layout break in mock responses."""
    if not text:
        return ""
    return text.replace("*", "\\*").replace("_", "\\_").replace("#", "\\#")

def generate_chat_text(system_prompt: str, history: List[Chat], new_message: str, text_preview: str = "") -> str:
    """Routes chat requests dynamically to Groq (first choice), Gemini (second choice), or Local Fallback."""
    # 1. Attempt Groq
    groq_key = os.environ.get("GROQ_API_KEY")
    if groq_key and groq_key.strip():
        try:
            print("[LLM Router] Routing chat request to Groq API...")
            return call_groq_chat(system_prompt, history, new_message)
        except Exception as e:
            print(f"[ERROR] Groq Chat API call failed: {e}")

    # 2. Attempt Gemini
    gemini_key = os.environ.get("GEMINI_API_KEY")
    if gemini_key and gemini_key.strip():
        try:
            print("[LLM Router] Routing chat request to Gemini API...")
            return call_gemini_chat(system_prompt, history, new_message)
        except Exception as e:
            print(f"[ERROR] Gemini Chat API call failed: {e}")

    # 3. Fallback to Local Mock
    print("[WARNING] Both LLM providers offline/exhausted. Falling back to local mock chat generator.")
    return (
        "### Study Assistant (Mock Fallback)\n\n"
        "I'm currently running in offline mock fallback mode because your LLM API keys are exhausted or unavailable.\n\n"
        f"You asked: *\"{escape_markdown(new_message)}\"*\n\n"
        f"Based on the provided content summary (*{escape_markdown(text_preview[:150])}...*), "
        "please make sure the backend server is connected to the internet and check your API quotas."
    )

def generate_text(prompt: str, is_json: bool = False, fallback_type: str = "text", text_preview: str = "") -> str:
    """Dispatches prompt generation to Groq (first choice), Gemini (second choice), or Local Mock Fallback."""
    # 1. Attempt Groq
    groq_key = os.environ.get("GROQ_API_KEY")
    if groq_key and groq_key.strip():
        try:
            print("[LLM Router] Routing request to Groq API...")
            return call_groq_api(prompt, json_mode=is_json)
        except Exception as e:
            print(f"[ERROR] Groq API call failed: {e}")
            
    # 2. Attempt Gemini
    gemini_key = os.environ.get("GEMINI_API_KEY")
    if gemini_key and gemini_key.strip():
        try:
            print("[LLM Router] Routing request to Gemini API...")
            genai.configure(api_key=gemini_key)
            model = genai.GenerativeModel("gemini-2.5-flash")
            response = generate_content_with_retry(model, prompt)
            if response.text:
                return response.text
        except Exception as e:
            print(f"[ERROR] Gemini API call failed: {e}")
            
    # 3. Fallback to Local Mock
    print(f"[WARNING] Both LLM providers offline/exhausted. Falling back to local mock generator for type: '{fallback_type}'")
    return generate_local_mock(fallback_type, text_preview)

@app.get("/")
async def root():
    return {
        "message": "Webpage Summariser API is running.",
        "gemini_configured": api_key is not None
    }

@app.post("/summarize", response_model=SummarizeResponse)
async def summarize(payload: SummarizeRequest, db: Session = Depends(get_db)):
    if not payload.text or not payload.text.strip():
        raise HTTPException(
            status_code=400,
            detail="No readable text provided for summarization."
        )

    # Format the prompt
    prompt = f"Summarize this content clearly for a college student: {payload.text}"
    
    try:
        response_text = generate_text(prompt, is_json=False, fallback_type="text", text_preview=payload.text)
        
        # Save to database
        db_summary = Summary(
            title=payload.title,
            url=payload.url,
            summary=response_text
        )
        db.add(db_summary)
        db.commit()
        db.refresh(db_summary)
            
        return SummarizeResponse(summary=response_text, id=db_summary.id)
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"API Error during summarization: {str(e)}"
        )

def extract_video_id(url: str) -> Optional[str]:
    # Handles watch?v=, embed/, shorts/, youtu.be/
    patterns = [
        r'(?:v=|\/shorts\/|\/embed\/|youtu\.be\/)([^#\&\?]{11})'
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

@app.post("/summarize-youtube", response_model=SummarizeResponse)
async def summarize_youtube(payload: YouTubeRequest, db: Session = Depends(get_db)):
    video_id = extract_video_id(payload.url)
    if not video_id:
        raise HTTPException(
            status_code=400,
            detail="Invalid YouTube URL. Could not parse video ID."
        )

    try:
        # Retrieve the list of all available transcripts
        transcript_list = YouTubeTranscriptApi().list(video_id)
        
        # Try to find an English transcript first
        try:
            transcript = transcript_list.find_transcript(['en'])
        except Exception:
            # Fallback: find the first available transcript and translate it to English
            try:
                first_transcript = next(iter(transcript_list))
                transcript = first_transcript.translate('en')
            except Exception as trans_err:
                raise Exception(f"No English transcript found and translation failed: {str(trans_err)}")
                
        transcript_data = transcript.fetch()
        transcript_text = " ".join([
            entry.get('text', '').strip() if isinstance(entry, dict) else (entry.text if hasattr(entry, 'text') else entry['text'])
            for entry in transcript_data
        ])
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to retrieve YouTube transcript: {str(e)}"
        )

    if not transcript_text.strip():
        raise HTTPException(
            status_code=400,
            detail="The transcript retrieved from the YouTube video was empty."
        )

    # Format the prompt
    prompt = f"Summarize this YouTube video transcript clearly for a college student: {transcript_text}"

    try:
        response_text = generate_text(prompt, is_json=False, fallback_type="text", text_preview=transcript_text)

        # Save to database
        db_summary = Summary(
            title=f"YouTube: {video_id}",
            url=payload.url,
            summary=response_text
        )
        db.add(db_summary)
        db.commit()
        db.refresh(db_summary)
            
        return SummarizeResponse(summary=response_text, id=db_summary.id)
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"API Error during YouTube summarization: {str(e)}"
        )

@app.post("/summarize-pdf", response_model=SummarizeResponse)
async def summarize_pdf(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Only PDF documents are supported."
        )

    try:
        pdf_bytes = await file.read()
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to read uploaded file: {str(e)}"
        )

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = ""
        for page in doc:
            page_text = page.get_text()
            if page_text:
                text += page_text + "\n"
        doc.close()
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to extract text from PDF document: {str(e)}"
        )

    clean_text = text.strip()
    if not clean_text:
        raise HTTPException(
            status_code=400,
            detail="The PDF document does not contain any extractable text."
        )

    # Format the prompt
    prompt = f"Summarize this PDF document clearly for a college student: {clean_text}"

    try:
        response_text = generate_text(prompt, is_json=False, fallback_type="text", text_preview=clean_text)

        # Save to database
        db_summary = Summary(
            title=f"PDF: {file.filename}",
            url=None,
            summary=response_text
        )
        db.add(db_summary)
        db.commit()
        db.refresh(db_summary)
            
        return SummarizeResponse(summary=response_text, id=db_summary.id)
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"API Error during PDF summarization: {str(e)}"
        )

@app.get("/summaries", response_model=List[SummaryItem])
async def get_summaries(db: Session = Depends(get_db)):
    try:
        summaries = db.query(Summary).order_by(Summary.created_at.desc()).all()
        return summaries
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Database query error: {str(e)}"
        )

@app.delete("/summaries/{id}")
async def delete_summary(id: int, db: Session = Depends(get_db)):
    try:
        db_summary = db.query(Summary).filter(Summary.id == id).first()
        if not db_summary:
            raise HTTPException(
                status_code=404,
                detail=f"Summary with ID {id} not found."
            )
        db.delete(db_summary)
        db.commit()
        return {"message": "Summary deleted successfully.", "id": id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Database error deleting summary: {str(e)}"
        )

@app.get("/dashboard", response_class=HTMLResponse)
async def get_dashboard():
    dashboard_path = project_root / "frontend" / "index.html"
    try:
        with open(dashboard_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read(), status_code=200)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Dashboard file not found: {str(e)}"
        )

@app.get("/summaries/{id}/flashcards", response_model=List[FlashcardItem])
async def get_flashcards(id: int, db: Session = Depends(get_db)):
    db_summary = db.query(Summary).filter(Summary.id == id).first()
    if not db_summary:
        raise HTTPException(
            status_code=404,
            detail=f"Summary with ID {id} not found."
        )
    return db.query(Flashcard).filter(Flashcard.summary_id == id).order_by(Flashcard.created_at.asc()).all()

@app.get("/summaries/{id}/quizzes", response_model=List[QuizItem])
async def get_quizzes(id: int, db: Session = Depends(get_db)):
    db_summary = db.query(Summary).filter(Summary.id == id).first()
    if not db_summary:
        raise HTTPException(
            status_code=404,
            detail=f"Summary with ID {id} not found."
        )
    return db.query(Quiz).filter(Quiz.summary_id == id).order_by(Quiz.created_at.asc()).all()

@app.post("/generate-flashcards", response_model=List[FlashcardItem])
async def generate_flashcards(payload: FlashcardRequest, db: Session = Depends(get_db)):
    db_summary = db.query(Summary).filter(Summary.id == payload.summary_id).first()
    if not db_summary:
        raise HTTPException(
            status_code=404,
            detail=f"Summary with ID {payload.summary_id} not found."
        )

    # Return existing if they already exist to avoid duplicate generation
    existing_cards = db.query(Flashcard).filter(Flashcard.summary_id == payload.summary_id).order_by(Flashcard.created_at.asc()).all()
    if existing_cards:
        print(f"[API] Returning {len(existing_cards)} existing flashcards for summary_id={payload.summary_id}")
        return existing_cards

    prompt = (
        "Generate 5 flashcards from this content. Return ONLY a JSON array like this:\n"
        '[{"front": "question here", "back": "answer here"}]\n'
        "No extra text, no markdown, just the raw JSON array.\n\n"
        f"Content:\n{db_summary.summary}"
    )

    try:
        response_text = generate_text(prompt, is_json=True, fallback_type="flashcards")
        
        try:
            flashcards_data = parse_json_response(response_text)
            if isinstance(flashcards_data, dict):
                for val in flashcards_data.values():
                    if isinstance(val, list):
                        flashcards_data = val
                        break
            if not isinstance(flashcards_data, list):
                raise ValueError("Response is not a JSON list array")
        except Exception as parse_err:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to parse generated flashcards JSON: {str(parse_err)}. Raw response: {response_text}"
            )
            
        db_flashcards = []
        for card in flashcards_data:
            front = card.get("front", "").strip()
            back = card.get("back", "").strip()
            if not front or not back:
                continue
            
            db_card = Flashcard(
                summary_id=db_summary.id,
                front=front,
                back=back
            )
            db.add(db_card)
            db_flashcards.append(db_card)
            
        db.commit()
        
        for db_card in db_flashcards:
            db.refresh(db_card)
            
        return db_flashcards

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating flashcards: {str(e)}"
        )

@app.post("/generate-quiz", response_model=List[QuizItem])
async def generate_quiz(payload: QuizRequest, db: Session = Depends(get_db)):
    db_summary = db.query(Summary).filter(Summary.id == payload.summary_id).first()
    if not db_summary:
        raise HTTPException(
            status_code=404,
            detail=f"Summary with ID {payload.summary_id} not found."
        )

    # Return existing if they already exist to avoid duplicate generation
    existing_quiz = db.query(Quiz).filter(Quiz.summary_id == payload.summary_id).order_by(Quiz.created_at.asc()).all()
    if existing_quiz:
        print(f"[API] Returning {len(existing_quiz)} existing quiz items for summary_id={payload.summary_id}")
        return existing_quiz

    prompt = (
        "Generate 5 multiple choice questions from this content. Return ONLY a JSON array like this:\n"
        '[{"question": "question here", "options": ["A", "B", "C", "D"], "answer": "A"}]\n'
        "No extra text, no markdown, just the raw JSON array.\n\n"
        f"Content:\n{db_summary.summary}"
    )

    try:
        response_text = generate_text(prompt, is_json=True, fallback_type="quiz")
        
        try:
            quiz_data = parse_json_response(response_text)
            if isinstance(quiz_data, dict):
                for val in quiz_data.values():
                    if isinstance(val, list):
                        quiz_data = val
                        break
            if not isinstance(quiz_data, list):
                raise ValueError("Response is not a JSON list array")
        except Exception as parse_err:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to parse generated quiz JSON: {str(parse_err)}. Raw response: {response_text}"
            )
            
        db_quizzes = []
        for item in quiz_data:
            question = item.get("question", "").strip()
            options = item.get("options", [])
            answer = item.get("answer", "").strip()
            
            if not question or not options or not answer:
                continue
                
            db_quiz = Quiz(
                summary_id=db_summary.id,
                question=question,
                options=options,
                answer=answer
            )
            db.add(db_quiz)
            db_quizzes.append(db_quiz)
            
        db.commit()
        
        for db_quiz in db_quizzes:
            db.refresh(db_quiz)
            
        return db_quizzes

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating quiz: {str(e)}"
        )

@app.get("/chat/general", response_model=List[ChatItem])
async def get_general_chat_history(db: Session = Depends(get_db)):
    return db.query(Chat).filter(Chat.summary_id == 0).order_by(Chat.created_at.asc()).all()

@app.get("/chat/{summary_id}", response_model=List[ChatItem])
async def get_chat_history(summary_id: int, db: Session = Depends(get_db)):
    db_summary = db.query(Summary).filter(Summary.id == summary_id).first()
    if not db_summary:
        raise HTTPException(
            status_code=404,
            detail=f"Summary with ID {summary_id} not found."
        )
    return db.query(Chat).filter(Chat.summary_id == summary_id).order_by(Chat.created_at.asc()).all()

@app.post("/chat", response_model=ChatResponse)
async def chat_with_assistant(payload: ChatRequest, db: Session = Depends(get_db)):
    summary_id = payload.summary_id if payload.summary_id is not None else 0

    if summary_id > 0:
        db_summary = db.query(Summary).filter(Summary.id == summary_id).first()
        if not db_summary:
            raise HTTPException(
                status_code=404,
                detail=f"Summary with ID {summary_id} not found."
            )

        # 1. Fetch the last 10 messages as history context before adding the new message
        history = db.query(Chat).filter(Chat.summary_id == summary_id).order_by(Chat.created_at.desc()).limit(10).all()
        history.reverse()

        # 2. Save user message to database
        user_chat = Chat(
            summary_id=summary_id,
            role="user",
            message=payload.message
        )
        db.add(user_chat)
        db.commit()
        db.refresh(user_chat)

        # 3. Generate AI response
        system_prompt = (
            "You are a helpful study assistant. You have been given this content to help "
            "the student understand it better. Answer their questions based on this content:\n"
            f"{db_summary.summary}\n\n"
            "IMPORTANT CONSTRAINT: If the user's message is asking about a topic, concept, "
            "or fact that is NOT mentioned in, NOT directly related to, and CANNOT be reasonably "
            "inferred from the summary content provided above, you MUST respond EXACTLY with this phrase "
            "and absolutely nothing else:\n"
            "\"I can only answer questions about the content you summarized.\"\n"
            "Do not explain, do not add any other text, warnings, or polite greetings. "
            "(Note: The user may ask follow-up questions referencing previous parts of the chat history. "
            "As long as the discussion is connected to the summary, you should answer it. However, if they "
            "introduce an entirely new unrelated topic, trigger the refusal.)"
        )

        try:
            ai_message = generate_chat_text(system_prompt, history, payload.message, text_preview=db_summary.summary)

            # 4. Save AI response to database
            assistant_chat = Chat(
                summary_id=summary_id,
                role="assistant",
                message=ai_message
            )
            db.add(assistant_chat)
            db.commit()
            db.refresh(assistant_chat)

            return ChatResponse(message=ai_message, chat_id=assistant_chat.id)

        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error generating chat response: {str(e)}"
            )
    else:
        # General chat mode (no summary context, mapped to summary_id = 0)
        history = db.query(Chat).filter(Chat.summary_id == 0).order_by(Chat.created_at.desc()).limit(10).all()
        history.reverse()

        # Save user message to database
        user_chat = Chat(
            summary_id=0,
            role="user",
            message=payload.message
        )
        db.add(user_chat)
        db.commit()
        db.refresh(user_chat)

        # System prompt for general chat
        system_prompt = (
            "You are a helpful study assistant. Answer the student's questions and help them study."
        )

        try:
            ai_message = generate_chat_text(system_prompt, history, payload.message, text_preview="general study help")

            # Save AI response to database
            assistant_chat = Chat(
                summary_id=0,
                role="assistant",
                message=ai_message
            )
            db.add(assistant_chat)
            db.commit()
            db.refresh(assistant_chat)

            return ChatResponse(message=ai_message, chat_id=assistant_chat.id)

        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error generating chat response: {str(e)}"
            )

@app.post("/chat/general", response_model=ChatResponse)
async def general_chat(payload: GeneralChatRequest, db: Session = Depends(get_db)):
    chat_payload = ChatRequest(message=payload.message, summary_id=None)
    return await chat_with_assistant(chat_payload, db)

import shutil

@app.post("/transcribe-audio")
async def transcribe_audio(file: UploadFile = File(...), db: Session = Depends(get_db)):
    # 1. Validate the file is not empty or invalid
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No filename provided in upload."
        )

    content = await file.read()
    if len(content) == 0:
        raise HTTPException(
            status_code=400,
            detail="Uploaded audio file is empty."
        )
    # Reset read cursor to beginning
    await file.seek(0)

    # 2. Setup temp directory
    temp_dir = os.path.join(os.path.dirname(__file__), "temp")
    os.makedirs(temp_dir, exist_ok=True)

    # Generate a safe local file path in the temp directory
    safe_filename = f"{int(time.time())}_{file.filename}"
    safe_filename = re.sub(r'[^a-zA-Z0-9_.-]', '_', safe_filename)
    temp_file_path = os.path.join(temp_dir, safe_filename)

    # 3. Save the uploaded file locally to the temp path
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save temporary audio file: {str(e)}"
        )

    # 4. Transcribe using pre-loaded whisper model
    try:
        if whisper_model is None:
            raise HTTPException(
                status_code=500,
                detail="Whisper model is not loaded/configured on the server."
            )
            
        print(f"[Whisper] Transcribing file: {temp_file_path}...")
        result = whisper_model.transcribe(temp_file_path)
        transcript = result.get("text", "").strip()
        print(f"[Whisper] Transcript: {transcript}")
        print("[Whisper] Transcription completed successfully!")
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error transcribing audio: {str(e)}"
        )
    finally:
        # 5. Clean up temporary file in finally block to ensure it's always deleted
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                print(f"[Whisper] Temporary file deleted: {temp_file_path}")
            except Exception as cleanup_err:
                print(f"[Whisper WARNING] Failed to delete temp file {temp_file_path}: {cleanup_err}")

    # 6. Validate transcript content
    if not transcript:
        raise HTTPException(
            status_code=400,
            detail="Transcription returned no text. The audio file might be silent or contain no speech."
        )

    # 7. Generate summary using Gemini/Groq via generate_text
    prompt = f"Summarize this video transcript clearly for a college student: {transcript}"
    
    try:
        summary_text = generate_text(prompt, is_json=False, fallback_type="text", text_preview=transcript)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate summary from transcript: {str(e)}"
        )

    # 8. Save the summary to database
    try:
        db_summary = Summary(
            title=f"Video: {file.filename}",
            url=None,
            summary=summary_text
        )
        db.add(db_summary)
        db.commit()
        db.refresh(db_summary)
        
        return {
            "summary": summary_text,
            "id": db_summary.id
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Database error saving summary: {str(e)}"
        )





