"""
AI Mock Interview Coach — Backend
Multi-agent system: Interviewer, Evaluator, Coach
"""

import os
import json
import logging
import base64
import tempfile
from uuid import uuid4
from pathlib import Path
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from groq import Groq

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("interview-coach")

load_dotenv(Path(__file__).parent / ".env")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    logger.warning("GROQ_API_KEY not set — API calls will fail at runtime.")

client = Groq(api_key=GROQ_API_KEY or "")

MODEL_NAME = "llama-3.1-8b-instant"
MAX_QUESTIONS = 10  # Hard cap on number of questions

PROMPTS_DIR = Path(__file__).parent / "prompts"

def _load_prompt(filename: str) -> str:
    path = PROMPTS_DIR / filename
    return path.read_text(encoding="utf-8")

INTERVIEWER_PROMPT = _load_prompt("interviewer.txt")
EVALUATOR_PROMPT   = _load_prompt("evaluator.txt")
COACH_PROMPT       = _load_prompt("coach.txt")

# ---------------------------------------------------------------------------
# In-memory session store
# ---------------------------------------------------------------------------

sessions: dict[str, dict] = {}

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class StartRequest(BaseModel):
    role: str
    background: str = ""
    focus_area: str = "mixed"

class AnswerRequest(BaseModel):
    session_id: str
    answer: str

class FeedbackRequest(BaseModel):
    session_id: str

# ---------------------------------------------------------------------------
# Agent 1 — Interviewer
# ---------------------------------------------------------------------------

def agent_interviewer(
    role: str,
    background: str,
    focus_area: str,
    conversation_history: list[dict],
    turn_count: int,
) -> str:
    # If we've hit the hard cap, force a closing message
    if turn_count >= MAX_QUESTIONS:
        return (
            "That wraps up our interview — we've covered a lot of ground today. "
            "Thank you so much for your time and thoughtful answers. Best of luck!"
        )

    system_prompt = INTERVIEWER_PROMPT.format(
        role=role,
        background=background or "Not provided",
        focus_area=focus_area,
    )

    messages = [{"role": "system", "content": system_prompt}]

    for msg in conversation_history:
        api_role = "user" if msg["role"] == "user" else "assistant"
        messages.append({"role": api_role, "content": msg["content"]})

    if not conversation_history:
        messages.append({
            "role": "user",
            "content": "Begin the interview now. Introduce yourself briefly and ask the first question.",
        })
    else:
        turns_left = MAX_QUESTIONS - turn_count
        if turns_left <= 2:
            messages.append({
                "role": "user",
                "content": (
                    f"Continue the interview. You have asked {turn_count} questions so far. "
                    f"You have at most {turns_left} question(s) left before you MUST wrap up. "
                    "If this is your last question, end with 'That wraps up our interview' after the candidate answers — "
                    "but for now ask your next question."
                ),
            })
        else:
            messages.append({
                "role": "user",
                "content": (
                    f"Continue the interview. You have asked {turn_count} questions so far out of a maximum of {MAX_QUESTIONS}. "
                    "Decide whether to follow up on the last answer or move to the next question."
                ),
            })

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.7,
    )

    return response.choices[0].message.content

# ---------------------------------------------------------------------------
# Agent 2 — Evaluator
# ---------------------------------------------------------------------------

def agent_evaluator(question: str, answer: str, role: str) -> dict | None:
    system_prompt = EVALUATOR_PROMPT.format(
        role=role,
        question=question,
        answer=answer,
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Evaluate the candidate's answer now. Return ONLY the JSON object."},
    ]

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.3,
    )

    raw = response.choices[0].message.content.strip()

    if raw.startswith("```"):
        lines = raw.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        raw = "\n".join(lines).strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse evaluator JSON: {e}\nRaw:\n{raw}")
        return None

# ---------------------------------------------------------------------------
# Agent 3 — Coach
# ---------------------------------------------------------------------------

def agent_coach(
    role: str,
    focus_area: str,
    conversation_history: list[dict],
    evaluations: list[dict],
) -> str:
    transcript_lines = []
    for msg in conversation_history:
        speaker = "Interviewer" if msg["role"] == "model" else "Candidate"
        transcript_lines.append(f"**{speaker}:** {msg['content']}")
    transcript_text = "\n\n".join(transcript_lines)

    eval_text = json.dumps(evaluations, indent=2)

    system_prompt = COACH_PROMPT.format(
        role=role,
        focus_area=focus_area,
        num_turns=len(evaluations),
        transcript=transcript_text,
        evaluations=eval_text,
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Generate the complete feedback report now."},
    ]

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.5,
    )

    return response.choices[0].message.content

# ---------------------------------------------------------------------------
# Audio Transcription (Groq Whisper)
# ---------------------------------------------------------------------------

def transcribe_audio(audio_bytes: bytes, filename: str) -> str:
    """Transcribe audio using Groq's Whisper API."""
    # Determine file extension
    ext = Path(filename).suffix.lower()
    if ext not in [".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm", ".ogg"]:
        ext = ".webm"  # Default for browser recordings

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        with open(tmp_path, "rb") as f:
            transcription = client.audio.transcriptions.create(
                model="whisper-large-v3",
                file=(Path(tmp_path).name, f),
                response_format="text",
            )
        return transcription.strip() if isinstance(transcription, str) else transcription.text.strip()
    finally:
        os.unlink(tmp_path)

# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------

app = FastAPI(title="AI Mock Interview Coach", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def serve_frontend():
    return FileResponse(
        Path(__file__).parent / "index.html",
        media_type="text/html",
    )

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/start")
async def start_interview(req: StartRequest):
    session_id = str(uuid4())

    session = {
        "role": req.role,
        "background": req.background,
        "focus_area": req.focus_area,
        "conversation_history": [],
        "evaluations": [],
        "turn_count": 0,
        "interview_done": False,
    }

    try:
        first_message = agent_interviewer(
            role=req.role,
            background=req.background,
            focus_area=req.focus_area,
            conversation_history=[],
            turn_count=0,
        )
    except Exception as e:
        logger.error(f"Interviewer agent failed: {e}")
        raise HTTPException(status_code=500, detail=f"Interviewer agent error: {str(e)}")

    session["conversation_history"].append({"role": "model", "content": first_message})
    session["turn_count"] = 1
    sessions[session_id] = session

    return {"session_id": session_id, "message": first_message, "turn": 1}


@app.post("/answer")
async def submit_answer(req: AnswerRequest):
    session = sessions.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if session["interview_done"]:
        raise HTTPException(status_code=400, detail="Interview already completed")

    session["conversation_history"].append({"role": "user", "content": req.answer})

    # Evaluator (silent)
    last_question = ""
    for msg in reversed(session["conversation_history"]):
        if msg["role"] == "model":
            last_question = msg["content"]
            break

    try:
        evaluation = agent_evaluator(
            question=last_question,
            answer=req.answer,
            role=session["role"],
        )
        if evaluation:
            evaluation["turn"] = session["turn_count"]
            session["evaluations"].append(evaluation)
    except Exception as e:
        logger.error(f"Evaluator agent failed (non-fatal): {e}")

    # Interviewer
    try:
        interviewer_response = agent_interviewer(
            role=session["role"],
            background=session["background"],
            focus_area=session["focus_area"],
            conversation_history=session["conversation_history"],
            turn_count=session["turn_count"],
        )
    except Exception as e:
        logger.error(f"Interviewer agent failed: {e}")
        raise HTTPException(status_code=500, detail=f"Interviewer agent error: {str(e)}")

    session["conversation_history"].append({"role": "model", "content": interviewer_response})
    session["turn_count"] += 1

    interview_done = (
        "that wraps up our interview" in interviewer_response.lower()
        or session["turn_count"] > MAX_QUESTIONS
    )
    session["interview_done"] = interview_done

    return {
        "message": interviewer_response,
        "turn": session["turn_count"],
        "interview_done": interview_done,
    }


@app.post("/transcribe")
async def transcribe(
    audio: UploadFile = File(...),
):
    """Transcribe audio and return the text."""
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file")

    try:
        text = transcribe_audio(audio_bytes, audio.filename or "recording.webm")
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription error: {str(e)}")

    return {"transcript": text}


@app.post("/feedback")
async def get_feedback(req: FeedbackRequest):
    session = sessions.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        feedback = agent_coach(
            role=session["role"],
            focus_area=session["focus_area"],
            conversation_history=session["conversation_history"],
            evaluations=session["evaluations"],
        )
    except Exception as e:
        logger.error(f"Coach agent failed: {e}")
        raise HTTPException(status_code=500, detail=f"Coach agent error: {str(e)}")

    return {"feedback_markdown": feedback}