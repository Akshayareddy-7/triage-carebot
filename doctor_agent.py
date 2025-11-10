# doctor_agent.py
import requests
from utils import OLLAMA_URL, MODEL_NAME
import json
import re

SYSTEM_PROMPT = """You are a compassionate general practice physician.
Your replies must be short, clear, and focused — no more than 3–4 sentences.
Ask open-ended questions first, then clarifying questions.
Use empathetic, non-alarming language. Avoid jargon or explain it simply.
Do not give a definitive diagnosis — focus on understanding symptoms, severity, timing, and red flags.
When provided a triage_context, deliver it clearly and explain next steps.
Avoid unnecessary repetition or filler words.
If you believe the consultation is complete, end your response with the token <END_CONVO>.
"""

def _call_ollama(prompt, max_tokens=180):
    """Call the Ollama model with a given prompt."""
    url = f"{OLLAMA_URL}/api/generate"
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.25,
        "stream": False
    }
    r = requests.post(url, json=payload, timeout=60)
    r.raise_for_status()
    resp = r.json()
    return resp.get("response", "")

def _shorten_reply(text, max_sentences=4):
    """Trim reply to a limited number of sentences."""
    sentences = re.split(r'(?<=[.!?]) +', text.strip())
    return " ".join(sentences[:max_sentences])

def build_prompt(message_history, triage_context=None):
    """Build the full conversation prompt."""
    prompt = SYSTEM_PROMPT + "\n\nThe following is a conversation between a doctor and a patient.\n\n"

    for m in message_history:
        role = m.get("role", "user").capitalize()
        content = m.get("content") or m.get("message") or ""
        prompt += f"{role}: {content}\n"

    if triage_context:
        prompt += f"\nTriage context: {triage_context}\n"

    prompt += "Doctor:"
    return prompt

def doctor_reply(message_history, triage_context=None):
    """Generate a doctor reply using Ollama and detect if the conversation should end."""
    prompt = build_prompt(message_history, triage_context)
    raw_reply = _call_ollama(prompt)
    reply = _shorten_reply(raw_reply)
    
    # Detect if conversation should end
    end_convo = "<END_CONVO>" in reply
    reply = reply.replace("<END_CONVO>", "").strip()
    
    return reply, end_convo
