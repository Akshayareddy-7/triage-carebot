# doctor_agent.py
import ollama
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

# Using Mistral on CPU
MODEL_NAME = "phi3:mini"

def _shorten_reply(text, max_sentences=4):
    """Trim reply to a limited number of sentences."""
    if not text:
        return ""
    sentences = re.split(r'(?<=[.!?]) +', text.strip())
    return " ".join(sentences[:max_sentences])

def build_messages(message_history, triage_context=None):
    """Build messages array for Ollama."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    for m in message_history:
        role = m.get("role", "user")
        content = m.get("content") or m.get("message") or ""
        
        if not content:
            continue
        
        # Normalize roles
        if role in ["user", "patient"]:
            messages.append({"role": "user", "content": content})
        elif role in ["assistant", "doctor"]:
            messages.append({"role": "assistant", "content": content})
    
    # Add triage context if provided
    if triage_context:
        messages.append({
            "role": "system",
            "content": f"Triage context: {triage_context}"
        })
    
    return messages

def doctor_reply(message_history, triage_context=None):
    """Generate a doctor reply using Ollama with Mistral on CPU."""
    try:
        messages = build_messages(message_history, triage_context)
        
        print(f"🔵 Calling Ollama with Phi3 (CPU-only mode)")
        
        # CPU-optimized settings for Mistral
        response = ollama.chat(
            model=MODEL_NAME,
            messages=messages,
            options={
                'num_ctx': 2048,         # Reduced context (saves memory)
                'num_predict': 120,      # Shorter responses (faster)
                'temperature': 0.25,
                'num_gpu': 0,            # FORCE CPU ONLY
                'num_thread': 4,         # Use 4 CPU cores
                'repeat_penalty': 1.1,
                'top_k': 40,
                'top_p': 0.9
            }
        )
        
        raw_reply = response.get('message', {}).get('content', '')
        
        if not raw_reply:
            print("⚠️  Empty response from Mistral")
            return "I'm sorry, I didn't receive a proper response. Could you rephrase your question?", False
        
        print(f"✅ Got reply from Mistral (CPU)")
        
        reply = _shorten_reply(raw_reply)
        
        # Detect if conversation should end
        end_convo = "<END_CONVO>" in reply
        reply = reply.replace("<END_CONVO>", "").strip()
        
        return reply, end_convo
    
    except ollama.ResponseError as e:
        error_msg = str(e)
        print(f"❌ Ollama ResponseError: {error_msg}")
        
        if "system memory" in error_msg:
            return "Mistral requires more RAM. Close other applications and try again.", False
        
        return "The AI model encountered an error. Please try again.", False
    
    except Exception as e:
        print(f"❌ Unexpected error: {type(e).__name__}: {e}")
        return "I'm having trouble processing your request. Please try again.", False

# Test function
if __name__ == "__main__":
    print(f"🧪 Testing Ollama with Mistral (CPU mode)")
    print("-" * 50)
    
    test_history = [
        {"role": "user", "content": "I have a headache"}
    ]
    
    try:
        reply, end = doctor_reply(test_history)
        print("-" * 50)
        print(f"✅ Reply: {reply}")
        print(f"🏁 End conversation: {end}")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
