from fastapi import FastAPI, File, UploadFile, HTTPException
import requests, json, os, re, time
from groq import Groq
from dotenv import load_dotenv
from fastapi.responses import FileResponse

# === FIXED ELEVENLABS IMPORTS (NO "client" MODULE) ===
from elevenlabs import ElevenLabs, VoiceSettings

# === GEMINI ===
import google.generativeai as genai

# ----------------------------------------
# Load ENV
# ----------------------------------------
load_dotenv()
app = FastAPI()

# API Keys from Environment (Railway Variables)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

DATADOG_API_KEY = os.getenv("DATADOG_API_KEY")
DATADOG_APP_KEY = os.getenv("DATADOG_APP_KEY")
DD_SITE = os.getenv("DD_SITE", "us5.datadoghq.com")

# Check keys (SAFELY)
missing = [k for k,v in {
    "GROQ_API_KEY":GROQ_API_KEY,
    "ELEVENLABS_API_KEY":ELEVENLABS_API_KEY,
    "GEMINI_API_KEY":GEMINI_API_KEY,
    "DATADOG_API_KEY":DATADOG_API_KEY,
    "DATADOG_APP_KEY":DATADOG_APP_KEY
}.items() if not v]

if missing:
    print("⚠️ Missing keys:", missing)
else:
    print("🔐 All API keys loaded successfully!")

# ----------------------------------------
# Datadog Metric Sender (No Crash if Missing)
# ----------------------------------------
def dd_metric(name, value=1, metric_type="count"):
    if not DATADOG_API_KEY or not DATADOG_APP_KEY:
        print("⚠️ Datadog keys missing, skipping metric.")
        return
    
    url = f"https://api.{DD_SITE}/api/v1/series?api_key={DATADOG_API_KEY}&application_key={DATADOG_APP_KEY}"
    payload = {
        "series": [{
            "metric": name,
            "type": metric_type,
            "points": [[int(time.time()), value]],
            "tags": ["env:prod", "service:medical_ai"]
        }]
    }
    try:
        requests.post(url, json=payload)
    except:
        print("⚠️ Failed to send metric:", name)

# ----------------------------------------
# Clients
# ----------------------------------------
client = Groq(api_key=GROQ_API_KEY)
tts = ElevenLabs(api_key=ELEVENLABS_API_KEY)
genai.configure(api_key=GEMINI_API_KEY)

gemini_model = genai.GenerativeModel(
    model_name="models/gemini-2.5-flash",
    generation_config={"temperature": 0.2, "max_output_tokens": 120}
)

BACKEND_MODEL_URL = "https://medical-project-api.onrender.com/predict"

# ----------------------------------------
# Clean JSON
# ----------------------------------------
def extract_json(txt):
    match = re.search(r"\{[\s\S]*\}", txt)
    return json.loads(match.group()) if match else {}

# ----------------------------------------
# Generate Medical Report via LLM
# ----------------------------------------
def generate_llm_report(prediction, confidence):
    prompt = f"""
Generate a structured medical report:

{{
 "disease": "{prediction}",
 "confidence": "{confidence*100:.2f}%",
 "severity": "Mild/Moderate/Severe (choose)",
 "symptoms": ["Fever","Fatigue","Chest discomfort"],
 "clinical_significance": "Meaning & impact",
 "recommended_next_steps": ["Meet specialist","Medical imaging","Lab tests"],
 "disclaimer": "This is AI assistance, not medical confirmation."
}}
"""
    res = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )
    return extract_json(res.choices[0].message.content)

# ----------------------------------------
# Gemini: Simplify Summary
# ----------------------------------------
def gemini_summary(report):
    try:
        return gemini_model.generate_content(f"Simplify: {report}").text
    except:
        return "⚠️ Gemini unavailable, summary skipped."

# ----------------------------------------
# ElevenLabs Voice Output
# ----------------------------------------
def generate_voice(report):
    try:
        text = f"Detected: {report['disease']}. Confidence: {report['confidence']}."
        audio = tts.text_to_speech.convert(
            text=text,
            voice_id="XrExE9yKIg1WjnnlVkGX",
            model_id="eleven_multilingual_v2",
            voice_settings=VoiceSettings(stability=0.50, similarity_boost=0.80)
        )
        with open("doctor_report.mp3", "wb") as f:
            for chunk in audio:
                f.write(chunk)
        return "doctor_report.mp3"
    except Exception as e:
        print("⚠️ ElevenLabs Error:", e)
        return None

# ----------------------------------------
# ROUTES
# ----------------------------------------
@app.post("/diagnose")
async def diagnose(file: UploadFile = File(...)):
    start = time.time()
    dd_metric("medical_ai.request")
    
    try:
        r = requests.post(BACKEND_MODEL_URL, files={"file": file.file})
        data = r.json()

        prediction = data.get("prediction")
        confidence = float(data.get("confidence", 0))

        report = generate_llm_report(prediction, confidence)
        summary = gemini_summary(report)
        generate_voice(report)

        return {
            "prediction": prediction,
            "confidence": confidence,
            "report": report,
            "summary": summary,
            "voice_report_url": "/voice-report"
        }

    except Exception as e:
        raise HTTPException(500, f"SERVER ERROR: {str(e)}")

@app.get("/voice-report")
def voice_report():
    return FileResponse("doctor_report.mp3", media_type="audio/mpeg")
