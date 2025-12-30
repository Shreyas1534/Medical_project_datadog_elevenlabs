from fastapi import FastAPI, File, UploadFile, HTTPException
import requests, json, os, re, time
from groq import Groq
from dotenv import load_dotenv
from fastapi.responses import FileResponse

# === UPDATED ELEVENLABS IMPORTS (NEW SDK) ===
from elevenlabs.client import ElevenLabs
from elevenlabs import VoiceSettings

import google.generativeai as genai  # Still usable

# ----------------------------------------
# Load ENV
# ----------------------------------------
load_dotenv()
app = FastAPI()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")  # NEW KEY HERE
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DATADOG_API_KEY = os.getenv("DATADOG_API_KEY")
DATADOG_APP_KEY = os.getenv("DATADOG_APP_KEY")
DD_SITE = os.getenv("DD_SITE", "us5.datadoghq.com")

if not all([GROQ_API_KEY, ELEVENLABS_API_KEY, GEMINI_API_KEY, DATADOG_API_KEY, DATADOG_APP_KEY]):
    raise Exception("❌ Missing API Keys in .env or Render Environment tab")

print("🚀 FastAPI running with Datadog Cloud Metrics")
print("📡 Sending metrics to:", f"https://api.{DD_SITE}")

# ----------------------------------------
# SEND METRICS → DATADOG HTTP API
# ----------------------------------------
def dd_metric(name, value=1, metric_type="count"):
    url = f"https://api.{DD_SITE}/api/v1/series?api_key={DATADOG_API_KEY}&application_key={DATADOG_APP_KEY}"
    payload = {
        "series": [{
            "metric": name,
            "type": metric_type,
            "points": [[int(time.time()), value]],
            "tags": ["env:prod", "service:medical_ai", "device:mobile"]
        }]
    }
    r = requests.post(url, json=payload)
    print(f"📡 Datadog [{name}] -> {r.status_code}")
    if r.status_code not in (200,202):
        print("⚠️ Datadog Rejected:", r.text)

# ----------------------------------------
# Clients (Groq, Gemini, ElevenLabs)
# ----------------------------------------
client = Groq(api_key=GROQ_API_KEY)
tts = ElevenLabs(api_key=ELEVENLABS_API_KEY)
genai.configure(api_key=GEMINI_API_KEY)

gemini_model = genai.GenerativeModel(
    model_name="models/gemini-2.5-flash",
    generation_config={"temperature": 0.2, "max_output_tokens": 90}
)

RENDER_URL = "https://medical-project-api.onrender.com/predict"

# ----------------------------------------
# JSON CLEANER
# ----------------------------------------
def extract_json(txt):
    match = re.search(r"\{[\s\S]*\}", txt)
    return json.loads(match.group()) if match else None

# ----------------------------------------
# GROQ MEDICAL REPORT GENERATION
# ----------------------------------------
def generate_llm_report(prediction, confidence):
    prompt = f"""
You are a medical report AI. Fill every field clearly.

PATIENT FINDING:
Condition: {prediction}
Confidence: {confidence*100:.2f}%

RETURN JSON ONLY:
{{
 "disease": "{prediction}",
 "confidence": "{confidence*100:.2f}%",
 "severity_assessment": "Mild/Moderate/Severe (choose best)",
 "detailed_explanation": "What this condition means medically.",
 "possible_symptoms": ["Fever", "Fatigue", "Chest discomfort"],
 "clinical_significance": "How serious this is and why it matters.",
 "recommended_next_steps": ["Meet a specialist", "Perform medical imaging tests", "Consider treatment options"],
 "disclaimer": "This is AI assistance, not medical confirmation."
}}
"""
    res = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )
    return extract_json(res.choices[0].message.content)

# ----------------------------------------
# Gemini summary
# ----------------------------------------
def gemini_summary(report):
    try:
        return gemini_model.generate_content(f"Simplify: {report}").text
    except:
        dd_metric("medical_ai.gemini.error")
        return "⚠️ Gemini quota exceeded. Use fallback."

# ----------------------------------------
# ElevenLabs Voice (MINIMAL OUTPUT TO SAVE CREDITS)
# ----------------------------------------
def generate_voice_from_report(report):
    try:
        # ⭐ Minimal credit usage voice output
        text = f"Detected disease: {report['disease']}. Confidence: {report['confidence']}."
        
        audio_stream = tts.text_to_speech.convert(
            text=text,
            voice_id="XrExE9yKIg1WjnnlVkGX",
            model_id="eleven_multilingual_v2",
            voice_settings=VoiceSettings(stability=0.50, similarity_boost=0.80)
        )

        with open("doctor_report.mp3", "wb") as f:
            for chunk in audio_stream:
                f.write(chunk)

        return "doctor_report.mp3"

    except Exception as e:
        print("❌ ElevenLabs ERROR:", e)
        dd_metric("medical_ai.voice.error")
        return None

# ----------------------------------------
# MAIN ENDPOINT
# ----------------------------------------
@app.post("/diagnose")
async def diagnose(file: UploadFile = File(...)):
    start = time.time()
    dd_metric("medical_ai.request")

    try:
        r = requests.post(RENDER_URL, files={"file": file.file})
        data = r.json()

        prediction = data["prediction"]
        confidence = float(data["confidence"])

        dd_metric("medical_ai.confidence", confidence, "gauge")

        report = generate_llm_report(prediction, confidence)
        summary = gemini_summary(report)
        generate_voice_from_report(report)

        dd_metric("medical_ai.latency", (time.time()-start)*1000, "gauge")

        return {
            "prediction": prediction,
            "confidence": confidence,
            "report": report,
            "summary": summary,
            "voice_report_url": "/voice-report"
        }

    except Exception as e:
        dd_metric("medical_ai.error")
        raise HTTPException(500, str(e))

@app.get("/voice-report")
def voice_report():
    return FileResponse("doctor_report.mp3", media_type="audio/mpeg")
