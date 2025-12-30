from fastapi import FastAPI, File, UploadFile, HTTPException
import requests, json, os, re, time
from groq import Groq
from dotenv import load_dotenv
from fastapi.responses import FileResponse
import google.generativeai as genai

# === ✔ CORRECT ELEVENLABS IMPORTS ===
from elevenlabs import generate, VoiceSettings, set_api_key


# ----------------------------------------
# Load ENV
# ----------------------------------------
load_dotenv()
app = FastAPI()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DATADOG_API_KEY = os.getenv("DATADOG_API_KEY")
DATADOG_APP_KEY = os.getenv("DATADOG_APP_KEY")
DD_SITE = os.getenv("DD_SITE", "us5.datadoghq.com")

# Set key for ElevenLabs
if ELEVENLABS_API_KEY:
    set_api_key(ELEVENLABS_API_KEY)


# ----------------------------------------
# Datadog Metric Sender (Safe)
# ----------------------------------------
def dd_metric(name, value=1, metric_type="count"):
    if not DATADOG_API_KEY or not DATADOG_APP_KEY:
        return
    try:
        url = f"https://api.{DD_SITE}/api/v1/series?api_key={DATADOG_API_KEY}&application_key={DATADOG_APP_KEY}"
        payload = {
            "series": [{
                "metric": name,
                "type": metric_type,
                "points": [[int(time.time()), value]],
                "tags": ["env:prod", "service:medical_ai"]
            }]
        }
        requests.post(url, json=payload)
    except:
        print("⚠️ Datadog metric failed")


# ----------------------------------------
# Clients
# ----------------------------------------
client = Groq(api_key=GROQ_API_KEY)
genai.configure(api_key=GEMINI_API_KEY)

gemini_model = genai.GenerativeModel(
    model_name="models/gemini-2.5-flash",
    generation_config={"temperature": 0.2, "max_output_tokens": 120}
)

BACKEND_MODEL_URL = "https://medical-project-api.onrender.com/predict"


# ----------------------------------------
# Helper
# ----------------------------------------
def extract_json(txt):
    match = re.search(r"\{[\s\S]*\}", txt)
    return json.loads(match.group()) if match else {}


# ----------------------------------------
# Groq Medical Report
# ----------------------------------------
def generate_llm_report(prediction, confidence):
    prompt = f"""
Return JSON:

{{
 "disease": "{prediction}",
 "confidence": "{confidence*100:.2f}%",
 "severity": "Mild/Moderate/Severe",
 "symptoms": ["Fever","Fatigue","Chest discomfort"],
 "clinical_significance": "What this means medically",
 "steps": ["Meet specialist","Tests recommended","Monitoring advice"],
 "disclaimer": "Not a substitute for medical diagnosis."
}}
"""
    res = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )
    return extract_json(res.choices[0].message.content)


# ----------------------------------------
# Gemini Summary
# ----------------------------------------
def gemini_summary(report):
    try:
        return gemini_model.generate_content(f"Summarize: {report}").text
    except:
        return "⚠️ Gemini unavailable."


# ----------------------------------------
# ElevenLabs Voice
# ----------------------------------------
def generate_voice(report):
    try:
        text = f"Detected: {report['disease']} with confidence {report['confidence']}."
        audio = generate(
            text=text,
            voice="Rachel",
            model="eleven_multilingual_v2",
            voice_settings=VoiceSettings(stability=0.50, similarity_boost=0.80)
        )
        with open("doctor_report.mp3", "wb") as f:
            f.write(audio)
        return "doctor_report.mp3"
    except Exception as e:
        print("⚠️ ElevenLabs error:", e)
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
