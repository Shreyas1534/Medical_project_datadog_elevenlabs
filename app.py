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

# API Keys (Railway Variables)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DATADOG_API_KEY = os.getenv("DATADOG_API_KEY")
DATADOG_APP_KEY = os.getenv("DATADOG_APP_KEY")
DD_SITE = os.getenv("DD_SITE", "us5.datadoghq.com")

# Set ElevenLabs key (no crash if missing)
if ELEVENLABS_API_KEY:
    set_api_key(ELEVENLABS_API_KEY)

# ----------------------------------------
# Optional Datadog Monitoring
# ----------------------------------------
def dd_metric(name, value=1, metric_type="count"):
    if not (DATADOG_API_KEY and DATADOG_APP_KEY):
        return
    try:
        url = f"https://api.{DD_SITE}/api/v1/series?api_key={DATADOG_API_KEY}&application_key={DATADOG_APP_KEY}"
        payload = {
            "series": [{
                "metric": name,
                "type": metric_type,
                "points": [[int(time.time()), value]],
                "tags": ["service:medical_ai", "env:prod"]
            }]
        }
        requests.post(url, json=payload)
    except:
        print("⚠️ Datadog metric failed:", name)

# ----------------------------------------
# AI Client Setup
# ----------------------------------------
client = Groq(api_key=GROQ_API_KEY)

genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",                 # ✔ correct public model
    generation_config={"temperature": 0.2, "max_output_tokens": 120}
)

BACKEND_MODEL_URL = "https://medical-project-api.onrender.com/predict"

# ----------------------------------------
# Tools
# ----------------------------------------
def extract_json(text):
    match = re.search(r"\{[\s\S]*\}", text)
    return json.loads(match.group()) if match else {}


def generate_llm_report(prediction, confidence):
    prompt = f"""
Return JSON only:

{{
 "disease": "{prediction}",
 "confidence": "{confidence*100:.2f}%",
 "severity": "Mild/Moderate/Severe (choose best)",
 "symptoms": ["Fever","Fatigue","Chest discomfort"],
 "clinical_significance": "Medical meaning",
 "steps": ["Meet specialist","Further tests","Monitoring recommended"],
 "disclaimer": "Not a medical confirmation. AI assistance only."
}}
"""
    res = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )
    return extract_json(res.choices[0].message.content)


def gemini_summary(report):
    try:
        summary = gemini_model.generate_content(f"Summarize this for patient: {report}")
        return summary.text
    except:
        return "⚠️ Gemini Unavailable. Summary Skipped."


def generate_voice(report):
    try:
        text = f"Detected condition: {report['disease']} with {report['confidence']} confidence."
        audio = generate(
            text=text,
            voice="Rachel",
            model="eleven_multilingual_v2",
            voice_settings=VoiceSettings(stability=0.50, similarity_boost=0.80),
            output_format="mp3"                      # ✔ required for Railway
        )
        with open("doctor_report.mp3", "wb") as f:
            f.write(audio)
        return "doctor_report.mp3"
    except Exception as e:
        print("⚠️ ElevenLabs TTS Error:", e)
        return None

# ----------------------------------------
# ROUTES
# ----------------------------------------
@app.post("/diagnose")
async def diagnose(file: UploadFile = File(...)):
    dd_metric("medical_ai.request")
    start = time.time()

    try:
        # Send image to backend ML API
        r = requests.post(BACKEND_MODEL_URL, files={"file": file.file})
        if r.status_code != 200:
            raise Exception("Backend model not reachable")

        data = r.json()
        prediction = data.get("prediction")
        confidence = float(data.get("confidence", 0))

        report = generate_llm_report(prediction, confidence)
        summary = gemini_summary(report)
        generate_voice(report)

        dd_metric("medical_ai.latency", (time.time() - start) * 1000, "gauge")

        return {
            "prediction": prediction,
            "confidence": confidence,
            "report": report,
            "summary": summary,
            "voice_report_url": "/voice-report"
        }

    except Exception as e:
        dd_metric("medical_ai.error")
        raise HTTPException(500, f"SERVER ERROR: {str(e)}")


@app.get("/voice-report")
def voice_report():
    if not os.path.exists("doctor_report.mp3"):
        raise HTTPException(404, "Voice report not generated yet")
    return FileResponse("doctor_report.mp3", media_type="audio/mpeg")
