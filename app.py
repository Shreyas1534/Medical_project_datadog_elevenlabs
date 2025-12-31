from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
import requests, json, os, re, time
from groq import Groq
from dotenv import load_dotenv
import google.generativeai as genai
from elevenlabs import generate, set_api_key   # 🟢 legacy SDK - works on Railway

# ----------------------------------------
# Load Environment Variables
# ----------------------------------------
load_dotenv()
app = FastAPI()

# API KEYS
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DATADOG_API_KEY = os.getenv("DATADOG_API_KEY")
DATADOG_APP_KEY = os.getenv("DATADOG_APP_KEY")
DD_SITE = os.getenv("DD_SITE", "us5.datadoghq.com")

# ElevenLabs Init
set_api_key(ELEVENLABS_API_KEY)
ALICE_VOICE = "Alice"  # 🎙 confirmed voice name

# Groq Client
client = Groq(api_key=GROQ_API_KEY)

# Gemini Setup
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config={"temperature": 0.2, "max_output_tokens": 200}
)

# Model Backend
BACKEND_MODEL_URL = "https://medical-project-api.onrender.com/predict"


# ----------------------------------------
# 📡 Datadog Metric (Auto-prefix + stable)
# ----------------------------------------
def dd_metric(name, value=1, metric_type="count"):
    if not DATADOG_API_KEY or not DATADOG_APP_KEY:
        print("⚠️ Datadog disabled (missing keys)")
        return
    try:
        # auto prefix to fix dashboard issue
        metric_name = f"medical_ai.{name}" if not name.startswith("medical_ai.") else name

        url = f"https://api.{DD_SITE}/api/v1/series?api_key={DATADOG_API_KEY}&application_key={DATADOG_APP_KEY}"
        payload = {
            "series": [{
                "metric": metric_name,
                "type": metric_type,
                "points": [[int(time.time()), value]],
                "tags": ["env:prod", "service:medical_ai", "runtime:fastapi"]
            }]
        }
        requests.post(url, json=payload)
        print(f"📡 Sent -> {metric_name}")
    except Exception as e:
        print("🚨 Datadog Send Failed:", e)


# ----------------------------------------
# JSON Extractor
# ----------------------------------------
def extract_json(text):
    match = re.search(r"\{[\s\S]*\}", text)
    return json.loads(match.group()) if match else {"error": "No JSON returned"}


# ----------------------------------------
# 🩺 Medical Report (Groq)
# ----------------------------------------
def generate_llm_report(prediction, confidence):
    score = f"{confidence * 100:.2f}%"
    prompt = f"""
    You are a medical diagnostic AI. Return JSON only:
    {{
      "disease": "{prediction}",
      "confidence_score": "{score}",
      "severity_assessment": "Low/Moderate/High",
      "detailed_explanation": "3-6 sentence medical explanation.",
      "possible_symptoms": ["symptom1","symptom2","symptom3"],
      "clinical_significance": "Why it matters medically.",
      "recommended_next_steps": ["test","treatment","doctor visit"],
      "specialist_to_consult": "Correct doctor type",
      "emergency_signs": ["danger sign 1","danger sign 2"],
      "patient_friendly_summary": "Easy explanation for patients.",
      "disclaimer": "AI assistance, not a confirmed diagnosis."
    }}
    """
    res = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return extract_json(res.choices[0].message.content)


# ----------------------------------------
# 🧠 Gemini Summary
# ----------------------------------------
def gemini_summary(report):
    try:
        dd_metric("gemini.request")
        out = gemini_model.generate_content("Simplify for patient: " + json.dumps(report))
        return out.text
    except Exception as e:
        dd_metric("gemini.error")
        return f"⚠️ Gemini failed: {e}"


# ----------------------------------------
# 🎙️ Voice (Alice) - Legacy Working Version
# ----------------------------------------
def generate_voice(report):
    try:
        text = (
            f"{report['disease']} detected with confidence {report['confidence_score']}. "
            f"{report.get('patient_friendly_summary','')}"
        )

        audio = generate(
            text=text,
            voice=ALICE_VOICE,
            model="eleven_multilingual_v2"
        )

        with open("doctor_report.mp3", "wb") as f:
            f.write(audio)

        dd_metric("voice.success")
        return "doctor_report.mp3"

    except Exception as e:
        dd_metric("voice.error")
        print("🚨 ElevenLabs Error:", e)
        return None


# ----------------------------------------
# 🌍 Root Health Check
# ----------------------------------------
@app.get("/")
def home():
    return {
        "status": "running",
        "service": "Medical AI",
        "routes": ["/diagnose", "/voice-report"]
    }


# ----------------------------------------
# 🚑 Diagnose Route
# ----------------------------------------
@app.post("/diagnose")
async def diagnose(file: UploadFile = File(...)):
    dd_metric("request")
    start = time.time()
    try:
        res = requests.post(BACKEND_MODEL_URL, files={"file": file.file})
        data = res.json()

        prediction = data.get("prediction")
        confidence = float(data.get("confidence", 0))

        report = generate_llm_report(prediction, confidence)
        summary = gemini_summary(report)
        generate_voice(report)

        dd_metric("latency", (time.time() - start) * 1000, "gauge")

        return {
            "prediction": prediction,
            "confidence": confidence,
            "medical_report": report,
            "patient_summary": summary,
            "voice_report_url": "/voice-report"
        }

    except Exception as e:
        dd_metric("error")
        raise HTTPException(500, f"SERVER ERROR: {str(e)}")


# ----------------------------------------
# 🔊 Voice File
# ----------------------------------------
@app.get("/voice-report")
def voice_report():
    if not os.path.exists("doctor_report.mp3"):
        raise HTTPException(404, "No voice report yet")
    return FileResponse("doctor_report.mp3", media_type="audio/mpeg")
