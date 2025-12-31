from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
import requests, json, os, re, time
from groq import Groq
from dotenv import load_dotenv
import google.generativeai as genai
from elevenlabs import generate, set_api_key

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

# ElevenLabs Setup
if ELEVENLABS_API_KEY:
    set_api_key(ELEVENLABS_API_KEY)

# Groq Client
client = Groq(api_key=GROQ_API_KEY)

# Gemini Setup
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config={"temperature": 0.2, "max_output_tokens": 200}
)

# Backend Model Prediction URL
BACKEND_MODEL_URL = "https://medical-project-api.onrender.com/predict"


# ----------------------------------------
# 📡 DATADOG Metric Sender (RESTORED)
# ----------------------------------------
def dd_metric(name, value=1, metric_type="count"):
    if not all([DATADOG_API_KEY, DATADOG_APP_KEY]):
        print("⚠️ Datadog disabled: missing keys")
        return
    
    try:
        url = f"https://api.{DD_SITE}/api/v1/series?api_key={DATADOG_API_KEY}&application_key={DATADOG_APP_KEY}"
        payload = {
            "series": [{
                "metric": name,
                "type": metric_type,
                "points": [[int(time.time()), value]],
                "tags": ["env:prod", "service:medical_ai", "device:api", "runtime:fastapi"]
            }]
        }
        r = requests.post(url, json=payload)
        print(f"📡 Sent -> {name} | Status: {r.status_code}")
        if r.status_code not in (200, 202):
            print("⚠️ Datadog Error:", r.text)
    except Exception as e:
        print("🚨 Metric Send Failed:", e)


# ----------------------------------------
# JSON Extraction Helper
# ----------------------------------------
def extract_json(text):
    match = re.search(r"\{[\s\S]*\}", text)
    return json.loads(match.group()) if match else {"error":"No JSON returned"}


# ----------------------------------------
# 🩺 Generate Medical Report (Groq)
# ----------------------------------------
def generate_llm_report(prediction, confidence):
    confidence_score = f"{confidence * 100:.2f}%"

    prompt = f"""
    You are a licensed medical diagnostic AI. Provide a structured medical report.

    Detected condition: {prediction}
    Confidence: {confidence_score}

    Return JSON ONLY:
    {{
      "disease": "{prediction}",
      "confidence_score": "{confidence_score}",
      "severity_assessment": "Low/Moderate/High",
      "detailed_explanation": "3-6 sentence medical explanation.",
      "possible_symptoms": ["symptom1", "symptom2", "symptom3"],
      "clinical_significance": "Why it matters medically.",
      "recommended_next_steps": ["test1", "treatment1", "doctor visit"],
      "specialist_to_consult": "Correct doctor type",
      "emergency_signs": ["danger sign 1", "danger sign 2"],
      "patient_friendly_summary": "Easily understandable version for a patient.",
      "disclaimer": "AI-based assistance, not a confirmed diagnosis."
    }}
    """

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4
    )

    return extract_json(response.choices[0].message.content)


# ----------------------------------------
# 🧠 Gemini Patient Friendly Summary
# ----------------------------------------
def gemini_summary(report):
    try:
        result = gemini_model.generate_content("Make patient-friendly: " + json.dumps(report))
        return result.text
    except:
        dd_metric("medical_ai.gemini.error")
        return "⚠️ Gemini unavailable."


# ----------------------------------------
# 🎙️ ElevenLabs Voice Report Generation
# ----------------------------------------
def generate_voice(report):
    try:
        text = (
            f"Detected condition: {report['disease']}. "
            f"Confidence: {report['confidence_score']}. "
            f"{report['patient_friendly_summary']}"
        )

        audio = generate(
            text=text,
            voice="Bella",
            model="eleven_multilingual_v2"
        )

        with open("doctor_report.mp3", "wb") as f:
            f.write(audio)

        return "doctor_report.mp3"
    except Exception as e:
        dd_metric("medical_ai.voice.error")
        print("🚨 Voice Error:", e)
        return None


# ----------------------------------------
# 🚑 MAIN DIAGNOSIS ROUTE
# ----------------------------------------
@app.post("/diagnose")
async def diagnose(file: UploadFile = File(...)):
    dd_metric("medical_ai.request")
    start = time.time()

    try:
        r = requests.post(BACKEND_MODEL_URL, files={"file": file.file})
        data = r.json()

        prediction = data.get("prediction")
        confidence = float(data.get("confidence", 0))

        dd_metric("medical_ai.confidence", confidence, "gauge")

        report = generate_llm_report(prediction, confidence)
        summary = gemini_summary(report)
        generate_voice(report)

        dd_metric("medical_ai.latency", (time.time() - start) * 1000, "gauge")

        return {
            "prediction": prediction,
            "confidence": confidence,
            "medical_report": report,
            "patient_summary": summary,
            "voice_report_url": "/voice-report"
        }

    except Exception as e:
        dd_metric("medical_ai.error")
        raise HTTPException(500, f"SERVER ERROR: {str(e)}")


# ----------------------------------------
# 🔊 Voice Report Download
# ----------------------------------------
@app.get("/voice-report")
def voice_report():
    if not os.path.exists("doctor_report.mp3"):
        raise HTTPException(404, "No voice report generated yet")
    return FileResponse("doctor_report.mp3", media_type="audio/mpeg")
