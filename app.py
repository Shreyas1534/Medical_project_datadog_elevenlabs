from fastapi import FastAPI, File, UploadFile, HTTPException
import requests, json, os, re, time
from groq import Groq
from dotenv import load_dotenv
from fastapi.responses import FileResponse
import google.generativeai as genai
from elevenlabs import generate, VoiceSettings, set_api_key

# ----------------------------------------
# Load ENV
# ----------------------------------------
load_dotenv()
app = FastAPI()

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DATADOG_API_KEY = os.getenv("DATADOG_API_KEY")
DATADOG_APP_KEY = os.getenv("DATADOG_APP_KEY")
DD_SITE = os.getenv("DD_SITE", "us5.datadoghq.com")

if ELEVENLABS_API_KEY:
    set_api_key(ELEVENLABS_API_KEY)

# ----------------------------------------
# Monitoring (Optional)
# ----------------------------------------
def dd_metric(name, value=1, metric_type="count"):
    if not (DATADOG_API_KEY and DATADOG_APP_KEY): return
    try:
        url = f"https://api.{DD_SITE}/api/v1/series?api_key={DATADOG_API_KEY}&application_key={DATADOG_APP_KEY}"
        payload = {
            "series":[{ 
                "metric": name, "type": metric_type,
                "points":[[int(time.time()), value]],
                "tags":["service:medical_ai", "env:prod"]
            }]
        }
        requests.post(url, json=payload)
    except:
        print("⚠️ Datadog metric failed")

# ----------------------------------------
# AI Clients
# ----------------------------------------
client = Groq(api_key=GROQ_API_KEY)

genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config={"temperature": 0.2, "max_output_tokens": 200}
)

BACKEND_MODEL_URL = "https://medical-project-api.onrender.com/predict"


# ----------------------------------------
# Helpers
# ----------------------------------------
def extract_json(text):
    match = re.search(r"\{[\s\S]*\}", text)
    return json.loads(match.group()) if match else {"error":"No JSON returned"}

# ---------------- GROQ Report Generation ----------------
def generate_llm_report(prediction, confidence):

    confidence_value = f"{confidence*100:.2f}%"

    prompt = f"""
You are a licensed clinical medical diagnostic AI.
Diagnosis: {prediction}
Confidence: {confidence_value}

Return ONLY JSON (no markdown, no extra text):

{{
  "disease": "{prediction}",
  "confidence_score": "{confidence_value}",
  "severity_assessment": "Low / Moderate / High",
  "detailed_explanation": "Brief medical explanation.",
  "possible_symptoms": ["symptom 1", "symptom 2", "symptom 3"],
  "clinical_significance": "Why condition matters",
  "recommended_next_steps": ["Step 1", "Step 2", "Step 3"],
  "specialist_to_consult": "Doctor name",
  "emergency_signs": ["critical sign 1", "critical sign 2"],
  "patient_friendly_summary": "Simplified explanation.",
  "disclaimer": "AI-generated, not a final diagnosis."
}}
"""
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role":"system","content":"Return ONLY JSON."},
            {"role":"user","content":prompt}
        ],
        extra_body={"temperature": 0.3}
    )

    return extract_json(response.choices[0].message.content)


# ----------------------------------------
# Gemini Friendly Summary
# ----------------------------------------
def gemini_summary(report):
    try:
        result = gemini_model.generate_content(
            "Explain simply for a normal patient: " + json.dumps(report)
        )
        return result.text
    except:
        return "⚠️ Gemini unavailable."


# ----------------------------------------
# ElevenLabs - IMPROVED VOICE OUTPUT
# ----------------------------------------
def generate_voice(report):
    try:
        disease = report.get("disease", "Unknown condition")
        confidence = report.get("confidence_score", "N/A")
        symptoms = ", ".join(report.get("possible_symptoms", [])[:3])  # first 3
        next_steps = ", ".join(report.get("recommended_next_steps", [])[:3])  # first 3

        text = f"""
Medical alert. The detected condition is {disease}.
Confidence score: {confidence}.
Possible symptoms include: {symptoms}.
Recommended next steps: {next_steps}.
Please consult a medical professional for further evaluation.
"""

        audio = generate(
            text=text,
            voice="Rachel",
            model="eleven_multilingual_v2",
            voice_settings=VoiceSettings(stability=0.55, similarity_boost=0.85),
            output_format="mp3"
        )

        with open("doctor_report.mp3","wb") as f:
            f.write(audio)

        return "doctor_report.mp3"
    except Exception as e:
        print("⚠️ ElevenLabs Error:", e)
        return None


# ----------------------------------------
# ROUTES
# ----------------------------------------
@app.post("/diagnose")
async def diagnose(file: UploadFile = File(...)):
    dd_metric("medical_ai.request")
    start = time.time()

    try:
        r = requests.post(BACKEND_MODEL_URL, files={"file": file.file})
        data = r.json()

        prediction = data.get("prediction")
        confidence = float(data.get("confidence",0))

        report = generate_llm_report(prediction,confidence)
        summary = gemini_summary(report)
        generate_voice(report)

        dd_metric("medical_ai.latency",(time.time()-start)*1000,"gauge")

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


@app.get("/voice-report")
def voice_report():
    if not os.path.exists("doctor_report.mp3"):
        raise HTTPException(404,"Voice report not generated yet")
    return FileResponse("doctor_report.mp3", media_type="audio/mpeg")
