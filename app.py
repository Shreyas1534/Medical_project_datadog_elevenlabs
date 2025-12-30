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
            "series":[{ "metric": name, "type": metric_type,
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

# ---------------- REAL GROQ MEDICAL REPORT ----------------
def generate_llm_report(prediction, confidence):

    confidence_score = f"{confidence*100:.2f}%"  # safe formatting

    prompt = f"""
You are a licensed medical diagnostic AI. Analyze the detected condition and generate a medically accurate clinical report.

Detected diagnosis: {prediction}
Model confidence: {confidence_score}

Return ONLY JSON. NO notes or markdown.

{{
  "disease": "{prediction}",
  "confidence_score": "{confidence_score}",
  "severity_assessment": "Assess condition as Low, Moderate, or High -- based on medical likelihood.",
  "detailed_explanation": "Explain this condition in 3-6 sentences using real medical context.",
  "possible_symptoms": "List 5-8 realistic symptoms linked to this condition.",
  "clinical_significance": "Medical impact, why this matters, what may happen without treatment.",
  "recommended_next_steps": "3-6 next steps including diagnostic tests or treatment guidance.",
  "specialist_to_consult": "Name relevant doctor like Neurologist/Oncologist/Endocrinologist.",
  "emergency_signs": "List 3-5 red flag symptoms requiring urgent ER care.",
  "patient_friendly_summary": "Explain clearly as if speaking to a non-medical person.",
  "disclaimer": "This is AI-generated medical assistance, not a confirmed diagnosis."
}}
    """

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4
    )

    return extract_json(response.choices[0].message.content)


# ----------------------------------------
# Gemini Friendly Summary
# ----------------------------------------
def gemini_summary(report):
    try:
        result = gemini_model.generate_content(
            "Make this easier to understand for a normal patient: " + json.dumps(report)
        )
        return result.text
    except:
        return "⚠️ Gemini unavailable."

# ----------------------------------------
# ElevenLabs - Audio Summary
# ----------------------------------------
# ----------------------------------------
# ElevenLabs - Audio Summary (FIXED & DYNAMIC)
# ----------------------------------------
def generate_voice(report):
    try:
        # Ensure fields are lists
        symptoms = report.get("possible_symptoms", [])
        if isinstance(symptoms, str): symptoms = [symptoms]

        steps = report.get("recommended_next_steps", [])
        if isinstance(steps, str): steps = [steps]

        text = (
        f"Detected condition: {report['disease']}. "
        f"Confidence score: {report['confidence_score']}. "
        f"Symptoms may include: {', '.join(symptoms)}. "
        f"Recommended next steps: {', '.join(steps)}. "
        f"Please consult a specialist for confirmation."
        )

        audio = generate(
            text=text,
            voice="Rachel",
            model="eleven_multilingual_v2"  # ❗No voice_settings needed
        )

        with open("doctor_report.mp3", "wb") as f:
            f.write(audio)

        print("🎤 Voice note generated successfully!")
        return "doctor_report.mp3"

    except Exception as e:
        print("🚨 ElevenLabs Voice Error:", e)
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
