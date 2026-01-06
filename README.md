This repository contains the backend AI pipeline for Auralysis, deployed on Railway.
It orchestrates medical report generation, summarization, voice output, and monitoring after receiving model predictions from a deployed ML service on Render.

What This Service Does

This backend acts as the central controller of the Auralysis pipeline.

It:

---Accepts prediction results from a TensorFlow model hosted on Render

---Generates a structured medical report using Groq LLaMA

---Simplifies the report using Google Gemini

---Converts the result into a doctor-style voice using ElevenLabs

---Pushes runtime metrics and logs to Datadog

---Returns text + voice output to the client
