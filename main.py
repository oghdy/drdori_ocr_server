from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
import os
from dotenv import load_dotenv
from PIL import Image
import io

load_dotenv()
app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Gemini 설정
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-2.5-flash')

print("✅ Gemini OCR Server Ready")

@app.post("/ocr")
async def ocr_endpoint(file: UploadFile = File(...)):
    try:
        # 이미지 읽기
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Gemini 프롬프트
        prompt = """
Analyze this Korean medication label image. It may contain one or multiple medications.

For EACH medication found in the image, provide:

**[Medication Name]**
- Korean name: (original Korean text)
- English translation: (translated name)
- Dosage timing: (when/how often to take)
- Instructions: (how to take it)
- Brief description: (what it's for, key warnings)

---

Repeat the above format for each medication in the image.
Keep explanations brief and practical. Focus on essential information patients need to know.
"""
        
        # Gemini API 호출
        response = model.generate_content([prompt, image])
        
        return {
            "success": True,
            "result": response.text
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"{type(e).__name__}: {str(e)}"
        }

@app.get("/")
async def root():
    return {"message": "Gemini OCR Server is running"}