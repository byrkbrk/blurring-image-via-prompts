from fastapi import FastAPI, UploadFile, File, Form
from image_blurring import BlurImage
from PIL import Image



app = FastAPI()
blur_image = BlurImage(None)

@app.get("/")
def root():
    return {"message": "Blur objects by prompts"}

@app.post("/blur-image")
async def blur(file: UploadFile = File(...), 
               prompt: str = Form(...),
               blur_intensity: int = Form(...)):
    prompt = prompt.split(",")
    blur_image.blur(Image.open(file.file), 
                    prompt,
                    blur_intensity,
                    )
    return {"filename": file.filename, 
            "prompt": prompt, 
            "blur_intensity": blur_intensity}
