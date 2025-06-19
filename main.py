import base64
import io
import time
import uuid
import logging
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel, HttpUrl
from PIL import Image
import requests
import numpy as np
import cv2
import easyocr
from lama_cleaner.model_manager import ModelManager
from lama_cleaner.schema import InpaintRequest

# --- CONFIGURATION ---
API_KEYS = {"your_api_key_123"}
STORAGE_URL_PREFIX = "https://your-storage.example.com/images/"  # Simulated

# --- LOGGING ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("watermark_removal_api")

# --- FASTAPI APP ---
app = FastAPI(title="AI Watermark/Logo/Text Removal API")

# --- MODELS ---
class ImageInput(BaseModel):
    url: Optional[HttpUrl] = None
    base64_data: Optional[str] = None

class ProcessRequest(BaseModel):
    images: List[ImageInput]
    output_format: Optional[str] = "png"  # "jpeg", "png", "webp"
    quality: Optional[int] = 90
    return_base64: Optional[bool] = False

class ImageResult(BaseModel):
    original_dimensions: List[int]
    processed_dimensions: List[int]
    processing_time_ms: int
    image_url: Optional[str] = None
    base64_data: Optional[str] = None
    status: str
    error: Optional[str] = None

class ProcessResponse(BaseModel):
    results: List[ImageResult]

# --- AUTH ---
def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key not in API_KEYS:
        logger.warning("Unauthorized access attempt.")
        raise HTTPException(status_code=401, detail="Invalid API Key")

# --- AI MODELS ---
reader = easyocr.Reader(['en'], gpu=False)
lama_model = ModelManager(name="lama", device="cpu")  # Use "cuda" if you have a GPU

# --- IMAGE HELPERS ---
def download_image(url: str) -> Image.Image:
    response = requests.get(url)
    response.raise_for_status()
    return Image.open(io.BytesIO(response.content)).convert("RGB")

def decode_base64_image(data: str) -> Image.Image:
    decoded = base64.b64decode(data)
    return Image.open(io.BytesIO(decoded)).convert("RGB")

def detect_text_mask(image: Image.Image) -> np.ndarray:
    np_img = np.array(image)
    results = reader.readtext(np_img)
    mask = np.zeros(np_img.shape[:2], dtype=np.uint8)
    for bbox, text, conf in results:
        pts = np.array(bbox, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)
    return mask

def ai_detect_and_remove_watermark(image: Image.Image) -> Image.Image:
    mask = detect_text_mask(image)
    if mask.sum() == 0:
        return image  # No text detected
    np_img = np.array(image)
    # LaMa expects BGR
    np_img_bgr = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
    mask = (mask > 0).astype(np.uint8) * 255
    # Inpaint with LaMa
    inpaint_req = InpaintRequest(
        image=np_img_bgr,
        mask=mask,
        ldm_steps=25,
        hd_strategy="Original",
        hd_strategy_crop_margin=32,
        hd_strategy_crop_trigger_size=512,
        hd_strategy_resize_limit=2048,
        prompt="",
        negative_prompt="",
        use_croper=False,
        croper_x=0,
        croper_y=0,
        croper_height=512,
        croper_width=512,
        sampler="ddim",
        ddim_steps=50,
        strength=1.0,
        seed=42,
        guidance_scale=7.5,
        eta=0.0,
        only_masked_area=False,
        box_threshold=0.3,
        padding=10,
        is_replace_background=False,
        background_color="#000000"
    )
    result = lama_model(image=inpaint_req.image, mask=inpaint_req.mask)
    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    return Image.fromarray(result_rgb)

def save_image_to_storage(image: Image.Image, fmt: str, quality: int) -> str:
    filename = f"{uuid.uuid4()}.{fmt}"
    image.save(f"/tmp/{filename}", format=fmt.upper(), quality=quality)
    return STORAGE_URL_PREFIX + filename

# --- API ENDPOINT ---
@app.post("/process", response_model=ProcessResponse, tags=["Image Processing"])
async def process_images(
    request: ProcessRequest,
    x_api_key: str = Depends(verify_api_key)
):
    results = []
    for img_input in request.images:
        start_time = time.time()
        try:
            if img_input.url:
                image = download_image(img_input.url)
            elif img_input.base64_data:
                image = decode_base64_image(img_input.base64_data)
            else:
                raise ValueError("No image input provided.")

            original_size = list(image.size)
            processed_image = ai_detect_and_remove_watermark(image)
            processed_size = list(processed_image.size)
            processing_time = int((time.time() - start_time) * 1000)

            if request.return_base64:
                buffered = io.BytesIO()
                processed_image.save(buffered, format=request.output_format.upper(), quality=request.quality)
                img_str = base64.b64encode(buffered.getvalue()).decode()
                result = ImageResult(
                    original_dimensions=original_size,
                    processed_dimensions=processed_size,
                    processing_time_ms=processing_time,
                    base64_data=img_str,
                    status="success"
                )
            else:
                url = save_image_to_storage(processed_image, request.output_format, request.quality)
                result = ImageResult(
                    original_dimensions=original_size,
                    processed_dimensions=processed_size,
                    processing_time_ms=processing_time,
                    image_url=url,
                    status="success"
                )
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            result = ImageResult(
                original_dimensions=[],
                processed_dimensions=[],
                processing_time_ms=int((time.time() - start_time) * 1000),
                status="error",
                error=str(e)
            )
        results.append(result)
    return ProcessResponse(results=results)
