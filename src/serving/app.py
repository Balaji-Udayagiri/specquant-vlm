"""Day 11: FastAPI server for the winning variant.

Winning variant: combined INT4 (bnb NF4, top-5 sensitive layers FP16)
plus FastV pruning keep 35%, no speculative decoding.
Speculative was excluded because Day 8-9 measured it 67-72% slower
with acceptance ~0.01-0.02, and Day 9 merged draft also failed.

Endpoints:
  GET  /health   -> {ok, model, variant, loaded}
  POST /generate {question, image_path} -> {text, latency_s, tokens, variant}
    image_path is a path on the server (pod), e.g. data/sample.jpg.
    For a local test, upload is not implemented; use a server-side
    path to keep the server simple and reproducible.
  POST /generate_upload (multipart file + question) -> same, for
    real clients that send an image file.

Model is loaded once at startup. Generation uses the manual
pruning loop from src.pruning.eval_pruned, not model.generate,
because pruning is implemented there.
Run: uvicorn src.serving.app:app --port 8000 --host 0.0.0.0
"""
import time, io, threading
import torch
from fastapi import FastAPI, UploadFile, File, Form
from PIL import Image

app = FastAPI(title="SpecQuant-VLM")
_lock = threading.Lock()
_model = None
_processor = None
_loaded = False
_variant = "combined_int4_pruning_keep035"

def _load():
    global _model, _processor, _loaded
    if _loaded:
        return
    from src.quantization.eval_quantized import load_quantized, restore_fp16_layers
    _model, _processor = load_quantized("llava-hf/llava-1.5-7b-hf", 4)
    _model = restore_fp16_layers(_model, "llava-hf/llava-1.5-7b-hf", [31, 1, 16, 10, 7])
    _loaded = True

@app.on_event("startup")
def startup():
    _load()

@app.get("/health")
def health():
    return {"ok": True, "model": "llava-hf/llava-1.5-7b-hf", "variant": _variant, "loaded": _loaded}

def _run(question, image):
    from src.common.load_model import make_prompt
    from src.pruning.eval_pruned import generate_pruned
    _load()
    prompt = make_prompt(_processor, question)
    inputs = _processor(text=prompt, images=image, return_tensors="pt").to(_model.device)
    t0 = time.time()
    with _lock:
        with torch.no_grad():
            gen_ids = generate_pruned(_model, _processor, inputs["input_ids"], inputs["pixel_values"], keep_ratio=0.35)
    secs = time.time() - t0
    text = _processor.tokenizer.decode(gen_ids, skip_special_tokens=True)
    return {"text": text, "latency_s": round(secs, 3), "tokens": len(gen_ids), "variant": _variant}

@app.post("/generate")
def generate(payload: dict):
    question = payload.get("question", "Describe this image in one sentence.")
    image_path = payload.get("image_path", "data/sample.jpg")
    image = Image.open(image_path).convert("RGB")
    return _run(question, image)

@app.post("/generate_upload")
async def generate_upload(file: UploadFile = File(...), question: str = Form("Describe this image in one sentence.")):
    data = await file.read()
    image = Image.open(io.BytesIO(data)).convert("RGB")
    return _run(question, image)
