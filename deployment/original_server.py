import io

import torch
from diffusers import DiffusionPipeline
from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel

from src.models.training import load_and_set_lora_ckpt

base = DiffusionPipeline.from_pretrained("segmind/tiny-sd", torch_dtype=torch.float32)
model = load_and_set_lora_ckpt(base, 300)


class ModelInput(BaseModel):
    prompt: str


app = FastAPI(redirect_slashes=False)


@app.post("/generate")
async def generate_image(input_text: ModelInput):
    generated_image = model(input_text.prompt).images[0]
    buf = io.BytesIO()
    generated_image.save(buf, format="PNG")
    buf.seek(0)
    return Response(content=buf.read(), media_type="image/png")
