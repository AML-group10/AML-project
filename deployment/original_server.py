import io

import torch
from diffusers import DiffusionPipeline
from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel

from archive.inference import load_and_set_lora_ckpt

base = DiffusionPipeline.from_pretrained("segmind/tiny-sd", torch_dtype=torch.float32)
model = load_and_set_lora_ckpt(base, "AML-group10/5e-4_30hyperparameter_tuning", 300)


class ModelInput(BaseModel):
    prompt: str


app = FastAPI(redirect_slashes=False)


@app.get("/health")
async def health():
    return Response(content="yo mama", media_type="txt")


@app.post("/generate")
async def generate_image(input_text: ModelInput):
    print(input_text)
    print(input_text.prompt)
    generated_image = model(input_text.prompt, num_inference_steps=30).images[0]
    buf = io.BytesIO()
    generated_image.save(buf, format="PNG")
    buf.seek(0)
    return Response(content=buf.read(), media_type="image/png")
