import io

import torch
from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel

from diffusers import DiffusionPipeline

model = DiffusionPipeline.from_pretrained("segmind/tiny-sd", torch_dtype=torch.bfloat16)
model.load_lora_weights(
    "AML-group10/5e-4_30hyperparameter_tuning", weight_name="300_lora.pt"
)


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
