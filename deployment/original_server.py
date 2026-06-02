import io

import torch
from diffusers import DiffusionPipeline
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from src.models.training.inference_loop import load_and_set_lora_ckpt

base = DiffusionPipeline.from_pretrained("segmind/tiny-sd", torch_dtype=torch.float32)
model = load_and_set_lora_ckpt(base, "AML-group10/5e-4_20hyperparameter_tuning", 200)


class ModelInput(BaseModel):
    prompt: str


app = FastAPI(
    redirect_slashes=False, description="This app generates images from text."
)


@app.get("/health")
async def health():
    """This function returns a message to confirm that the API works.

    Returns:
        Response: The confirmation message.
    """
    return Response(content="This works.", media_type="txt")


@app.post("/generate")
async def generate_image(input_text: ModelInput):
    """Generates an image from a prompt, using a fine-tuned
    stable diffusion model, using 30 inference steps.

    Args:
        input_text (ModelInput): The prompt from which the image is generated.

    Raises:
        HTTPException 404: Prompt not found.
        HTTPException 500: Text to image generation failed.

    Returns:
        Response: The generated PNG image.
    """
    if not input_text.prompt:
        raise HTTPException(status_code=404, detail="Prompt not found.")

    try:
        generated_image = model(input_text.prompt, num_inference_steps=30).images[0]
        buf = io.BytesIO()
        generated_image.save(buf, format="PNG")
        buf.seek(0)
        return Response(content=buf.read(), media_type="image/png", status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Text to image genration failed.")
