import torch
from diffusers.pipelines.wuerstchen.pipeline_wuerstchen_prior import  WuerstchenPriorPipeline
from diffusers.pipelines.wuerstchen import DEFAULT_STAGE_C_TIMESTEPS
import matplotlib.pyplot as plt


def main():
    pipeline = WuerstchenPriorPipeline.from_pretrained("wuerstchen-prior-naruto-model/prior_pipeline", torch_dtype=torch.float16).to("cuda")

    # caption = "A cute bird naruto holding a shield"
    # images = pipeline(
    #     caption,
    #     width=1024,
    #     height=1536,
    #     prior_timesteps=DEFAULT_STAGE_C_TIMESTEPS,
    #     prior_guidance_scale=4.0,
    #     num_images_per_prompt=2,
    # ).images
    # plt.imshow(images[0])
    # plt.show()


if __name__ == "__main__":
    main()
