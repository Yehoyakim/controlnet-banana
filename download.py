import os
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel

def download_model():
    # Download model on first launch
    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-depth", torch_dtype=torch.float16)
    midas_model = torch.hub.load("intel-isl/MiDaS", "MiDaS")

if __name__ == "__main__":
    download_model()
