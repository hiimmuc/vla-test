# Install minimal dependencies (`torch`, `transformers`, `timm`, `tokenizers`, ...)
# conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=12.1 -c pytorch -c nvidia
# > pip install -r https://raw.githubusercontent.com/openvla/openvla/main/requirements-min.txt
# pip install numpy==1.26.4
# pip install accelerate
# pip install "flash-attn==2.5.5" --no-build-isolation

import warnings

import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

warnings.filterwarnings("ignore", category=FutureWarning)

# Load Processor & VLA
processor = AutoProcessor.from_pretrained("openvla/openvla-v01-7b", trust_remote_code=True)
vla = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-v01-7b",
    attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
).to("cuda:0")

# Grab image input & format prompt
image: Image.Image = Image.open("sample.png").convert("RGB")
INSTRUCTION = "cook the egg"
prompt = f"In: What action should the robot take to {INSTRUCTION}?\nOut:"

# Predict Action (7-DoF; un-normalize for BridgeData V2)
inputs = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)
action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
print(f"Predicted Action: {action}")
# Execute...
# robot.act(action, ...)
