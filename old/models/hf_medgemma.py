# %% cell 1
# ! uv pip install --upgrade accelerate bitsandbytes transformers

# %% cell 2
from transformers import BitsAndBytesConfig
import torch
from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env.

model_variant = "4b-it"  # @param ["4b-it", "27b-it", "27b-text-it"]
model_id = f"google/medgemma-{model_variant}"

use_quantization = True  # @param {type: "boolean"}

# @markdown Set `is_thinking` to `True` to turn on thinking mode. **Note:** Thinking is supported for the 27B variants only.
is_thinking = False  # @param {type: "boolean"}

model_kwargs = dict(
    dtype=torch.bfloat16,
    device_map="auto"
)

if use_quantization:
    model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True) 
    

# %% cell 3
from transformers import pipeline
print("Loading model... This may take a while.")
if "text" in model_variant:
    print("Text-only model detected.")
    pipe = pipeline("text-generation", model=model_id, model_kwargs=model_kwargs,token="hf_your_token_here")
else:
    print("Image-text model detected.")
    pipe = pipeline("image-text-to-text", model=model_id, model_kwargs=model_kwargs,token="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")
print("Model pipeline created.")
pipe.model.generation_config.do_sample = False  # pyright: ignore

print(f"Model loaded: {model_id}")

# %% cell 4
import os
# from PIL import Image
from IPython.display import Image as IPImage, display, Markdown

prompt = "Describe X-ray"  # @param {type: "string"}

# Image attribution: Stillwaterising, CC0, via Wikimedia Commons
# image_url = "https://upload.wikimedia.org/wikipedia/commons/c/c8/Chest_Xray_PA_3-8-2010.png"  # @param {type: "string"}
# ! wget -nc -q {image_url}
# image_filename = os.path.basename(image_url)
# image = Image.open(image_filename)


# %% cell 5
role_instruction = "You are an expert radiologist."
if "27b" in model_variant and is_thinking:
    system_instruction = f"SYSTEM INSTRUCTION: think silently if needed. {role_instruction}"
    max_new_tokens = 1300
else:
    system_instruction = role_instruction
    max_new_tokens = 300

messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": system_instruction}]
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            # {"type": "image", "image": image}
        ]
    }
]
    
    
# %% cell 6  
output = pipe(text=messages, max_new_tokens=max_new_tokens)
response = output[0]["generated_text"][-1]["content"]

display(Markdown(f"---\n\n**[ User ]**\n\n{prompt}"))
# display(IPImage(filename=image_filename, height=300))
if "27b" in model_variant and is_thinking:
    thought, response = response.split("")
    thought = thought.replace("thought\n", "")
    display(Markdown(f"---\n\n**[ MedGemma thinking ]**\n\n{thought}"))
display(Markdown(f"---\n\n**[ MedGemma ]**\n\n{response}\n\n---"))

# %%
