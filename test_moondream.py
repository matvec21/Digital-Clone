
# %%

from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to("cpu")

def prepare_image(image_path):
    img = Image.open(image_path).convert("RGBA")
    canvas = Image.new("RGBA", img.size, (255, 255, 255, 255))
    canvas.paste(img, mask=img)
    return canvas.convert("RGB")

def describe_image(image_path):
    raw_image = prepare_image(image_path)
    inputs = caption_processor(raw_image, return_tensors="pt").to("cpu")
    out = caption_model.generate(**inputs)
    return caption_processor.decode(out[0], skip_special_tokens=True)

# %%

print(describe_image('my_media/sticker (10).png'))

# %%
