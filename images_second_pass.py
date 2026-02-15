import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Чистим память перед загрузкой
import gc
gc.collect()
torch.cuda.empty_cache()

model_id = "vikhyatk/moondream2"
revision = "2024-08-26"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Загружаем токенизатор
tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

# Загружаем модель (с этой версией transformers ошибка pad_token_id должна исчезнуть)
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    trust_remote_code=True, 
    revision=revision,
    torch_dtype=torch.float16,
    device_map={"": device}
)

model.eval()
print("Ура! Модель Moondream2 загружена без ошибок.")

from PIL import Image
import os
import json
from tqdm import tqdm

from pathlib import Path

image_folder = "./images" # Твоя папка с картинками
media_db = {}

# Список поддерживаемых форматов
valid_extensions = ('.png', '.jpg', '.jpeg', '.webp')
files = [f for f in os.listdir(image_folder) if f.lower().endswith(valid_extensions)]

print(f"Начинаю описывать {len(files)} изображений...")

source = Path(image_folder)

for file in tqdm(source.rglob("*")):
    if not file.is_file():
        continue

    filename = str(file)

    try:
        image = Image.open(filename).convert("RGB")
        with torch.no_grad():
            image_embeds = model.encode_image(image)
            
            # Используем более специфичный промпт
            question = "Describe the specific action, emotion, and social tone of this image in one short sentence. Be direct and avoid generic terms like 'gesture' if the action is specific."
            
            # Если ты используешь стандартный метод модели:
            description = model.answer_question(
                image_embeds, 
                question, 
                tokenizer,
                max_new_tokens=30, # Ограничиваем длину, но даем пространство
            )
            
            # Если ответ все равно слишком короткий (1-2 слова), 
            # можно попробовать добавить "Give a detailed description:"
            if len(description.split()) < 3:
                description = model.answer_question(image_embeds, "Detailed description of this image:", tokenizer)

        media_db[str(file.relative_to(source).with_suffix(''))] = description.strip()
        
    except Exception as e:
        print(f"\nОшибка с файлом {filename}: {e}")

# Сохраняем результат в JSON
with open("media_db.json", "w", encoding="utf-8") as f:
    json.dump(media_db, f, ensure_ascii=False, indent=2)

print(f"\nГотово! Описано файлов: {len(media_db)}")

# !pip install transformers==4.44.2 einops timm --force-reinstall
