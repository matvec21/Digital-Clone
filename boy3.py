# %%

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer, util
from peft import PeftModel

from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

import telebot
from telebot import types

import threading
import time

import json
import re
import os

import cv2

from lottie.importers.core import import_tgs
from lottie.exporters.cairo import export_png

import queue
import threading

if not os.path.exists('temp'):
    os.makedirs('temp')

# %%

media_embeddings = torch.load('media_embeddings.pt')
model_text2vec = SentenceTransformer('all-MiniLM-L6-v2')

with open('media_db.json', 'r', encoding = 'utf-8') as f:
    media_path : list = list(json.load(f).keys())

caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to("cpu")

# %%

adapter_path = './digital_clone' 
base_model_name = 'unsloth/Qwen2.5-3B-Instruct-bnb-4bit'
MAX_HISTORY = 10

print('Загрузка модели...')
tokenizer = AutoTokenizer.from_pretrained(adapter_path)
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config = BitsAndBytesConfig(load_in_4bit=True),
    device_map = 'auto',
    trust_remote_code = True
)
model.resize_token_embeddings(151666)
model = PeftModel.from_pretrained(model, adapter_path)
model.eval()

SYSTEM_PROMPT = {
    'role': 'system', 
    'content': (
        'Ты — цифровой клон в личной переписке. Пиши как живой человек. Используй сленг. '
        'Теги [MEDIA: description] — это фото, то, что ты видишь сейчас, ты ОБЯЗАН использовать это описание как факт, frog это жаба. '
    )
}

# %%

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

def generate_answer(history):
    prompt = tokenizer.apply_chat_template(history, tokenize = False, add_generation_prompt = True)
    inputs = tokenizer(prompt, return_tensors = 'pt').to(model.device)

    new_token_id = tokenizer.convert_tokens_to_ids('<|NEW|>')
    skobka = tokenizer.convert_tokens_to_ids(')')
    square = tokenizer.convert_tokens_to_ids('[')

    bad_words = ['<tool_call>', '</tool_call>', '<|plugin|>', 'tool_call', 'spNet', 'assistant\n']
    bad_words_ids = [tokenizer.encode(word, add_special_tokens=False) for word in bad_words]
    eos_token_id = tokenizer.eos_token_id

    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens = 128,
            min_new_tokens = 5,
            temperature = 0.5,
            min_p = 0.2,
            repetition_penalty = 1.05,

#           sequence_bias = {tuple([new_token_id]): 1.5, tuple([skobka]): -0.5, tuple([eos_token_id]): -0.5, tuple([square]): 4.0},
#           sequence_bias = {tuple([new_token_id]): 1.5, tuple([eos_token_id]): -0.2, tuple([square]): 4.0},
            sequence_bias = {tuple([square]): 3.0},

            bad_words_ids = bad_words_ids,
            pad_token_id = tokenizer.eos_token_id,
            eos_token_id = tokenizer.eos_token_id
        )
    
    output_tokens = outputs[0][inputs.input_ids.shape[-1]:]
    decoded = tokenizer.decode(output_tokens, skip_special_tokens=True)

    return decoded.strip()

# %%

def get_closest_image(description : str):
    embeddings = model_text2vec.encode(description, convert_to_tensor = True)
    hits = util.semantic_search(embeddings, media_embeddings, top_k = 1)
    path = 'data/' + media_path[hits[0][0]['corpus_id']]
    filename = path.split('/')[-1]
    path = path[:-len(filename)]
    files = os.listdir(path)
    files = [i for i in files if filename in i]
    if not files:
        raise BaseException(f'No such file as {path + filename}')
    filename = files[0]
    print('Found image', path + filename)
    return path + filename

# %%

def tgs_to_png(tgs_path, output_path):
    try:
        animation = import_tgs(tgs_path)
        export_png(animation, output_path, frame = 0)
    except:
        print('error')

def video_to_png(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
    
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(output_path, frame)
    
    cap.release()
    return ret

# %%

user_memories = {}

def get_user_history(chat_id):
    if chat_id not in user_memories:
        user_memories[chat_id] = [SYSTEM_PROMPT]
    return user_memories[chat_id]

# %%

task_queue = queue.Queue()

def bot_worker():
    while True:
        message, text_to_process = task_queue.get()
        try:
            me_answer(message, text_to_process)
        except Exception as e:
            print(f"Ошибка в воркере: {e}")
        finally:
            task_queue.task_done()

threading.Thread(target = bot_worker, daemon = True).start()

# %%

def send_typing_action(chat_id, stop_event):
    """Функция, которая будет крутить статус 'typing', пока модель думает"""
    while not stop_event.is_set():
        bot.send_chat_action(chat_id, 'typing')
        time.sleep(4)

from dotenv import load_dotenv
load_dotenv()
TOKEN = os.getenv('API_TOKEN')
bot = telebot.TeleBot(TOKEN)

@bot.message_handler(commands = ['start'])
def onstart(message):
    bot.send_message(message.chat.id, text = 'Я мы - matvec21. Все работает.\nНапиши "clear", чтобы очистить мне память, как будто мы не общались сутки'.format(message.from_user))

def me_answer(message : types.Message, text : str):
    stop_typing = threading.Event()
    typing_thread = threading.Thread(target = send_typing_action, args = (message.chat.id, stop_typing))
    typing_thread.start()

    try:
        chat_history = get_user_history(message.chat.id)
        chat_history.append({'role': 'user', 'content': text})
        reply = generate_answer(chat_history)
        chat_history.append({'role': 'assistant', 'content': reply})

        if len(chat_history) > MAX_HISTORY:
            user_memories[message.chat.id] = [chat_history[0]] + chat_history[-(MAX_HISTORY-1):]
    finally:
        stop_typing.set()
        typing_thread.join()

    for part in reply.split('<|NEW|>'):
        print('Processing', repr(part))
        part = part.replace('[IMAGE:', '[MEDIA:')
        media_match = re.search(r'\[MEDIA:\s*(.*?)\]', part)

        if media_match:
            description = media_match.group(1)
            clean_text = re.sub(r'\[MEDIA:.*?\]', '', part).strip()

            try:
                image_path = get_closest_image(description)
                if clean_text:
                    bot.send_message(message.chat.id, text = clean_text)
                with open(image_path, 'rb') as sticker:
                    bot.send_sticker(message.chat.id, sticker)
            except Exception as e:
                print(f'Ошибка при отправке фото: {e}')
                bot.send_message(message.chat.id, text = part)
        else:
            if part.strip():
                bot.send_message(message.chat.id, text = part)

    print(message.from_user.id, message.from_user.full_name, '|', text, '|', repr(reply))

    with open('users', 'a', encoding = 'utf-8') as users:
        print(str(message.from_user.id) + ' ||||| ' + message.from_user.full_name + ' ||||| ' + text + ' ||||| ' + reply, file = users)

@bot.message_handler(content_types = ['text'])
def ontext(message : types.Message):
    text = message.text
    print(message.from_user.full_name, 'sent:', text)

    if text.lower() == 'clear':
        user_memories[message.chat.id] = [SYSTEM_PROMPT]
        bot.send_message(message.chat.id, text = 'Прошли сутки...')
        return

    task_queue.put((message, text))

@bot.message_handler(content_types = ['photo', 'sticker'])
def on_photo(message: types.Message):
    bot.send_chat_action(message.chat.id, 'typing')
    print('Залетел', message.content_type)

    temp_input = f'temp/raw_{message.chat.id}'
    temp_png = f'temp/proc_{message.chat.id}.png'

    try:
        if message.content_type == 'photo':
            file_id = message.photo[-1].file_id
            suffix = '.jpg'
        else:
            file_id = message.sticker.file_id
            if message.sticker.is_animated: suffix = '.tgs'
            elif message.sticker.is_video: suffix = '.webm'
            else: suffix = '.webp'

        file_info = bot.get_file(file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        
        input_path = temp_input + suffix
        with open(input_path, 'wb') as f:
            f.write(downloaded_file)

        if suffix == '.tgs':
            tgs_to_png(input_path, temp_png)
        elif suffix == '.webm' or suffix == '.mp4':
            video_to_png(input_path, temp_png)
        else:
            img_ready = prepare_image(input_path)
            img_ready.save(temp_png)

        if not os.path.exists(temp_png):
            raise BaseException('Не вышло скачать')

        description = describe_image(temp_png)
        print(f'User {message.from_user.full_name} sent a photo. BLIP: {description}')

        text_with_photo = f'[MEDIA: {description}]'
        if message.caption:
            text_with_photo = message.caption + ' ' + text_with_photo

        task_queue.put((message, text_with_photo))

    except Exception as e:
        print(f'Ошибка при обработке фото: {e}')
        bot.reply_to(message, 'Ты меня сломал')

    finally:
        for p in [temp_png, temp_input + '.tgs', temp_input + '.webm', temp_input + '.webp', temp_input + '.jpg']:
            if os.path.exists(p): os.remove(p)

bot.infinity_polling(timeout = 10, long_polling_timeout = 5)

# %%
