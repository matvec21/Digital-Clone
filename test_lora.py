import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# --- КОНФИГУРАЦИЯ ---
adapter_path = "./digital_clone" 
base_model_name = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"
MAX_HISTORY = 10  # Сколько сообщений помнить (5 твоих + 5 бота)

print("Загрузка модели...")
tokenizer = AutoTokenizer.from_pretrained(adapter_path)
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=BitsAndBytesConfig(load_in_4bit=True),
    device_map="auto",
    trust_remote_code=True
)
model.resize_token_embeddings(151666) # Твой фиксированный размер словаря
model = PeftModel.from_pretrained(model, adapter_path)
model.eval()

# Список для хранения истории
# Добавляем системный промпт для стиля
chat_history = [
    {"role": "system", "content": "Ты общаешься в личной переписке. Пиши как живой человек, используй сленг, не стесняйся дробить сообщения через <|NEW|>, если мыслей много."}
]

def generate_answer(history):
    prompt = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    new_token_id = tokenizer.convert_tokens_to_ids("<|NEW|>")
    skobka = tokenizer.convert_tokens_to_ids(")")
    
    # Список запрещенных "технических" токенов, которые ты видел
    # Добавим туда типичные теги Qwen, которые могут вылезать
    bad_words = ["<tool_call>", "</tool_call>", "<|plugin|>", "tool_call", "spNet", "assistant\n"]
    bad_words_ids = [tokenizer.encode(word, add_special_tokens=False) for word in bad_words]
    eos_token_id = tokenizer.eos_token_id  # ID токена завершения

    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=128,
            temperature=0.4,      # Снижаем с 0.9 до 0.7 для стабильности
            min_p=0.1,             # ОГРАНИЧИВАЕМ выбор топ-40 словами (уберет иероглифы)
            repetition_penalty=1.15,
            
            # Снижаем бонус до 2.0 - 4.0. Этого хватит, чтобы он был, но не ломал логику.
            sequence_bias={tuple([new_token_id]): 1.5, tuple([skobka]): -0.5, tuple([eos_token_id]): -0.5} if new_token_id else None,
            
            bad_words_ids=bad_words_ids, # Запрещаем мусор
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Отрезаем промпт и декодируем
    output_tokens = outputs[0][inputs.input_ids.shape[-1]:]
    decoded = tokenizer.decode(output_tokens, skip_special_tokens=True)
    
    # Финальная чистка от иероглифов (на всякий случай)
    # Если в ответе больше 20% иероглифов — это бред, можно просто почистить
    return decoded.strip()

print("\n--- КЛОН ЗАПУЩЕН ---")
print("(Пиши сообщения. 'exit' для выхода, 'clear' чтобы бот всё забыл)\n")

while True:
    user_input = input("Ты: ")
    
    if user_input.lower() == 'exit': break
    if user_input.lower() == 'clear':
#        chat_history = chat_history[:1] # Оставляем только системный промпт
        chat_history = []
        print("Бот: (память очищена)")
        continue

    # 1. Добавляем твое сообщение в историю
    chat_history.append({"role": "user", "content": user_input})
    
    # 2. Генерируем ответ
    reply = generate_answer(chat_history)
    
    # 3. Добавляем ответ бота в историю (ВАЖНО для контекста!)
    chat_history.append({"role": "assistant", "content": reply})
    
    # 4. Ограничиваем длину истории, чтобы не переполнить память
    if len(chat_history) > MAX_HISTORY + 1:
        chat_history = [chat_history[0]] + chat_history[-(MAX_HISTORY):]

    # 5. Красивый вывод с разделением баблов
    if "<|NEW|>" in reply:
        bubbles = reply.split("<|NEW|>")
        for b in bubbles:
            if b.strip():
                print(f"Клон: {b.strip()}")
    else:
        print(f"Клон: {reply}")
