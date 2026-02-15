# DIGITAL CLONE

# %%

import json
import re

# %%

with open('data_with_files.json', 'r', encoding = 'utf-8') as f:
    chats = json.load(f)

with open('media_db.json', 'r', encoding = 'utf-8') as f:
    media_db = json.load(f)

# %%

def replace_files(text):
    def sub_func(match):
        full_path = match.group(1)
        if full_path == '(File not included. Change data exporting settings to download.)':
            raise 1
            return '[MEDIA: Unknown]'

        path = '.'.join(full_path.split('.')[:-1])
        description = media_db.get(path)

        if description:
            return f'[MEDIA: {description}]'
        else:
            print(path)
            return '[MEDIA: Изображение]'

    pattern = r"<\|FILE\|>\s*(.*?)\s*<\|FILE_END\|>"
    return re.sub(pattern, sub_func, text)

for session in chats:
    for msg in session['conversations']:
        msg['content'] = replace_files(msg['content'])

with open('data_with_meanings.json', 'w', encoding = 'utf-8') as f:
    json.dump(chats, f, ensure_ascii = False, indent = 2)

# %%
