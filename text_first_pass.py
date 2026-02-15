
# DIGITAL CLONE

# %%

import json

# %%

CHAT_TYPE = 'personal_chat'

with open('data/result.json', 'r', encoding = 'utf-8') as f:
    data : dict = json.load(f)['chats']['list']

data = [i for i in data if i['type'] == CHAT_TYPE] # no saved messages
data = [i for i in data if i['name'] is not None] # no spam

# %%

result = []

for chat in data:
    chat = chat['messages']

    last_id = None
    last_time = float('inf')
    conv = []

    for msg in chat:
        time = int(msg['date_unixtime'])
        delta_time = time - last_time
        last_time = time # seconds

        if delta_time > 3600 * 12 and len(conv) > 1:
            result.append({ 'conversations' : conv })

            conv = []
            last_id = None
            last_time = float('inf')

        text = ''.join((i if type(i) is str else i['text']) for i in msg['text'])
        if 'file' in msg:
            if msg['file'] == '(File not included. Change data exporting settings to download.)':
                continue
            text += '<|FILE|>' + msg['file'] + '<|FILE_END|>'

        if 'from_id' not in msg and 'action' in msg:
            msg['from_id'] = msg['actor_id']
            text += '(' + str.upper(msg['action']) + ')'
        
        if text == '':
            continue

        if 'from_id' not in msg:
            print(msg)

        if msg['from_id'] == last_id: # several messages
            conv[-1]['content'] += '<|NEW|>' + text
        else:
            if msg['from_id'] == 'user1637916753': # me
                conv.append({ 'role' : 'assistant', 'content' : text })
            else: # not me
                conv.append({ 'role' : 'user', 'content' : text })

        last_id = msg['from_id']

    if len(conv) > 1:
        result.append({ 'conversations' : conv })

# %%

result

# %%

with open('data_with_files.json', 'w', encoding = 'utf-8') as f:
    json.dump(result, f, ensure_ascii = False)

# %%

len(result)

# %%

'''
result = 
[
  {
    "conversations": [
      {"role": "user", "content": "Привет<|NEW|><|FILE|>chats/chat0/jaba.webp<|FILE_END|><|NEW|>зацени стикер"},
      {"role": "assistant", "content": "Ого\nкакой крутой"},
      {"role": "user", "content": "Ага"},
      {"role": "assistant", "content": "[MEDIA: жаба в новогодней шапке]"}
    ]
  },
  {
    "conversations": [
      {"role": "user", "content": "Спишь?"},
      {"role": "assistant", "content": "Не"}
    ]
  }
]
'''

# %%
