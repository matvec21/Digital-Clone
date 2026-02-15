
# DIGITAL CLONE

# %%

from sentence_transformers import SentenceTransformer, util
import json

# %%

with open('media_db.json', 'r', encoding = 'utf-8') as f:
    data : dict = json.load(f)

data

# %%

model = SentenceTransformer('all-MiniLM-L6-v2')

sentences = list(data.values())

media_embeddings = model.encode(sentences, convert_to_tensor = True)

# %%

query = "A close-up image of a frog's face, with wide-open eyes and a slightly raised nose, conveys curiosity and alertness."
query_embedding = model.encode(query, convert_to_tensor = True)

hits = util.semantic_search(query_embedding, media_embeddings, top_k = 1)
print(sentences[hits[0][0]['corpus_id']])

# %%

import torch

torch.save(media_embeddings, 'media_embeddings.pt')

# %%
