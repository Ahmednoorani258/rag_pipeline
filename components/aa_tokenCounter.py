import tiktoken

def num_tokens_from_string(string:str,encodingname:str) -> int:
    encoding = tiktoken.get_encoding(encodingname)
    num_tokens = len(encoding.encode(string))
    return num_tokens 

num_tokens = num_tokens_from_string("Hello world","cl100k_base")
print(num_tokens)

import numpy as np

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    # if norm_vec1 == 0 or norm_vec2 == 0:
    #     return 0.0
    return dot_product / (norm_vec1 * norm_vec2)
    