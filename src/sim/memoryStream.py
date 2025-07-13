from operator import attrgetter
from datetime import datetime
import pickle, math, os
import numpy as np
from pathlib import Path
import faiss
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

embedding_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
embedding_model = AutoModel.from_pretrained('nomic-ai/nomic-embed-text-v1', trust_remote_code=True)
embedding_model.eval()
MEMORY_DIR = Path.home() / '.local/share/AllTheWorld/memories/'
MEMORY_DIR.mkdir(parents=True, exist_ok=True)

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class MemoryStream():
    def __init__(self, name, cot):
        self.host_name = name
        filename = name + '_MemoryStream.pkl'
        faiss_filename = name + '_MemoryStream.faiss'
        self.filename=MEMORY_DIR / filename
        self.faiss_filename=MEMORY_DIR / faiss_filename
        self._stream = {}
        self.memory_indexIDMap=None # set by load
        self.earliest_memory_date = None # set by load
        self.load()
        if len(self._stream)> 0 and not all(isinstance(key, int) for key in self._stream):
            raise ValueError("Not all keys are integers.")
        elif len(self._stream) > 0:
            self.last_id = max(self._stream.keys())
        self.unsaved_memories = 0

    class Memory():
        def __init__(self, text, salience):
            self._created = datetime.now()
            self._accessed = datetime.now()
            self._text = text
            self._salience = salience # not currently used

        @property
        def text(self):
            self._accessed = datetime.now()
            return self._text
        @property
        def created(self):
            return self._created
        @property
        def accessed(self):
            return self._accessed
        @property
        def salience(self):
            return self._salience
        def age(self):
            return int((datetime.now() - self._created).total_seconds()/60)

    @property
    def memories(self):
        return self._stream
    
    def n_most_recent(self, n=8):
        # Get a list of all memory items from the dictionary
        if n <= 0:
            return []
        memory_items = list(self._stream.values())
        # Sort the memory items based on the 'created' field in ascending order
        sorted_items = sorted(memory_items, key=attrgetter('created'))
        # Return the four most recently created items
        return sorted_items[-n:]
    
    def embedding_request(self, text, request_type='search_document: '):
        global embedding_tokenizer, embedding_model
        if type(text) != str:
            print(f'\nError - type of embedding text is not str {text}')
        text_batch = [request_type+' '+text]
        # preprocess the input
        encoded_input = embedding_tokenizer(text_batch, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = embedding_model(**encoded_input)
        embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        #print(f'embedding_request response shape{embeddings.shape}')
        
        embedding = embeddings[0]
        # print(f'embedding_request response[0] shape{embedding.shape}')
        return embedding.detach().numpy()

    def save(self):
        with open(self.filename, 'wb') as file:
            pickle.dump(self._stream, file)
        faiss.write_index(self.memory_indexIDMap, str(self.faiss_filename))

    def load(self):
        try:
            with open(self.filename, 'rb') as file:
                self._stream = pickle.load(file)
        except FileNotFoundError:
            self._stream = {}
            with open(self.filename, 'wb') as file:
                pickle.dump(self._stream, file)
        try:
            self.memory_indexIDMap = faiss.read_index(str(self.faiss_filename))
            # now set earliest created date for normalization of created dates in retrieval
            if len(self._stream)> 0 and all(isinstance(key, int) for key in self._stream):
                earliest_memory_id = min(self._stream.keys())
                self.earliest_memory_date = self._stream[earliest_memory_id].created
            else:
                self.earliest_memory_date = datetime.now()

            print(f"loaded {self.faiss_filename} items {len(self._stream)}")
        except Exception:
            self.memory_indexIDMap = faiss.IndexIDMap(faiss.IndexFlatL2(768))
            faiss.write_index(self.memory_indexIDMap, str(self.faiss_filename))
            self.earliest_memory_date = datetime.now()
            print(f"created {self.faiss_filename}")

    def clear(self):
        self._stream = {}
        with open(self.filename, 'wb') as file:
            pickle.dump(self._stream, file)
        # now set earliest created date for normalization of created dates in retrieval
        self.memory_indexIDMap = faiss.IndexIDMap(faiss.IndexFlatL2(768))
        faiss.write_index(self.memory_indexIDMap, str(self.faiss_filename))
        self.earliest_memory_date = datetime.now()
        print(f"recreated {self.filename} and {self.faiss_filename}")

    def generate_new_key(self):
        if len(self._stream)> 0 and all(isinstance(key, int) for key in self._stream):
            return int(max(self._stream.keys())+1)
        elif len(self._stream) ==0:
            return int(1)
        else:
            raise ValueError("Not all keys are integers.")
            
