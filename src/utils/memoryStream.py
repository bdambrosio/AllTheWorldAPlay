from operator import attrgetter
from datetime import datetime
import pickle, math, os
import numpy as np
from pathlib import Path
import faiss
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import utils.pyqt as pyqt_utils

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
        ok = pyqt_utils.confirmation_popup("Really? Not recoverable!", '   ')
        if not ok:
            return
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
            

    def remember(self, text, salience=0):
        """ add a memory to the memory stream
            note that memories are synopses, not full faith memories of long items
        """
        memory_sections = {"Synopsis":{"instruction":'A concise summary.'},
                           }
        if len(text)> 240:
            synopsis = text[:240].strip()
        else:
            synopsis = text.strip()
        id = self.generate_new_key()
        # synopsis has only one section for now, but may add ltr. Should pbly add headings
        text_embedding = self.embedding_request('\n'.join(synopsis), 'search_document: ')
        self._stream[id] = self.Memory(synopsis, salience)
        ids_np = np.array([id], dtype=np.int64)
        embeds_np = np.array([text_embedding], dtype=np.float32)
        self.memory_indexIDMap.add_with_ids(embeds_np, ids_np)
        self.unsaved_memories += 1

        if self.unsaved_memories > 10:
            self.save()
            self.unsaved_memories = 0

    def recall(self, thought, recent=8, top_k=8):
        text_embedding = self.embedding_request(thought, 'query_document: ')
        embeds_np = np.array([text_embedding], dtype=np.float32)
        scores, ids = self.memory_indexIDMap.search(embeds_np, 20) # get plenty of candidates!
        if len(ids) < 1:
            print(f'No items found')
            return [],[]
        age_normalizer = math.log(600.0+(datetime.now() - self.earliest_memory_date).total_seconds())
        rescored_tuples = []
        recent = self.n_most_recent(recent)
        for id, score in zip(ids[0], scores[0]):
            if id < 0:
                break
            memory = self._stream[id]
            if memory in recent: # we're going to include the top n anyway, skip those
                continue
            #age and recency are normalized to total age of memory stream
            age = math.log((datetime.now() - memory.created).total_seconds()+600.0) /age_normalizer 
            recency = math.log((datetime.now() - memory.accessed).total_seconds()+600.0) /age_normalizer
            overall = age*0.05+recency*0.05+score
            #print(id, score, age, recency, overall)
            rescored_tuples .append((overall, id))
        #sort by score and pick top 8
        scored_memory_tuples = sorted(rescored_tuples, key=lambda x: x[0])[:8]
        scored_memories = [self._stream[m_tuple[1]] for m_tuple in scored_memory_tuples]
        #now resort by date
        recalled = sorted(scored_memories, key=lambda m: m._created)

        final = recalled+recent
        #print (f'\n{self.host_name} recall returning:')
        #for memory in final:
        #    print(f'{memory._created} {memory._text}')

        return final
        
if __name__ == '__main__':
    ms = MemoryStream('xxx')
    ms.remember('xxx thinks I wonder what time it is?')
    ms.remember('xxx wonders I wonder where I am?')
    ms.remember('xxx says who are you ?')
    print(ms.recall('wonder'))
    print(ms.recall('time'))
