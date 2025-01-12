import os
import gc
import pandas as pd
import numpy as np
import re
import math
import time
import traceback
from collections.abc import Iterable
import torch
import faiss
from faiss import write_index, read_index
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from torch.utils.data import DataLoader
from typing import Optional, Union
import warnings
warnings.filterwarnings("ignore")


class OpenBook():
    def __init__(self,top_k_articles:int=3,
                 top_k_matches:int=3,
                 model_name:str="BAAI/bge-small-en", 
                 #model_name:str="BAAI/bge-base-en-v1.5", 
                 device:str='cuda',
                 max_length:int=512, 
                 batch_size:int=16,
                 wikipedia_data_path:str="/home/bruce/Downloads/enwiki/en_wiki_202212",
                 wikipedia_index_path:str="/home/bruce/Downloads/enwiki/en_wiki_202212/index.parquet",
                 wikipedia_faiss_index_path:str="/home/bruce/Downloads/enwiki/en_wiki_202212_ivf256_sq8.index",
                 nprobe:int=8):
        self.top_k_articles=top_k_articles
        self.top_k_matches=top_k_matches
        self.model_name=model_name
        self.device=device
        self.max_length=max_length
        self.batch_size=batch_size
        self.wikipedia_data_path=wikipedia_data_path
        self.wikipedia_index_path=wikipedia_index_path
        self.wikipedia_faiss_index_path=wikipedia_faiss_index_path
        self.nprobe=nprobe

        ## Setup the model and tokenizer
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
        ## Setup the wikipedia articles index
        self.wikipedia_faiss_index = read_index(self.wikipedia_faiss_index_path)
        self.wikipedia_faiss_index_ivf = faiss.extract_index_ivf(self.wikipedia_faiss_index)
        self.wikipedia_faiss_index_ivf.nprobe = nprobe
        ## Load the wikipedia article index file
        self.wikipedia_article_index = pd.read_parquet(self.wikipedia_index_path)

    def get_embedding(self, query):    
        # Compute token embeddings
        with torch.no_grad():
            encoded_input = self.tokenizer(
                [query], 
                padding=True, 
                truncation=True, 
                max_length=self.max_length, 
                return_tensors='pt'
            ).to(self.device)

            model_output = self.model(**encoded_input)
            
            # Perform pooling. In this case, cls pooling.
            _sentence_embeddings = model_output[0][:, 0, :]
            
            # normalize embeddings
            _sentence_embeddings = torch.nn.functional.normalize(_sentence_embeddings, p=2, dim=1)
            sentence_embeddings = _sentence_embeddings.detach().cpu().numpy()
            #print(f'get_embeddings shape {sentence_embeddings.shape}')
        return sentence_embeddings    

    def search1(self, query):
        ## Get question choice embeddings - will be used later
        embeddings = self.get_embedding(query)
        ## Get the indices corresponding to the wikipedia article that best matches
        self.query_article_indices = []
        _, search_index = self.wikipedia_faiss_index.search(embeddings, self.top_k_articles)
        self.query_article_indices.append(list(set(search_index.flatten())))
  
    def search2(self, query):
        ## Identifying which files to perform look up and which question questions are associated with which articles
        self.query_article_data = []
        for article_index in self.query_article_indices:
            ## Within the Wikipedia Index get the articles (and associated file values) that are closest to the choices for each question
            _df = self.wikipedia_article_index.loc[article_index].copy()
        self.query_article_data.append(_df)
        self.query_article_data = pd.concat(self.query_article_data).reset_index(drop=True)
    
        ## Create the data to tell us which files to look up
        self.wikipedia_article_data = self.query_article_data[['id','file_id']].drop_duplicates().sort_values(['file_id', 'id']).reset_index(drop=True)
    
        ## Obtaining the article text data 
        self.wikipedia_article_text = []
        for file_id in self.wikipedia_article_data.file_id.unique():
            ## For the file, get all the ids pertinent that exist in that file
            _id = [i for i in self.wikipedia_article_data[self.wikipedia_article_data['file_id']==file_id]['id'].tolist()]
            _df = pd.read_parquet(f"{self.wikipedia_data_path}/{file_id}.parquet")
            _df = _df[_df['id'].isin(_id)]
            self.wikipedia_article_text.append(_df)

        wikipedia_article_text = pd.concat(self.wikipedia_article_text).drop_duplicates().reset_index(drop=True)
        wikipedia_article_text['document_id'] = wikipedia_article_text.apply(lambda x: f"{x['id']}_{x['paragraph_id']}", axis=1)
        return wikipedia_article_text
                    
    def search(self, query):
        self.search1(query)
        wikipedia_article_text = self.search2(query)
        query_embedding = self.get_embedding(query)[0]
        # Encode each sentence  
        wikipedia_article_text['embedding'] = wikipedia_article_text.apply(lambda x: self.get_embedding(x['title'] + '\n' + x['text'])[0], axis=1)
        # Cosine similarity  
        wikipedia_article_text['score'] = wikipedia_article_text['embedding'].apply(lambda x: np.dot(query_embedding, x) 
                                                                                              / (np.linalg.norm(query_embedding) * np.linalg.norm(x)))    
        scored_articles = wikipedia_article_text.sort_values(['score'], ascending=False).reset_index(drop=True)

        docs_consumed = []
        final_text = ''
        best_score = scored_articles.head(1).score.values.tolist()[0]
        for doc in scored_articles.itertuples():
            print(doc.title, doc.score, doc.id, doc.paragraph_id, doc.document_id)
            if doc.id not in docs_consumed and doc.score >= 0.95*best_score:# and len(final_text) < 2048:
                docs_consumed.append(doc.id)
                final_text += doc.title+'\n'+doc.text+'\n'
                for doc2 in scored_articles.itertuples():
                    if doc2.document_id != doc.document_id and doc2.score > 0.95*doc.score and len(final_text) < 2048:
                        final_text += doc2.text +'\n'
                final_text += '\n'
        print(final_text)
        return final_text
