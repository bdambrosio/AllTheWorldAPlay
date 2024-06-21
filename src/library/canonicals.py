from operator import attrgetter
from datetime import datetime
import sys, re
import pickle, math, random
import itertools
import numpy as np
import faiss
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
#import rewrite as rw
from PyQt5.QtWidgets import QApplication
#import OwlCoT as cot
#import Interpreter
#from LLMScript import LLMScript
import spacy
import threading
from openai import OpenAI

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"
client = OpenAI(api_key=openai_api_key, base_url=openai_api_base,)
models = client.models.list()
model = models.data[0].id

embedding_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
embedding_model = AutoModel.from_pretrained('nomic-ai/nomic-embed-text-v1', trust_remote_code=True)
embedding_model.eval()
CANONICALS_DIR = 'owl_data/'
spacy_ner = spacy.load("en_core_sci_scibert")

ADD_LOCK = threading.Lock()
EMBEDDING_LOCK = threading.Lock()

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def sample_kegg_nes (sample_size=0.004):
    global example_terms
    sample_size = int(len(rw.KEGG_SYMBOLS) * sample_size)  # Calculate 0.25% of the set size
    KEGG_Sample = set(random.sample(rw.KEGG_SYMBOLS, sample_size))
    form_prompt = """Analyze the set of KEGG named-entities in Text1 to identify the most significant lexical and semantic dimensions along which they vary. Reason step-by-step to select a set of no more than 67 canonical examples of the set of KEGG named-entities provided. Your goal should be to span the space of forms, both lexically and semantically, and to represent the diversity of forms. That is, each example should vary on at least one lexical or semantic dimension from every other example in your chosen set. NEs related to intracellular signaling pathways such as genes, RNAs (including non-coding), proteins, etc. are particularly relevant to this task, and should be heavily, although not exclusively, represented. Note that some of the KEGG NEs are composite, that is, they include subparts that are themselves NEs. An Example is '2-Methyl-branched fatty aldehyde'. Others are simple names of objects or processes. Some are predicative expressions, e.g. 'fatty aldehyde' or predicative expressions where the predicate is a phrase. 
Attend to captialization conventions, especially as they are correlated with entity class (e.g., micro RNAs might have different capitalization conventions than proteins). 
Terms like 'Rubidium hydroxide', ie simple chemical compounds, should be considered a simple name.

Return a simple numbered list of NEs, one on each line.
Respond only with your list of canonical examples, with no introductory, discursive, or explanatory text.

end your response with:
</Response>
"""
    example_terms = si.process1(arg1=str(KEGG_Sample),
                              instruction=form_prompt,
                              max_tokens=2000,
                              #template='claude-opus'
                              )
    return example_terms

#below from claude-opus with .09 sample in
example_terms="""1. TMEM229B
2. MAP-2
3. rbp6
4. MIRN213
5. hrm1
6. (Gal)2 (GalNAc)2 (GlcNAc)3
7. TAFII130
8. RCH2NH2
9. OR51A12
10. (Gal)3 (Man)6
11. (Abe)1 (Man2Ac)1
12. GPR32
13. an1
14. CLEC-1
15. DDX60
16. MIR1538
17. H2B
18. CYP7
19. hsa-mir-664b
20. TBX21
21. JIP3
22. MIRN632
23. KFABP
24. d-2-amino-3-hydroxybutyric acid
25. HPE3
26. icp55
27. eef-2
28. ACTC
29. prp38a
30. creb2
31. KOX20
32. COL18A1
33. NEMP2
34. Ipi3
35. TRPML1
36. ACO2
37. Istamycin AP
38. Klk7
39. "N2'-Acetylgentamicin C1a"
40. N-Ethylglycine
41. ZCCHC19
42. LINC00493
43. UGT2B8
44. Betaxolol
45. snRNP-B
46. COXPD12
47. omega-Cycloheptylundecanoic acid
48. d-glucurone
49. miR-21
50. miR-5010
51. "2',3'-Cyclic nucleotide"
52. 11,12-dihydroxyabieta-8,11,13-trien-20-oic acid
53. C16orf96
54. POU4F1
55. dextrose
56. (2,5-Anhydro-Man)1 (D/LHexA)2 (GlcN)3 (LIdoA)1 (S)4
57. snrnp
58. Hexachlorophene
59. IL6Q
60. IL8RB
61. P138-TOX
62. 3-methylsufonylpropyl glucosinolate
63. D-Galactose 6-phosphate
64. (GalNAc)2 (GlcNAc)4 (LFuc)2 (Man)3
65. ApoC-I
66. Methyl isothiocyanate
67. FE65
68. TWIST2
69. Gulono-1,4-lactone
70. HDACA
71. cav1.1
72. hist2h4b
73. (15:1)-Cardanol
74. retinoic acid
75. Ammelide
76. tef-4
77. (Fuc)1 (Gal)4 (Glc)1 (GlcNAc)3 (LFuc)2
78. GTF2IRD2B
79. OR5K1
80. 3-[(3aS,4S,7aS)-7a-Methyl-1,5-dioxo-octahydro-1H-inden-4-yl]propanoate
81. MIRN577
82. 2,4-Dichlorobenzoyl-CoA
83. LAMP2
84. ms2
85. 7,8-Didemethyl-8-hydroxy-5-deazariboflavin
86. CNR6
87. 17beta-Amino-5alpha-androstan-11beta-ol
88. MIR548O
89. LEPRE1
90. 1d-myo-inositol hexakisphosphate
91. VEGFB
92. n-(3,5-dichlorophenyl)-2-hydroxy-2-methyl-3-butenamide
93. DTD
94. PN2
95. PAP-2c
96. (3-(acetylhydroxyamino)propyl)phosphonic acid
97. NaG
98. SNAR-G2
99. il27b
100. MIRN1255B2
101. RCH2NH2
102. Desglucomusennin
103. RNASEP1
104. Compound IV
105. SPC7
106. dgat2l5
107. Potassium phosphate, dibasic (JAN/USP)
108. cas1
109. Pentane-2,4-dione
110. TASK-3
111. OCT6
112. 1-Alkyl-sn-glycero-3-phosphate
113. D-Homo-A-nor-17a-oxaandrost-3(5)-ene-2,17-dione
114. 3-Hydroxyethylchlorophyllide a
115. TIMAP
116. Pholedrine sulfate
117. xct
118. GPR20
119. rnf49
120. ACACT
121. 3-aminopropionamide
122. Nitidine
123. GPIBD18
124. MIR6825
125. p7b2
126. (5-L-Glutamyl)-L-glutamine
127. col12a1l
"""
example_terms = [re.sub(r'\([^)]*\)', '', line.strip()).strip('0123456789. ') for line in example_terms.split('\n')]
example_terms = '\n'.join(example_terms)

class Canonicals:
    def __init__(self, name):
        self.host_name = name
        self.filename=CANONICALS_DIR+name+'canonicals.pkl'
        self.faiss_filename=CANONICALS_DIR+name+'canonicals.faiss'
        self.terms = {}    # dictionary of registered terms: self.terms[term.lower()] -> term
        self.term_ids = {} # dictionary faiss_id -> term
        self.canonicals_indexIDMap=None # set by load
        self.unsaved_canonicals = 0
        self.load()
        if len(self.term_ids)> 0 and not all(isinstance(key, int) for key in self.term_ids):
            raise ValueError("Canonicals - Not all keys are integers.")

    
    def embedding_request(self, text, request_type='search_document: '):
        global embedding_tokenizer, embedding_model,EMBEDDING_LOCK
        if type(text) != str:
            print(f'\nError - type of embedding text is not str {text}')
        with EMBEDDING_LOCK:
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
            pickle.dump([self.terms, self.term_ids], file)
        faiss.write_index(self.canonicals_indexIDMap, self.faiss_filename)

    def load(self):
        try:
            with open(self.filename, 'rb') as file:
                terms_pkl = pickle.load(file)
                self.terms = terms_pkl[0]
                self.term_ids = terms_pkl[1]
            self.canonicals_indexIDMap = faiss.read_index(self.faiss_filename)
            print(f"loaded canonicals - {len(self.terms)} terms")
        except Exception as e:
            print(f"Canonicals load error {str(e)}\n  recreating canonicals...", end='')
            self.terms = {}
            self.term_ids = {}
            with open(self.filename, 'wb') as file:
                pickle.dump([self.terms, self.term_ids], file)
            self.canonicals_indexIDMap = faiss.IndexIDMap(faiss.IndexFlatL2(768))
            faiss.write_index(self.canonicals_indexIDMap, self.faiss_filename)
            print(f"loading KEGG terms - about 200000 total ...")
            self.load_KEGG()
            print(f"created canonicals")

    def clear(self):
        ok = input("Really? Not recoverable!")
        if not ok.lower().startswith('y'):
            return
        self.terms = {}
        self.term_ids = {}
        self.canonicals_indexIDMap = faiss.IndexIDMap(faiss.IndexFlatL2(768))
        self.save()
        print(f"recreated {self.filename} and {self.faiss_filename}")


    def generate_new_key(self):
        id = None
        while id is None:
            id = random.randint(0, sys.maxsize)
            if id in self.term_ids:
                id = None
            return id
        
    def get(self, term):
        terml = term.lower()
        if terml in self.terms:
            return self.terms[terml]
        else:
            return None
        
    def has(self, term):
        return term.lower() in self.terms

    def add(self, term, preferred_form):
        """ add a term. Ignores problem of faiss index update if term already in self.terms, for now """
        terml = term.lower()
        if terml in self.terms:
            if self.terms[terml] != preferred_form:
                print(f'conflicting preferred form for {term} registered: {self.terms[terml]}, arg: {preferred_form}')
            return self.terms[terml]
        # only enters in faiss on new term!
        with ADD_LOCK:
            self.terms[terml] = preferred_form
            id = self.generate_new_key()
            self.term_ids[id] = term
            term_embedding = self.embedding_request(terml, 'search_document: ')
            ids_np = np.array([id], dtype=np.int64)
            embeds_np = np.array([term_embedding], dtype=np.float32)
            self.canonicals_indexIDMap.add_with_ids(embeds_np, ids_np)
            self.unsaved_canonicals += 1
            # autosave, 
            if self.unsaved_canonicals > 50:
                self.save()
                self.unsaved_canonicals = 0
            return self.terms[terml]

    def KEGG_batch_iterate(self, batch_size=100):
        iter_strings = iter(rw.KEGG_SYMBOLS)
        while True:
            batch = list(itertools.islice(iter_strings, batch_size))
            if not batch:
                break
            yield batch

    def load_KEGG(self):
        batch_num=0
        for batch in self.KEGG_batch_iterate(100):
            print(f"Processing batch: {batch_num}")
            ids = [self.generate_new_key() for i in range(len(batch))]
            for n, term in enumerate(batch):
                self.term_ids[ids[n]] = batch[n]
                self.terms[term.lower()]=term
            term_embeddings = [self.embedding_request(term.lower(), 'search_document: ') for term in batch]
            ids_np = np.array(ids, dtype=np.int64)
            embeds_np = np.array(term_embeddings, dtype=np.float32)
            self.canonicals_indexIDMap.add_with_ids(embeds_np, ids_np)
            batch_num += 1
            if batch_num % 10 == 0:
                self.save()
    
    def find(self, key,  form):
        """ find multiple occurences of an xml field in a string """
        idx = 0
        items = []
        forml = form.lower()
        keyl = key.lower()
        keyle = keyl[0]+'/'+keyl[1:]
        if keyl not in forml:
            return None
        while idx < len(forml):
            start_idx = forml[idx:].find(keyl)+len(keyl)
            if start_idx < 0:
                return items
            end_idx = forml[idx+start_idx:].find(keyle)
            if end_idx < 0:
                return items
            items.append(form[idx+start_idx:start_idx+end_idx].strip())
            idx += start_idx+end_idx
        return items
            
    def analyze_term(self, term, context):
        """ analyze new term form and return preliminary canonical form """
        global example_terms
        if self.has(term):
            return canonicals.get(term)
        instructions="""Analyze Text1 in the context of biomedical named-entities and the statement provided in Text2 using known fact and the canonical examples provided in Text3. Reason step-by-step:
1. If Text1 is a named-entity, update it as needed to resemble canonical form in the examples in Text3 (syntax, capitalization, hyphenation, case, etc)
2. If Text1 is a compound phrase rather than a single named-entity, return the phrase correcting capitalization and hyphenation according to biomedical convention, as well as normaling any named-entities appearing in it as in step 1 above.
3. Do not make any changes that modify the meaning of Text1 as it appears in Text2. Ensure that the modified phrase retains the original meaning of Text2.
Respond with the updated form for Text1 in XML format as follows:

<Term>
normalized form for Text1
</Term>

Do not include any introductory, discursive, or explanatory text.
"""
        completion = client.chat.completions.create(
            messages=[ {
                "role": "user",
                "content":instructions+"\n<Text1>\n"+term+"\n</Text1>\n"+"\n<Text2>\n"+context+"\n</Text2>\n"+"\n<Text3>\n"+str(example_terms)+"\n</Text3>\n"
            }],
            model=model, temperature= 0.02, stop=['</END>', '<|eot_id|>'], max_tokens= 50)
        response = completion.choices[0].message.content.replace('\\n','\n')
        cterms = self.find('<Term>', response)
        if cterms is None or len(cterms)==0:
            print(f"\nanalyze term '{term}', canonicalization failed\n {response}")
            return term
        return cterms[0]


    def synonym_check(self, term, candidate, context):
        instruction="""Your task is to decide if the phrase in Text2 below is a synonym for the phrase in Text1 below.
Reason step-by-step:
 - Does phrase Text2 have the same meaning as Text1? That is, does it designate the same ontologic entity?
 - Can Text2 replace Text1 in the statement provided in Text3 without changing its meaning? Sometimes the ontologic entity designated by a phrase can change depending on context.

====Example:
<Text1>
ice
</Text1>

<Text2>
ice cubes
</Text2>

<Text3>
ice is cold.
</Text3>

Response: 
False

Reasoning: While the statement in Text3 is still true following substitution, it is not the case that it's meaning is unchanged. Ice cubes are made of ice, but ice cubes are not the same ontologic entity as ice, violating our first criterion


====Example:
<Text1>
ice
</Text1>

<Text2>
frozen water
</Text2>

<Text3>
ice is cold.
</Text3>

Response: 
True

Reasoning: ice is the frozen form of water, so the first criterion is satisfied. Further, the meaning of the statement in Text3 is unchanged by the substitution.

====END Examples====

Respond True or False. 
Do not include your reasoning in your response.
Do not include any introductory, discursive, or explanatory text.
Simply respond 'True' or 'False'.
End your response with:
</END>
"""
        completion = client.chat.completions.create(
            messages=[ {
                "role": "user",
                "content":instruction+"\n<Text1>\n"+term+"\n</Text1>\n"+"\n<Text2>\n"+candidate+"\n</Text2>\n"+"\n<Text3>\n"+context+"\n</Text3>\n"
            }],
            model=model, temperature= 0.05, stop=['</END>', '<|eot_id|>'], max_tokens= 20)
        response = completion.choices[0].message.content
        if 'true' in response.lower():
            return True
        else:
            return False
        
    def register(self, term, context):
        global spacy_ner
        """ register a new candidate term.
            assumes we accept all subterms as is, so they should have been checked first
            - check if term is known
            - if not, check spacy
            - if not, do faiss similarity search
            - gather first couple, then ask llm to pick best term to register:
                original term if no candidate maintains original meaning in context
                otherwise best known.
            - add original term to canonicals if original chosen
            - return chosen term
        """
        if self.has(term):
            return term

        candidates = []
        # see if spacy can find something
        found = spacy_ner(term)
        #print(f'register {term}, spacy found {found}')
        if len(found)==1: # if more than one presumably this is a compound, deal with it below
            #print(f'using spacy term')
            self.add(term, str(found[0]))
            return str(found[0])
        elif len(str(found[0])) >= len(term)-2: # if first found is at least as long, it's pbly good!
            #print(f'using spacy {str(found[0])}')
            self.add(term, str(found[0]))
            return str(found[0])
            
        else:
            #print(f'spacy found {len(found)}')
            candidates = [str(ent) for ent in found]

        # see if llm can suggest a more normalized form
        if term not in candidates:
            candidates.append(term)
        candidate = self.analyze_term(term, context)
        synonym = self.synonym_check(term, candidate, context)
        if synonym and candidate not in candidates:
            candidates.append(candidate)

        text_embedding = self.embedding_request(term, 'query_document: ')
        embeds_np = np.array([text_embedding], dtype=np.float32)
        scores, ids = self.canonicals_indexIDMap.search(embeds_np, 3) # two is enough, mostly
        # check vector similarity for nearby forms
        for id, score in zip(ids[0], scores[0]):
            if id < 0:
                break
            candidate = self.term_ids[id]
            synonym = self.synonym_check(term, candidate, context)
            if synonym and candidate not in candidates:
                candidates.append(candidate)

        if len(candidates) > 0:
            instruction="which of the entries following is the preferred form of the term as used in the biomedical literature?\n\n"\
                + '\n'.join(candidates)\
                + """
            
Respond with the preferred form.
Respond ONLY with the preferred form, with no introductory, discursive, or explanatory text.
End your response with:
</END>
"""
            completion = client.chat.completions.create(messages=[{"role":"user", "content":instruction}],
                                                        model=model,
                                                        max_tokens=20,  stop=['</END>', '<|eot_id|>']  )
            response = completion.choices[0].message.content
        # check if any candidates in response:
        for candidate in candidates:
            if candidate.strip() == response.strip():
                #register term as candidate
                self.add(term, candidate)
                return candidate
        return term
        
if __name__ == '__main__':
    c=Canonicals('signaling')
    #print(c.has('Evasion of growth suppressors'))
    print(c.has('miR-346'))
    print(c.register('miRNA-346', 'miRNA-346 controls release of TNFa in Rheumatoid Arthritis (RA) tissues'))
    #import pathways
    #print(pathways.analyze_term2("miRNA-346"))
    print(c.register("controls release of TNFa","controls release of TNFa"))
    print(c.register("Rheumatiod Arthritis (RA) tissues","Rheumatiod Arthritis (RA) tissues"))
    
    
