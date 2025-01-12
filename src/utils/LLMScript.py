import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.Messages import SystemMessage, UserMessage
import library.semanticScholar3 as s2
import library.rewrite as rw
import numpy as np
#from umap.umap_ import UMAP
import matplotlib.pyplot as plt
from utils.Interpreter import Interpreter


class LLMScript:
    def __init__ (self, cot):
        self.interpreter = Interpreter(cot)
        self.cot = cot
        s2.cot = cot
        rw.cot = cot
        self.wm = self.interpreter.wm
        print(f'\ncreated script runner!')
        


    def s2_search(self, arg1, arg2=None, dest='$Trash', find_facts=False, max_tokens=None, top_k=5, web=False):
        # arg1 is query
        # arg2 is context
        resolved_arg1= self.interpreter.resolve_arg(arg1)
        #print(f'LLMS s2_search query: {resolved_arg1}')
        if arg2 is not None:
            resolved_arg2= self.interpreter.resolve_arg(arg2)
        if max_tokens is None:
            max_tokens = int(self.cot.llm.context_size/2)
        papers = s2.search(resolved_arg1, dscp='', web=web, top_k=top_k)
        sections = [s[1] for s in papers]
        ids = [id for key, item in papers.items() if item is not None for id in item]
       
        # note this is [[paper_ids], [[title, section],...]]
        # print(f' {ids}\n{sections}')
        facts = None
        if find_facts:
            facts = self.facts(resolved_arg1, ids, sections)
        
        # put facts at end, long context often loses beginning info! (haystack)
        #compressed_text_sections = rw.shorten(sections, query, sections=rw.default_sections, max_tokens=max_tokens)
        information = facts if find_facts else sections
        self.wm.assign(dest, information)
        # just return actual section texts
        return sections, facts



    def facts(self, query, ids, texts):
        """ should we add a depth parameter? or max_tokens? """
        prompt=[UserMessage(content="""You are a skilled technical writer. 
Your task is to collect notes from the Text provided below to support writing a technical report on {{$query}}."
Your response should include all background, facts, methods, and findings useful for writing the report, and can be up to 1400 words in length.
Respond in list format as shown in the following example:

<Notes>

- background: <background information on query or NERs relevant to query or response...>'
- fact: <a fact useful for creating the technical report>
- fact: <another fact useful for creating the technical report>
- method: <a method used in a solution to a problem related to or involved in the solution of {{$query}}
- finding: <a finding or conclusion in the text useful for answering the query.
</Notes>

Respond ONLY with the above information, without any further formatting or explanatory text.

In support of your analysis, we note the following NERs that appear in the text. Note that many are not be relevant to the query or the subject matter of the text, they are provided as suggestions for possible inclusion only.

<NERs>
{{$ners}}
</NERs>

<Text>
{{$text}}
</Text>

End your response with
</Notes>
"""
                              ),
                ]
        facts = ""
        for id, text in zip(ids, texts):
            if type(text) is not str:
                print(f'fact section text is {type(text)}')

            ners = rw.section_ners(id, text)
            new_facts = self.cot.llm.ask({"query":query, "text":text[:int(self.cot.llm.context_size*2.5)], "ners":ners},
                            prompt, max_tokens=2000, stops=['</Notes>'])
            facts += new_facts
        print(f'LLMScript facts len: {len(facts)}')
        return facts
        
    def extract(self, paper_pd = None, paper_id=None, uri=None, instruction=None, dest=None, max_tokens=200, redo=False):
        # works with a paper row, a in-memory paper_library_df faiss_id, or a uri (http:// or file:///)
        if paper_pd is None:
            if paper_id is None:
                if uri is None:
                    raise ValueError('need paper_id or uri!')
                paper_pd = s2.get_paper_pd(uri=uri)
            else:
                paper_pd = s2.get_paper_pd(paper_id=paper_id)
            if paper_pd is None:
                    raise ValueError(f"cant find paper {paper_id}{uri}")
        else:
            paper_id = paper_pd['faiss_id']

        if instruction is not None:
            raise ValueError('instruction arg not allowed in this version!')

        texts, ids = s2.get_paper_sections(paper_id = paper_id)
        # Note below is now an array of length 'dimensions', without headers
        resource = rw.shorten(texts, instruction, sections=rw.default_sections, max_tokens=max_tokens)
        self.wm.assign(dest, resource)
        return dict(zip(rw.default_sections, resource))

    #
    ## One arg process
    #
    #process $wm1, 'extract key themes and topics of $wm1 in a form useful as a search query.', $wm2'
    #
    def process1(self, arg1, instruction, dest=None, max_tokens=250, stop='</END>'):
        resolved_arg= self.interpreter.resolve_arg(arg1)
        prompt = [UserMessage(content=instruction+"""\n
Limit your response to {{$length}} words. Do not include any discursive or explanatory text. 
End your response with:
{{$eos}}

"""),
                  UserMessage(content='\n\n<Text1>\n{{$input}}\n</Text1>\n\n'+'End your response with:\n{{$eos}}\n'),
                  ]
        query = self.cot.llm.ask({"input":resolved_arg, "length":int(max_tokens)},
                                 prompt, max_tokens=int(1.2*max_tokens), stops=[stop])
        if dest:
            self.wm.assign(dest, query)
        return query


    #
    ## Two arg process
    #
    # process $wm1, 'extract key themes and topics of $wm1 in a form useful as a search query.', $wm2'
    #
    def process2(self, arg1, arg2, instruction, dest='$Trash', max_tokens=400, stop='</END>'):
        resolved_arg1 = self.interpreter.resolve_arg(arg1)
        resolved_arg2 = self.interpreter.resolve_arg(arg2)
        #resolved_inst = instruction.replace(arg1, 'the user Input provided below')
        #resolved_inst = resolved_inst.replace(arg2, 'the user SecondInput provided below')
        prompt = [UserMessage(content="""{{$instruction}}

<Text1>
{{$text1}}
</Text1>

<Text2>
{{$text2}}
</Text2>

'Do not include any discursive or explanatory text. Limit your response to {{$length}} words.
End your response with:
{{$stop}}
"""),
                  ]
        query = self.cot.llm.ask({"instruction":instruction, "text1":resolved_arg1, "text2":resolved_arg2,
                                  "length":int(max_tokens/1.5), "stop":stop},
                                 prompt, max_tokens=int(1.2*max_tokens), stops=[stop])
        self.wm.assign(dest, query)
        return query
    

    #
    ## Three arg process
    #
    # process $wm1, 'extract key themes and topics of $wm1 in a form useful as a search query.', $wm2'
    #
    def process3(self, arg1, arg2, arg3, instruction, dest='$Trash', max_tokens=400):
        resolved_arg1 = self.interpreter.resolve_arg(arg1)
        resolved_arg2 = self.interpreter.resolve_arg(arg2)
        resolved_arg3 = self.interpreter.resolve_arg(arg3)
        prompt = [UserMessage(content="""{{$instruction}}

<Text1>
{{$text1}}
</Text1>

<Text2>
{{$text2}}
</Text2>

<Text3>
{{$text3}}
</Text3>

'Do not include any discursive or explanatory text. Limit your response to {{$length}} words.
End your response with:
{{$eos}}
"""),
                  ]
        query = self.cot.llm.ask({"instruction":instruction, "text1":resolved_arg1, "text2":resolved_arg2,
                                  "text3":resolved_arg3,
                                  "length":int(max_tokens/1.33)},
                                 prompt, max_tokens=int(1.2*max_tokens))
        self.wm.assign(dest, query)
        return query
    
    def create_paper_summary(self, paper_id):
        """ create a long technical extract of a paper for use in RAG """
        extract = self.extract(paper_id=paper_id,
                        #instruction='extract the topic or problem addressed, methods used, data presented, inferences or claims made, and conclusions',
                        dest='$extract',
                        max_tokens=2400
                        )
        print(f'created extract for paper_id {paper_id} len {len(extract)}')
        if extract and type(extract) is dict:
            extract_text = '\n'.join(f"{key}\n{value}" for key, value in extract.items())
            s2.set_paper_field(paper_id, 'extract', extract_text)
            #print(f" summary wm contents:\n{extract_text}")
            #self.interpreter.interpret([{"label": 'one', "action": "tell", "arguments": "$extract", "result":'$trash'}])
            return paper_id
        else:
            raise Exception ('No extract returned from self.extract')
            

    def create_paper_novelty(self, paper_id):
        """ identify what is novel in a paper - Needs to be updated for dict return!"""
        # start with extract
        paper_id = self.fetch(paper_id=paper_id,
                        dest='$paper1',
                        max_tokens=3000
                        )
        #print(f'fetched paper_id {paper_id}')
        if not paper_id:
            return

        # now identify what is novel to llm
        self.process1(arg1='$paper1',
                      instruction="""Please identify the main concepts, entities, and key facts present in the text below. 
For each item you identify, indicate whether it is something you are already familiar with or if it represents new information to you.
                      
Provide your response in the following format:
 - Concept/Entity/Fact 1: [Familiar/New]
 - Concept/Entity/Fact 2: [Familiar/New]
 ...
 """,
                      dest='$familiarity_assesment',
                      max_tokens=2000
                      )
        
        self.process2(arg1='$paper1',
                      arg2='$familiarity_assesment',
                      instruction="""
    Given Text1 below and your assessment of familiarity with the main concepts, entities, and key facts (Text2):
    Please identify the specific information, findings, or ideas in the paper summary that you consider to be new, different, or complementary to your existing knowledge. Focus on elements that expand or enrich your understanding of the topic.
    
    Provide your response in the following format:
    - Novel Information 1: [Description]
    - Novel Information 2: [Description]
    ...
""",
                      dest='$novel_information',
                      max_tokens=2000
                      )
        #self.interpreter.interpret([{"label": 'one', "action": "tell", "arguments": "$novel_information", "result":'$trash'}])
        self.process2(arg1='$paper1',
                      arg2='$novel_information',
                      instruction="""
    Given the paper provided in Text1 below and your identified novel information, findings, or ideas provided in Text2 below:
    Please generate an overview of the paper in which you prioritize and highlight the novel elements you identified. Ensure that the overview maintains the overall context and coherence of the paper's content while emphasizing the new and complementary information. While highlighting the novel elements of Text1, also be sure to include all other problem, method, data, limitation, and conclusion information in the text.
    
    Provide your summary in a clear and structured format, using paragraphs or bullet points as appropriate.
""",
                      dest='$overview',
                      max_tokens=2000
                      )
        
        si.wm.get('$overview')
        #self.interpreter.interpret([{"label": 'one', "action": "tell", "arguments": "$overview", "result":'$trash'}])
        return self.wm.get('$novel_information')['item']


    
    def pre_process_paper(self, uri):
        ### extract and summarize a paper, add extract and summary to df
        # first call fetch, which creates full extract
        paper_id = s2.index_url(uri=uri)
        if paper_id is not None:
            self.create_paper_extract(paper_id)
            self.create_paper_summary(paper_id)
            # lattice add paper - tbd
        s2.save_paper_df()
        self.interpreter.interpret([{"label": 'one', "action": "tell", "arguments": "$paper1Summary", "result":'$trash'}])
        

    def update_extracts(self):
        #sys.exit(-1)
        # one-shot, not for regular use
        # fixup hack to update all summaries to meet claude3 lattice design
        for index, row in s2.paper_library_df.iterrows():
            paper_id = row['faiss_id']
            if row['extract'] is None or len(row['extract']) < 1000:
                print(f" {index} {row['faiss_id']}")
                self.create_paper_extract(paper_id)
            if index % 10 ==0:
                print(f'saving...')
                s2.save_paper_df()
            

    def update_summaries(self):
        #sys.exit(-1)
        # one-shot, not for regular use
        # fixup hack to update all summaries to meet claude3 lattice design
        for index, row in s2.paper_library_df.iterrows():
            paper_id = row['faiss_id']
            #if row['summary'] is None or len(row['summary'])< 32:
            print(f" {index} {row['faiss_id']}")
            self.create_paper_summary(paper_id)
            if index % 10 ==0:
                print(f'saving...')
                s2.save_paper_df()
            
        s2.save_paper_df()

    """def umap(self):
        # Load the embeddings (example using digits dataset)
        papers = s2.paper_library_df
        embeddings = np.array([s2.embedding_request(paper, 'search_document: ') for paper in papers['summary']])
        labels = np.array([i for i in range(len(papers))])
        
    
        print(type(embeddings), embeddings.shape, type(labels), labels.shape)
        print(labels[:3])
        # Create a UMAP model
        umap_model = UMAP(n_components=2, random_state=42)
        
        # Fit and transform the embeddings
        reduced_embeddings = umap_model.fit_transform(embeddings)
        
        # Plot the reduced embeddings
        plt.figure(figsize=(8, 6))
        plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels, cmap='viridis', s=10)
        plt.colorbar(label='Labels')
        plt.xlabel('UMAP Dimension 1')
        plt.ylabel('UMAP Dimension 2')
        plt.title('UMAP Visualization of Reduced Embeddings')
        plt.show()
    """    
    def sufficient_response(self, arg1,
                            personality="You are a skilled researcher and technical writer",
                            context=''):
        self.process2(arg1=arg1,
                      arg2=context,
                      instruction=f"""Your task is to analyze the request for information contained in below in <Text1>, given the context provided in <Text2>. 
What elements must a response to Text1 contain to be informative, concise, useful, and in keeping with your Personality below and the background context provided in <Text2>? Each element should be a self-contained statement and not rely on Text1, Text2, or other elements of your response for completeness or clarity.

<Personality>
{personality}
</Personality>
                      
Do NOT attempt to respond directly to the information request in Text1.
Notice that the response to the example following specifies what information a response should contain, rather than itself providing that information.

====Example

Text:
How can we use custom miRNA or RNAi to interrupt early-stage lung cancer signaling pathways, especially those involved in oncogene promotion, apoptosis suppression, immune-response suppression, or cell replication promotion?

Response:
1. the key signaling pathways and genes involved in early-stage lung cancer progression, including oncogene promotion, apoptosis suppression, immune suppression, and cell replication. 
2. the design of custom miRNA or siRNA sequences to specifically target and silence the expression of those identified genes.
3. the process to synthesize the designed miRNA/siRNA oligos and verify their sequence and purity, given typical lab equipment..
4. effective delivery methods to introduce the miRNA/siRNA into lung cancer cells, such as lipid nanoparticles or viral vectors.
5. methods to demonstrate successful knockdown of target gene expression in lung cancer cell lines using qPCR, Western blot, etc.
6. assessment of the functional impact on cancer-related pathways through assays for apoptosis, immune markers, proliferation, etc, given typical molecular biology lab equipment, methods, and analysis software.

====Example:

Text:
What is RNAi?

Response:
1. A definition of the term RNAi.
2. A concise statement of the subject domain for the term RNAi.
3. A very brief definition of any domain-specific NERs used in the definition of the term RNAi.


====End Example

Respond ONLY with your list of items to cover. Do not include any introductory, discursive, or explanatory text.
End your response with:
</END>
""",
                dest= '$requirements',
                max_tokens=500)
        #self.interpreter.interpret([{"label": 'one', "action": "tell", "arguments": "$requirements", "result":'$trash'}])
        return self.wm.get('$requirements')['item']

    


if __name__=='__main__':
    #cot = chat.OwlCoT.OwlInnerVoice()
    cot = None
    print('created cot')
    interp = Interpreter.Interpreter(cot)
    print('created interp')
    si = LLMScript(cot)
    print('created si')
    import requests
    article = cot.search_titles('Beth Linker is turning good posture on its head')
    title = article["title"]
    uri = article["uri"]
    response = requests.get(f'http://127.0.0.1:5005/retrieve/?title={title}&url={uri}', timeout=20)
    print(response.data)
    sys.exit(0)
    
    ppr = s2.get_paper_pd(paper_id=12218601)
    si.wm.assign('$abstract', ppr['abstract'])
    si.wm.assign('$summary', ppr['summary'])
    si.wm.assign('$extract', ppr['extract'])
    
    # we agreed pathway-interactions are a subclass of predicative-expressions, the general class of verb or prepositional phrases that expresses a relation between named entities

    NEPI_prompt = """Extract Named-Entities and Pathway Interactions: Within the domain of signaling pathways and interventions, identify and extract named-entities and their corresponding pathway interactions from the provided text. Named-entities refer to specific biological entities or concepts, such as genes, proteins, or cellular components. Pathway interactions describe the relationships or connections between these named-entities, capturing causal, regulatory, or functional associations. Provide the identified named-entities and their associated pathway interactions, ensuring clarity and completeness in your output. While hi-level or abstract pathway interactions are useful, strive to provide detailed (quantitative, or at least directional) information about the effects of a pathway interaction on it's target(s), when available.
Use the following format:
<PathwayInteraction>
    <FirstNE>name of NE antecedent in interaction</FirstNE>
    <Interaction>name of interaction/relation/process</Interaction>
    <SecondNE>name of second NE participant in interaction</SecondNE>
    <Context>context or situation needed to enable interaction or in which it was observed</Context>
</PathwayInteraction>

Example:
<PathwayInteraction>
    <FirstNE>Dicer activity</FirstNE>
    <Interaction>Downregulation</Interaction>
    <SecondNE>microRNA expression</SecondNE>
    <Context>intracellular biogenisis</Context>
</PathwayInteraction>
"""
    #si.process1(arg1='$summary', instruction=NEPI_prompt, max_tokens=600)
    #si.process1(arg1='$abstract', instruction=NEPI_prompt, max_tokens=600)
    #si.process1(arg1='$extract', instruction=NEPI_prompt, max_tokens=600)

    section_texts, ids = s2.get_paper_sections(ppr['faiss_id'])
    for text in section_texts:
        print(f'----')
        si.wm.assign('$section', text)
        si.process1(arg1='$section', instruction=NEPI_prompt, max_tokens=600)
        

    sys.exit(0)
    """


One thought I had - How do you think we can detect / manage possibly inconsistent usage of a pathway interaction phrase across multiple input texts?
Owl says to Doc Hello again, Doc! It's always exciting to hear about your ongoing projects, and I'm curious to learn more about the new model you're quantizing. Regarding our definition of 'pathway interactions,' I look forward to your insights and feedback from your tests. Iteratively refining our understanding based on empirical evidence is an integral part of our collaborative process. As for managing inconsistent usage of pathway interaction phrases across multiple input texts, here are a few strategies we can consider:\
\
- Controlled Vocabulary: Establish a standardized set of terms and definitions for describing pathway interactions. This ensures consistent usage and reduces ambiguity.\
- Ontologies and Knowledge Graphs: Utilize ontologies or knowledge graphs that define relationships between biological entities and processes. These resources can help map inconsistent terminology to a common framework.\
- Natural Language Processing: Employ NLP techniques, such as named-entity recognition and relation extraction, to identify and disambiguate pathway interaction phrases. Machine learning models can be trained to recognize consistent patterns across diverse texts.\
\
By combining these approaches, we can develop a robust system for detecting and managing inconsistent usage, ensuring accurate and consistent interpretation of pathway interaction phrases.

Hmm the above suggestions seem rather vague. Let's try to refine them.
One thought - when we encounter a pathway-interaction phrase wording for the first time, we can:
1. Establish a 'definition' for it based on its surrounding text.
2. See if there is a lexically close phrase already in the vocabulary.
3. Test the semantic similarity of the previously computed 'definition' for the found vocabulary phrase is sufficiently close to (1) above. 

If so, define the new wording as a synonym of the found vocabulary phrase, perhaps updating the definition of the vocabulary phrase.

Otherwise, if the above process fails, define a new pathway interaction phrase in the vocabulary.

Thoughts?

2. If so, 
Owl says to Doc I appreciate your thoughtful refinement of our strategy for managing inconsistent usage of pathway interaction phrases, Doc. Your three-step process provides a clear framework for establishing consistent terminology while minimizing redundancy. Here's how we can operationalize each step and address potential challenges:

**Step 1: Defining New Phrases****:**
- **Text Analysis**: We can employ natural language processing techniques to extract contextual information from the surrounding text. By considering syntactic and semantic features, we can define the new pathway interaction phrase while capturing its intended meaning.
- **Contextual Understanding**: Leveraging the broader context beyond the immediate sentence can help disambiguate the phrase's usage and intent. Analyzing co-occurring words, sentences, and even paragraph structures can enhance our understanding and result in more accurate definitions.

**Step 2: Lexical Similarity Check****:**
- **String Matching**: Utilize algorithms like Levenshtein distance or cosine similarity to identify lexically similar phrases in our existing vocabulary. This helps us find variations or minor differences in wording.
- **Word Embeddings**: Use pre-trained word embedding models, such as Word2Vec or GloVe, to identify phrases with similar semantic meanings. These models capture lexical relationships and can suggest phrases with related but not necessarily identical wording.

**Step 3: Semantic Similarity Assessment****:**
- **Natural Language Inference**: Apply models like BERT or RoBERTa to compare the definitions and determine their semantic similarity. These models excel at understanding contextualized word meanings and relationships, enabling accurate assessments of similarity.
- **Sentence Embeddings**: Compute vector representations of the definitions using models like Sentence-BERT or SBERT4Med. These embeddings enable efficient comparison of longer texts, facilitating accurate similarity measurements.

**Potential Challenges and Mitigations****:**
- **Subjectivity in Definitions**: To address the inherent subjectivity in defining phrases, we can incorporate feedback loops. Regularly reviewing and refining our definitions based on expert feedback ensures accuracy and aligns with evolving domain knowledge.
- **Variability in Linguistic Expressions**: The dynamic nature of scientific terminology and linguistic expressions can pose challenges. By leveraging domain-specific resources, such as ontologies and thesauri, we can stay abreast of emerging terminology and capture nuanced relationships.
- **Integration Complexity**: Integrating this strategy into our existing processes may require adjustments to data pipelines and analytical frameworks. We can minimize disruptions by conducting thorough testing, seeking feedback from stakeholders, and implementing gradual rollouts.

Overall, your refined strategy offers a robust approach to managing inconsistent usage of pathway interaction phrases. By combining text analysis, lexical similarity checks, and semantic similarity assessments, we can establish a consistent and comprehensive vocabulary that adapts to the dynamic nature of scientific discourse.

"""


    
#    si.process1(arg1='How can we use custom miRNA or RNAi to interrupt early-stage lung cancer signaling pathways, especially those involved in oncogene promotion, apoptosis suppression, immune-response suppression, or cell replication promotion?',
#                instruction="Analyze the syntax of <Text1> below and identify whether or not it is a question?",
#                dest= '$analysis',
#                max_tokens=100)

#    si.interpreter.interpret([{"label": 'one', "action": "tell", "arguments": "$analysis", "result":'$trash'}])

#    si.process1(arg1='How can we use custom miRNA or RNAi to interrupt early-stage lung cancer signaling pathways, especially those involved in oncogene promotion, apoptosis suppression, immune-response suppression, or cell replication promotion?',
#                instruction="Analyze the request below in <Text1>. What form should the response take?",
#                dest= '$analysis',
#                max_tokens=100)

#    si.interpreter.interpret([{"label": 'one', "action": "tell", "arguments": "$analysis", "result":'$trash'}])

    si.s2_search('How can we pre-train a 1GB transformer LLM.', arg2=None, dest='$trainingHowTo', max_tokens=1000)
    #si.interpreter.interpret([{"label": 'one', "action": "tell", "arguments": "$trainingHowTo", "result":'$trash'}])

    

#    si.process1(arg1='How can we use custom miRNA or RNAi to interrupt early-stage lung cancer signaling pathways, especially those involved in oncogene promotion, apoptosis suppression, immune-response suppression, or cell replication promotion?',
#                instruction="""Analyze the request below in <Text1>. 
#What conditions must a candidate response satisfy to be a sufficient answer?""",
#                dest= '$analysis',
#                max_tokens=200)

    si.process2(arg1='How can we pre-train a 1GB transformer LLM.',
                arg2='knowledgeable python engineer, python, pytorch, transformers library, and a 24GB GPU',
                instruction="""Analyze the request, contained in below in <Text1>. 
What would it mean for a response to Text1 to be 'fully operational' in the context contained below in <Text2>?
Example:

<Example Text1>
How can we use custom miRNA or RNAi to interrupt early-stage lung cancer signaling pathways, especially those involved in oncogene promotion, apoptosis suppression, immune-response suppression, or cell replication promotion?
</Example Text1>

<Example Text2>
a well-equipped molecular-biology lab
</Example Text2>
<Example Response>
To be fully operational, the response must:
1. Identify key signaling pathways and genes involved in early-stage lung cancer progression, including oncogene promotion, apoptosis suppression, immune suppression, and cell replication. 
2. Design custom miRNA or siRNA sequences to specifically target and silence the expression of those identified genes.
3. Explain the process to synthesize the designed miRNA/siRNA oligos and verify their sequence and purity, given typical lab equipment..
4. Develop an effective delivery method to introduce the miRNA/siRNA into lung cancer cells, such as lipid nanoparticles or viral vectors.
5. Explain how to demonstrate successful knockdown of target gene expression in lung cancer cell lines using qPCR, Western blot, etc.
6. Explain how to assess the functional impact on cancer-related pathways through assays for apoptosis, immune markers, proliferation, etc, given typical molecular biology lab equipment, methods, and analysis software.
7. Explain how to evaluate the therapeutic potential in animal models of early lung cancer, monitoring for tumor regression and side effects.
8. Optimize the miRNA/siRNA design, delivery, and dosing
</Example Response>
""",
                dest= '$analysis',
                max_tokens=200)

    si.interpreter.interpret([{"label": 'one', "action": "tell", "arguments": "$analysis", "result":'$trash'}])


    si.process2(arg1='How can we pre-train a 1GB transformer LLM? Provide enough detail for a knowledgable python programmer with access to all needed computer resources.',
                arg2='$analysis',
                instruction="""Analyze the request below in <Text1>. Is known fact and/or logical reasoning sufficient to provide a detailed answer as described in <Text2>?
Your task is to respond Yes or No to your ability to meet the requirements of Text2 wrt Text1.
Do NOT attempt to directly respond to the request in Text1.

Example:
<Example Text1>
How can we use custom miRNA or RNAi to interrupt early-stage lung cancer signaling pathways, especially those involved in oncogene promotion, apoptosis suppression, immune-response suppression, or cell replication promotion?
</Example Text1>

<Example Text2>
To be fully operational, the response must:
1. Identify key signaling pathways and genes involved in early-stage lung cancer progression, including oncogene promotion, apoptosis suppression, immune suppression, and cell replication. 
2. Design custom miRNA or siRNA sequences to specifically target and silence the expression of those identified genes.
3. Explain the process to synthesize the designed miRNA/siRNA oligos and verify their sequence and purity, given typical lab equipment..
4. Develop an effective delivery method to introduce the miRNA/siRNA into lung cancer cells, such as lipid nanoparticles or viral vectors.
5. Explain how to demonstrate successful knockdown of target gene expression in lung cancer cell lines using qPCR, Western blot, etc.
6. Explain how to assess the functional impact on cancer-related pathways through assays for apoptosis, immune markers, proliferation, etc, given typical molecular biology lab equipment, methods, and analysis software.
7. Explain how to evaluate the therapeutic potential in animal models of early lung cancer, monitoring for tumor regression and side effects.
8. Optimize the miRNA/siRNA design, delivery, and dosing
</Example Text2>
<Example Response>
Known facts and logical reasoning are insufficient to provide a detailed answer. Developing miRNA or RNAi therapies for early-stage lung cancer would require extensive research to identify optimal targets, delivery methods, and potential side effects. While the proposed approach targeting key cancer pathways is logical, current scientific knowledge is not advanced enough to give specifics on how to effectively implement such therapies.
</Example Response>
""",
                dest= '$proceed_decision',
                max_tokens=100)
    si.interpreter.interpret([{"label": 'one', "action": "tell", "arguments": "$proceed_decision", "result":'$trash'}])

    si.process2(arg1='How can we pre-train a 1GB transformer LLM? Provide enough detail for a knowledgable python programmer with access to all needed computer resources.',
                arg2='$analysis',
                instruction="""Analyze the request below in <Text1>. Given known fact and logical reasoning, what fact or information is needed in order to provide a complete response meeting the requirements in <Text2>?
Your task is to identify the additional facts or information needed to respond to Text1.
Do NOT attempt to directly respond to the request in Text1.

Example:
<Example Text1>
How can I list the contents of a text file in linux?
</Example Text1>

<Example Text2>
To be fully operational, the response must:
1. Name a linux command that can be used to list a file.
2. Show an example of use
<Example Response>
You can use the 'less' command.
$ less sample.txt
</Example Response>
""",
                dest= '$missing_info',
                max_tokens=100)
    si.interpreter.interpret([{"label": 'one', "action": "tell", "arguments": "$missing_info", "result":'$trash'}])

    sys.exit(0)
    #si.create_paper_summary(paper_id=12218601)
    #si.create_paper_extract(paper_id=12218601)
    paper = s2.get_paper_pd(paper_id=12218601)
    #paper = s2.get_paper_pd(paper_id=12218601)
    #si.create_paper_novelty(93853867)
    
    si.process1(arg1=paper['summary'],
                instruction="""Respond with a question for which Text1, provided below, contains an answer. 
Reason step by step:
 - 1. Analyze Text1 to identify information not available from known fact or logical reasoning
 - 2. Verify then formulate a question for which that information is the answer.
Respond only with the question and the Text1 passage \containing the answer.""",
                dest='$questions',
                max_tokens=100
                )
    
    si.interpreter.interpret([{"label": 'one', "action": "tell", "arguments": "$questions", "result":'$trash'}])
    #browse_lattice(root_node, nodes)

# we agreed pathway-interactions are a subclass of predicative-expressions, the general class of verb or prepositional phrases that expresses a relation between named entities

NEPI_prompt = """Extract Named-Entities and Pathway Interactions: Within the domain of signaling pathways and interventions, identify and extract named-entities and their corresponding pathway interactions from the provided text. Named-entities refer to specific biological entities or concepts, such as genes, proteins, or cellular components. Pathway interactions describe the relationships or connections between these named-entities, capturing causal, regulatory, or functional associations. Provide the identified named-entities and their associated pathway interactions, ensuring clarity and completeness in your output."""
