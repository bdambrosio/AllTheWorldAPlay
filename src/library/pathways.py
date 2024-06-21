import sys
import threading
import itertools
from transformers import AutoTokenizer, AutoModel
import re
import library.semanticScholar3 as s2
from library.canonicals import Canonicals

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=openai_api_key,
    base_url=openai_api_base,
)

models = client.models.list()
model = models.data[0].id
print(model)

embedding_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
embedding_model = AutoModel.from_pretrained('nomic-ai/nomic-embed-text-v1', trust_remote_code=True)
embedding_model.eval()
MEMORY_DIR = 'owl_data/'
PATHWAY_INTERACTIONS_FILEPATH = 'pathways/interactions70B_Int8-3.xml'

NEPI = """Within the domain of biomedical signaling pathways and -onomic interactions, identify and extract named-entities and their corresponding interactions from the provided text. Named-entities refer to specific biological entities or concepts, such as genes, RNAs, proteins, cellular components, drugs, pathologies or disease states. Interactions describe the mechanisms, processes, relationships or connections between these named-entities, capturing causal, regulatory, or functional associations. Provide the identified named-entities and their associated interactions, ensuring clarity and completeness in your output. While hi-level or abstract pathway interactions are useful, strive to provide detailed - quantitative where available, or qualitative, directional (e.g. 'increases', 'induces', etc) - information about the effects of an interaction on its target(s).

The text to be processed for pathway interactions is provided in <Text1>.
Use the following format for your response:

<Interaction>
  <Inputs>
    <Input>name of NE input or cause of interaction</Input>
  </Inputs>
  <Effect>name of specific interaction/relation/process</Interaction>
  <Targets>
    <Target>name of NE affected by interaction</Target>
  </Targets>
  <Context>The context or situation in which the interaction is observed. Include information about: 
 - where (in) organ-system, organ, cell-type, intra-cellular organelle, extra-cellular location, etc.
 - when (during, before, after, following) temporal relation to related context or active interaction.
 - enabling conditions (in the presence of, ...) sufficient condition supporting interaction: Drug, disease-state, named-entity or interaction.
 - inhibiting conditions (except when, ...) sufficient context discouraging or inhibiting this interaction
Include details from the sentence in which the interaction is mentioned as well as those in the broader surrounding text that pertain to the interaction.
</Context>
</Interaction>

====Example:
Text:
2 miRNAs were identified (miR-10a, miR-519a), the expressions of which decreased after tamoxifen treatment of patients with breast cancer.

<Interaction>
  <Inputs>
    <Input>Tamoxifen</Input>
  </Inputs>
  <Effect>decreases expression</Effect>
  <Targets>
    <Target>miR-10a</Target>
    <Target>miR-519a</Target>
  </Targets>
  <Context>during: Breast Cancer</Context>
</Interaction>

====Example:
Text:
Evading growth supression and sustaining proliferative signaling
Ectopic expression of miR-221/222 accelerated cell proliferation, whereas their suppression induced G1 cell-cycle
arrest in cancer cells. Moreover, miR-221/222 expression has been found to be upregulated in a variety of human tumors,demonstrating that miR-221/222 regulation of p27Kip1 is a bona fide oncogenic pathway.

<Interaction>
  <Inputs>
    <Input>miR-221 Ectopic expression</Input>
  </Inputs>
  <Effect>accelerates</Effect>
  <Targets>
    <Target>cell proliferation</Target>
  </Targets>
  <Context>in: cancer cells, during: cell proliferation</Context>
</Interaction>

<Interaction>
  <Inputs>
    <Input>miR-221 suppression</Input>
  </Inputs>
  <Effect>induces</Effect>
  <Targets>
    <Target>G1 cell-cycle arrest</Target>
  </Targets>
  <Context>in: cancer cells, during: apoptosis</Context>
</Interaction>

<Interaction>
  <Inputs>
    <Input>miR-221/222</Input>
  </Inputs>
  <Effect>regulates</Effect>
  <Targets>
    <Target>p27Kip1</Target>
  </Targets>
  <Context>in: variety of human tumors</Context>
</Interaction>

====Example:
Text:
Host Immune Response.
Yang and colleagues have studied various malignant cell lines and viral-infected cell lines. They found that on treatment with Interferon (IFN), the expression of miR-21 was increased three-to five fold.

<Interaction>
  <Inputs>
    <Input>Interferon (IFN)</Input>
  </Inputs>
  <Effect>increases expression</Effect>
  <Targets>
    <Target>miR-21</Target>
  </Targets>
  <Context>during: Host Immune Response, enabling: Malignant Tumors/Viral Infections</Context>
</Interaction>

====Example:
Text:
Transcriptional control of miRNAs\nMiRNA expression is tightly controlled by different transcription factors, so abnormal expression of miRNA in cancer could be due to dysregulation of some key transcription factors, such as c-Myc and p53.\nO\'Donnell et al. \n28\n discovered that c-Myc, frequently upregulated in many malignancies to regulate cell proliferation and apoptosis, activates the transcription of oncogenic miR-17-92 cluster through binding to E-box elements in miR-17-92 promoter. Consistent with its oncogenic role, c-Myc also represses transcriptional activity of tumor suppressive miRNAs such as mir-15a, miR-26, miR-29, mir-30 and let-7 families.

<Interaction>
  <Inputs>
    <Input>c-Myc</Input>
  </Inputs>
  <Effect>activates transcription</Effect>
  <Targets>
    <Target>miR-17-92 cluster</Target>
  </Targets>
  <Context>Cancer Cell Proliferation and Apoptosis</Context>
</Interaction>

<Interaction>
  <Inputs>
    <Input>c-Myc</Input>
  </Inputs>
  <Effect>represses transcription</Effect>
  <Targets>
    <Target>mir-15a</Target>
    <Target>miR-26</Target>
    <Target>miR-29</Target>
    <Target>mir-30</Target>
    <Target>let-7 family</Target>
  </Targets>
  <Context>during: Cancer Cell Proliferation and Apoptosis</Context>
</Interaction>

====Example:
Text:
For example, Pawlicki and Steitz \n[11]\n observed that pri-miRNA transcripts retained at transcription sites due to the deletion of 3′-end processing signals are converted into precursor miRNAs (pre-miRNAs) more efficiently than pri-miRNAs that are cleaved, polyadenylated, and released.

<Interaction>
  <Inputs>
    <Input>pri-miRNA transcripts retained at transcription sites</Input>
  </Inputs>
  <Effect>converted more efficiently</Effect>
  <Targets>
    <Target>precursor miRNAs (pre-miRNAs)</Target>
  </Targets>
  <Context>deletion of 3′-end processing signals</Context>
</Interaction>

====Example:
Text:
Hata and Kashima \n[5]\n have reviewed and analyzed various defects in miRNA biogenesis and some examples of changes in the activity of microprocessor Drosha.

*** No Interactions ***

==== End Examples ====

INSTRUCTION
Within the domain of biomedical signaling pathways and -onomic interactions, identify and extract named-entities and their corresponding interactions from the provided text. Named-entities refer to specific biological entities or concepts, such as genes, RNAs, proteins, or cellular components. Pathway interactions describe the mechanisms, processes, relationships or connections between these named-entities, capturing causal, regulatory, or functional associations. Provide the identified named-entities and their associated pathway interactions, ensuring clarity and completeness in your output. While hi-level or abstract pathway interactions are useful, strive to provide detailed - quantitative where available, or qualitative, directional (e.g. 'increases', 'induces', etc) - information about the effects of an interaction on its target(s).

The text to be processed for pathway interactions is provided in <Text1>.


"""

ontology_term_pattern = r'<text>(.*?)</text>'
def analyze_term2(term, context):
    ### extend Kegg w terms in interactions.xml
    ### lookup subphrases! (for startword (for subphrase starting at remaining words ...))
    ### n^2 in phrase length, not too bad
    ### prompt w these recognized NEs
    prompt="""Please parse the biomedical phrase provided in Text1 into subphrases that might be considered named entities in biomedical ontologies. For each subphrase, provide the following information in XML format:

- The subphrase itself
- A concise sentence that describes the subphrase referent, esp including its immediate parent class if relevant.
- The type of biomedical ontology or ontologies where the subphrase is likely to be found as a named entity.

If the entire phrase is likely to be a named entity in itself, please include that information as well.

====Example
biomedical phrase: 
Evasion of growth suppressors

parse:
<biomedical_phrase>
  <subphrase>
    <text>evasion</text>
    <description>A process by which an organism or cell avoids or escapes from a particular stimulus or response</description>
    <ontology_types>
      <ontology>Biological process ontologies</ontology>
    </ontology_types>
  </subphrase>
  
  <subphrase>
    <text>growth suppressors</text>
    <description>Molecules or genes that inhibit or prevent cell growth and proliferation</description>
    <ontology_types>
      <ontology>Gene ontologies</ontology>
      <ontology>Molecular function ontologies</ontology>
    </ontology_types>
  </subphrase>
  <entire_phrase>
    <text>Evasion of growth suppressors</text>
    <description>A mechanism by which cells avoid the inhibitory effects of growth suppressors, often leading to uncontrolled growth and tumorigenesis</description>
    <ontology_types>
      <ontology>Cancer-related ontologies</ontology>
      <ontology>Cellular process ontologies</ontology>
    </ontology_types>
  </entire_phrase>
  <relevant_ontologies>
    <ontology>Gene Ontology (GO)</ontology>
    <ontology>Pathway Ontology (PW)</ontology>
    <ontology>National Cancer Institute Thesaurus (NCIt)</ontology>
    <ontology>Cell Ontology (CL)</ontology>
    <ontology>Biological Process Ontology (BPO)</ontology>
  </relevant_ontologies>
</biomedical_phrase>

====Example
biomedical phrase:
Microprocessor-mediated miRNA biogenesis

parse:
<biomedical_phrase>
  <subphrase>
    <text>Microprocessor</text>
    <description>A complex of proteins involved in the processing of primary microRNA transcripts</description>
    <ontology_types>
      <ontology>Protein complex ontologies</ontology>
      <ontology>RNA processing-related ontologies</ontology>
    </ontology_types>
  </subphrase>
  
  <subphrase>
    <text>miRNA</text>
    <description>A type of non-coding RNA molecule that regulates gene expression by binding to messenger RNA</description>
    <ontology_types>
      <ontology>RNA ontologies</ontology>
      <ontology>Gene regulation-related ontologies</ontology>
    </ontology_types>
  </subphrase>
  
  <subphrase>
    <text>biogenesis</text>
    <description>The process of formation and development of a molecule or cellular component</description>
    <ontology_types>
      <ontology>Biological process ontologies</ontology>
      <ontology>Molecular function-related ontologies</ontology>
    </ontology_types>
  </subphrase>

  <entire_phrase>
    <text>Microprocessor-mediated miRNA biogenesis</text>
    <description>The process of miRNA formation and development mediated by the Microprocessor complex</description>
    <ontology_types>
      <ontology>RNA processing-related ontologies</ontology>
      <ontology>Gene regulation-related ontologies</ontology>
    </ontology_types>
  </entire_phrase>

  <relevant_ontologies>
    <ontology>Gene Ontology (GO)</ontology>
    <ontology>RNA Ontology (RNAO)</ontology>
    <ontology>Protein Ontology (PRO)</ontology>
    <ontology>Ontology for Biomedical Investigations (OBI)</ontology>
    <ontology>Sequence Ontology (SO)</ontology>
  </relevant_ontologies>
</biomedical_phrase>

====Example
biomedical phrase:
hnRNP A1 Binding

parse:
biomedical_phrase>
  <subphrase>
    <text>hnRNP A1</text>
    <description>A specific heterogeneous nuclear ribonucleoprotein (hnRNP) that binds to RNA and plays a role in various aspects of RNA metabolism</description>
    <ontology_types>
      <ontology>Protein ontologies</ontology>
      <ontology>RNA-binding protein ontologies</ontology>
    </ontology_types>
  </subphrase>
  
  <subphrase>
    <text>Binding</text>
    <description>A molecular interaction between two or more molecules, such as proteins and RNA</description>
    <ontology_types>
      <ontology>Molecular interaction ontologies</ontology>
    </ontology_types>
  </subphrase>

  <entire_phrase>
    <text>hnRNP A1 Binding</text>
    <description>The process by which hnRNP A1 interacts with RNA or other molecules</description>
    <ontology_types>
      <ontology>Molecular interaction ontologies</ontology>
      <ontology>RNA-processing ontologies</ontology>
    </ontology_types>
  </entire_phrase>

  <relevant_ontologies>
    <ontology>UniProt Knowledgebase (UniProtKB)</ontology>
    <ontology>Gene Ontology (GO)</ontology>
    <ontology>Protein Ontology (PRO)</ontology>
    <ontology>RNA Ontology (RNAO)</ontology>
    <ontology>Interaction Network Ontology (INO)</ontology>
  </relevant_ontologies>
</biomedical_phrase>

End your response with:
</END>
"""
    response=si.process1(arg1=term, 
                         instruction=prompt,
                         max_tokens=4000,
                         eos='</END>')
    if type(response) is not str:
        return response
    #canonicalize all proposed terms
    print(f'response\n{response}')
    print(f'term\n{term}')
    matches = re.findall(ontology_term_pattern, term)
    # sort by length (shortest first),
    # update all downstream matches with registered form
    for match in matches:
        candidate = match[len('<text>'):-len('</text>')]
        print(f' proposed term {candidate}')
        registered_term = canonicals.register(candidate, context)
        print(f' result of registered {registered_term}')
        if registered_term != candidate:
            response = response.replace(candidate, registered_term)
    return response
    
#sys.exit(0)
    
def find(key,  form):
    """ find multiple occurences of an xml field in a string """
    idx = 0
    items = []
    forml = form.lower()
    keyl = key.lower()
    keyle = keyl[0]+'/'+keyl[1:]
    while idx < len(forml):
        start_idx = forml[idx:].find(keyl)+len(keyl)
        if start_idx < 0:
            return items
        end_idx = forml[idx+start_idx:].find(keyle)
        if end_idx < 0:
            return items
        items.append(form[idx+start_idx:idx+start_idx+end_idx].strip())
        idx += start_idx+end_idx
    return items

def register_term(term, context):
    term = canonicals.register(term, context)
    return term

registered_interactions = []
def register_interaction(interaction_xml):
    global registered_interactions
    """ register all terms in an interaction xml and update the xml """
    if interaction_xml is None or '<Inputs>' not in interaction_xml:
        return None
    sectionIds = find('<SectionId>', interaction_xml)
    inputs = find('<Input>', interaction_xml)
    #print(f'inputs {inputs}')
    effects = find('<Effect>', interaction_xml)
    effects = effects[0] if effects is not None and len(effects)>0 else None
    #print(f'interaction {interaction}')
    targets= find('<Target>', interaction_xml)
    #print(f'targets {targets}')
    contexts = find('<Context>', interaction_xml)
    context = contexts[0] if contexts is not None and len(contexts)>0 else None
    #print(f'context {context}')
    if inputs is None or len(inputs)==0 or effects is None\
       or targets is None or len(targets)==0 or context is None:
        print(f'parse failure\n {interaction_xml}\nEnd')
        return None
    sectionId="NA"
    if sectionIds is not None and len(sectionIds) > 0:
        sectionId = sectionIds[0]
        
    interaction_text = ' and '.join(inputs)+' '+effects+' '+' and '.join(targets)+' in '+context
    #print(f'register_interaction context: {interaction_text}')
    #
    ## process and canonicalize found pathway interaction
    #
    reg_inputs = []
    for term in inputs:
        reg_input = register_term(term, interaction_text)
        reg_inputs.append(reg_input)
        
    reg_effects = register_term(effects, interaction_text)
    
    reg_targets = []
    for target in targets:
        reg_target = register_term(target, interaction_text)
        reg_targets.append(reg_target)
        
    reg_context = register_term(context, interaction_text)
    
    registered_interaction = make_xml_interaction(sectionId, reg_inputs, reg_effects, reg_targets, reg_context)
    registered_interactions.append(registered_interaction)
    return registered_interaction


def make_xml_interaction(sectionId, inputs, interaction, targets, context):
    xml_sectionId="  <SectionId>"+sectionId+"</SectionId>\n"
    xml_inputs = []
    for input in inputs:
        if input is not None and len(input)>0:
            xml_inputs.append("    <Input>"+input+"</Input>")
    xml_inputs = "  <Inputs>\n"+'\n'.join(xml_inputs)+"\n  </Inputs>\n"

    xml_interaction = "  <Effect>"+interaction+"</Effect>\n"
    
    xml_targets = []
    for target in targets:
        if target is not None and len(target)>0:
            xml_targets.append("    <Target>"+target+"</Target>")
    xml_targets = "  <Targets>\n"+'\n'.join(xml_targets)+"\n  </Targets>\n"

    xml_context = "  <Context>"+context+"</Context>\n"
    
    return "".join(["<Interaction>\n",xml_sectionId, xml_inputs,xml_interaction,xml_targets,xml_context,"</Interaction>"])


responses = []
threads = []

def send_request(text,id):
    global responses
    completion = client.chat.completions.create(
        messages=[ {
            "role": "user",
            "content":NEPI+"\n<Text1>\n"+text+"\n</Text1>\n"
        }],
        model=model,
        temperature= 0.02,
        stop=['</END>','<|eot_id|>'],
        max_tokens= 2160)
    #print(completion.choices[0].message.content.replace('\\n','\n'))
    interactions = completion.choices[0].message.content.replace('\\n','\n')
    #insert section id as source ref
    interactions = interactions.replace('<Interaction>',f'<Interaction>\n  <SectionId>{id}</SectionId>')
    responses.append(interactions)
    print(f'send_request {len(responses)}')

def get_pathways(sections, ids):
    global threads, responses
    threads = []; responses = []
    for section, id in zip(sections,ids):
        thread = threading.Thread(target=send_request, args=(section, id,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()
    print(f'get_pathways {len(responses)}')
    return '\n'.join(responses)

def interaction_batch_iterate(interactions, batch_size=50):
        iter_strings = iter(interactions)
        while True:
            batch = list(itertools.islice(iter_strings, batch_size))
            if not batch:
                break
            yield batch

def split_interactions(long_string):
    # Split the long string on <Interaction> and keep the delimiter
    interactions = long_string.split("<Interaction>")
    # Remove any empty strings from the resulting list
    interactions = [interaction for interaction in interactions if interaction.strip()]
    # Prepend "<Interaction>" to each split except the first one
    interactions = ["<Interaction>" + interaction for interaction in interactions[1:]]
    # Add the first split (if non-empty) without prepending "<Interaction>"
    if interactions[0].strip():
        interactions.insert(0, interactions[0])
    return interactions

def pathway_interactions(ppr_ids):
    global registered_interactions
    in_cnt = 0; terms = {}; pathway_cnt = 0
    for id in ppr_ids:
        ppr = None
        try:
            ppr = s2.get_paper_pd(paper_id=id)
        except Exception as e:
            print(f'Exception getting paper {id} str(e)')
        if ppr is None:
            continue
        section_texts, ids = s2.get_paper_sections(ppr['faiss_id'])
        print(f' sections {len(section_texts)}')
        pathways = get_pathways(section_texts, ids)
        interactions = split_interactions(pathways)

        interaction_batch_num = 0
        for batch in interaction_batch_iterate(interactions):
            interaction_batch_num += 1
            print(f"Processing interaction batch: {interaction_batch_num}")

            threads = []
            for interaction in batch:
                #print(f'processing \n{interaction}')
                thread = threading.Thread(target=register_interaction, args=(interaction,))
                threads.append(thread)
                thread.start()
            for thread in threads:
                thread.join()
            print(f'processed batch, writing {len(registered_interactions)}')
            with open(PATHWAY_INTERACTIONS_FILEPATH, 'a') as pf:
                for interaction in registered_interactions:
                    pf.write(interaction+'\n')
            registered_interactions = []
        canonicals.save()
            

if __name__ == '__main__':
    canonicals = Canonicals('signaling')
    ppr_ids = [147305678]
    pathway_interactions(ppr_ids)
    sys.exit(0)
