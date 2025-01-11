import os, sys, logging, glob, time
import torch
import traceback
from PyQt5.QtWidgets import QPushButton, QHBoxLayout, QComboBox, QLabel, QSpacerItem, QApplication, QCheckBox

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import webbrowser
import library.rewrite as rw
#import grobid
# used for title matching
#from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.metrics.pairwise import cosine_similarity
import requests, json
from lxml import etree
url = "http://localhost:8070/api/processFulltextDocument"

headers = {"Response-Type": "application/xml"}#, "Content-Type": "multipart/form-data"}
# Define the namespace map to handle TEI namespaces
ns = {'tei': 'http://www.tei-c.org/ns/1.0'}

def process_xml_table(table):
    start_row_index = -1
    # Try to extract headers using <th> elements; if not present, use the first row's <td> elements
    th_elements = table.xpath('.//tei:tr[1]/tei:th', namespaces=ns)
    if th_elements and len(th_elements)>0 :
        headers = [th.text for th in th_elements]
        start_row_index = 1  # Start from the first row for data rows if headers were found in <th>
    else:
        # If no <th> elements, use the first row's <cell> for headers
        headers = [cell.text for cell in table.xpath('.//tei:row[1]/tei:cell', namespaces=ns)]
        start_row_index = 2  # Start from the second row for data rows, since headers are in the first
        
    #print(f'start_row {start_row_index} headers {headers}')
    # Initialize an empty list to hold each row's data as a dictionary
    table_data = []
        
    # Iterate over each row in the table, excluding the header row
    for row in table.xpath(f'.//tei:row[position()>={start_row_index}]', namespaces=ns):
        # Extract text from each cell (<td>) in the row
        row_data = [cell.text for cell in row.xpath('.//tei:cell', namespaces=ns)]
        if len(row_data) != len(headers): # don't try to process split rows
            continue
        # Create a dictionary mapping each header to its corresponding cell data
        row_dict = dict(zip(headers, row_data))
        table_data.append(row_dict)
            
    # Convert the table data to JSON
    if len(table_data) >0 : # don't store tables with no rows (presumably result of no rows matching headers len)
        #json_data = json.dumps(table_data, indent=4)
        return table_data
    else:
        return None

def parse_pdf(pdf_filepath):
    print(f'grobid parse_pdf {pdf_filepath}')
    max_section_len = 3072 # chars, about 1k tokens
    files= {'input': open(str(pdf_filepath), 'rb')}
    extract = {"title":'', "authors":'', "abstract":'', "sections":[], "tables": []}
    response = requests.post(url, files=files)
    if response.status_code == 200:
        with open('test.tei.xml', 'w') as t:
            t.write(response.text)
    else:
        print(f'grobid error {response.status_code}, {response.text}')
        return None
    # Parse the XML
    xml_content = response.text
    ns = {'tei': 'http://www.tei-c.org/ns/1.0'}
    tree = etree.fromstring(xml_content.encode('utf-8'))
    # Extract title
    title = tree.xpath('.//tei:titleStmt/tei:title[@type="main"]', namespaces=ns)
    title_text = title[0].text if title else 'Title not found'
    extract["title"]=title_text

    # Extract authors
    authors = tree.xpath('.//tei:teiHeader//tei:fileDesc//tei:sourceDesc/tei:biblStruct/tei:analytic/tei:author/tei:persName', namespaces=ns)
    authors_list = [' '.join([name.text for name in author.xpath('.//tei:forename | .//tei:surname', namespaces=ns)]) for author in authors]
    extract["authors"]=', '.join(authors_list)

    # Extract abstract
    abstract = tree.xpath('.//tei:profileDesc/tei:abstract//text()', namespaces=ns)
    abstract_text = ''.join(abstract).strip()
    extract["abstract"]=abstract_text

    # Extract major section titles
    # Note: Adjust the XPath based on the actual TEI structure of your document
    section_titles = tree.xpath('./tei:text/tei:body/tei:div/tei:head', namespaces=ns)
    titles_list = [title.text for title in section_titles]
    body_divs = tree.xpath('./tei:text/tei:body/tei:div', namespaces=ns)
    pp_divs = tree.xpath('./tei:text/tei:body/tei:p', namespaces=ns)
    figures = tree.xpath('./tei:text/tei:body/tei:figure', namespaces=ns)
    pdf_tables = []
    for figure in figures:
        # Retrieve <table> elements within this <figure>
        # Note: Adjust the XPath expression if the structure is more complex or different
        tables = figure.xpath('./tei:table', namespaces=ns)
        if len(tables) > 0:
            pdf_tables.append(process_xml_table(tables[0]))
    extract['tables'] = pdf_tables
    sections = []
    for element in body_divs:
        #print(f"\nprocess div chars: {len(' '.join(element.xpath('.//text()')))}")
        head_texts = element.xpath('./tei:head//text()', namespaces=ns)
        all_text = element.xpath('.//text()')
        # don't need following anymore, concatenating text segments with '\n'
        #for head in head_texts:
        #    #print(f'  process head {head}')
        #    for t, text in enumerate(all_text):
        #        if head == text:
        #            #print(f'  found head  in all_text {t}')
        #            all_text[t] = head+'\n'

        # Combine text nodes into a single string
        combined_text = '\n'.join(all_text)
        if len(combined_text) < 24:
            continue
        if len(combined_text) > max_section_len: #break into chunks on pp
            pps = ''
            for text in all_text:
                if len(pps) + len(text) < max_section_len:
                    pps += '\n'+text
                elif len(pps) > int(.5* max_section_len):
                    sections.append(pps)
                    pps = text
                else:
                    sections.append(pps+'\n'+text)
                    pps = ''
        else:
            sections.append(combined_text)
                    
    extract["sections"] = sections
    print(f'title: {title_text}')
    #print(f"Abstract: {len(abstract_text)} chars, Section count: {len(body_divs)}, tables: {len(pdf_tables)}, max_section_len: {max_section_len}")
    return extract

def get_title(title):
    title = title.strip()
    try:
        url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={title}&fields=url,title,year,abstract,authors,citationStyles,citationCount,influentialCitationCount,isOpenAccess,openAccessPdf,s2FieldsOfStudy"
        headers = {'x-api-key':ssKey, }
        response = requests.get(url, headers = headers)
        if response.status_code != 200:
            print(f'SemanticsSearch fail code {response.status_code}')
            return None
        results = response.json()
        #print(f' s2 response keys {results.keys()}')
        total_papers = results["total"]
        if total_papers == 0:
            return None
        papers = results["data"]
        #print(f'get article search returned first {len(papers)} papers of {total_papers}')
        for paper in papers:
            paper_title = paper["title"].strip()
            print(f'considering {paper_title}')
            if paper_title.startswith(title):
                #data = get_semantic_scholar_meta(paper['paperId'])
                print(f'title meta-data {paper.keys()}')
                return paper
    except Exception as e:
        traceback.print_exc()

def reform_strings(strings, min_length=32, max_length=2048):
    """
    Combine sequences of shorter strings in the list, ensuring that no string in the resulting list 
    is longer than max_length characters, unless it existed in the original list.
    also discards short strings.
    """
    combined_strings = []
    current_string=''
    for string in strings:
        if len(string) < 24:
            continue
        if not current_string:
            current_string = string
        elif len(current_string) + len(string) > max_length:
            combined_strings.append(current_string)
            current_string = string
        else:
            current_string += ('\n' if len(current_string)>0 else '') + string
    if current_string:
        combined_strings.append(current_string)
    return combined_strings
    
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

