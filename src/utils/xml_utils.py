import xml.dom.minidom
# a couple of utils for retrieving data from XML
# ridiculous computationally, but LLMs seem much more robust at producing XML than JSON.

def findall(key, form):
    """ find multiple occurrences of an xml field in a string """
    idx = 0
    items = []
    forml = form.lower()
    keyl = key.lower()
    keyle = keyl[0] + '/' + keyl[1:]
    while idx < len(forml):
        start_idx = forml[idx:].find(keyl)
        if start_idx == -1:
            return items
        start_idx += len(keyl)
        end_idx = forml[idx + start_idx:].find(keyle)
        if end_idx == -1:
            return items
        items.append(form[idx + start_idx:idx + start_idx + end_idx].strip())
        idx += start_idx + end_idx + len(keyle)
    return items


def find(key, form):
    """ find first occurrences of an xml field in a string """
    if form is None:
        raise ValueError('None!')
    forml = form.lower()
    keyl = key.lower()
    keyle = keyl[0] + '/' + keyl[1:]
    start_idx = forml.find(keyl)
    if start_idx == -1:
        return None
    start_idx += len(keyl)
    end_idx = forml[start_idx:].find(keyle)
    if end_idx == -1:
        return form[start_idx:]
    return form[start_idx: start_idx + end_idx]


def set(key, form, value):
    current = find(key, form)
    forml = form.lower()
    keyl = key.lower()
    keyle = keyl[0] + '/' + keyl[1:]
    if current:
        new_form = form.replace(keyl+current+keyle, keyl+value+keyle)
    else:
        new_form = form+keyl+value+keyle
    return new_form

def format_xml(xml_string, indent=2):
    """
    Print a formatted version of the given XML string with specified indentation.

    :param xml_string: The XML string to format
    :param indent: Number of spaces for each indentation level (default: 2)
    """
    dom = xml.dom.minidom.parseString(xml_string)
    pretty_xml = dom.toprettyxml(indent=' ' * indent)

    # Remove empty lines
    lines = pretty_xml.split('\n')
    non_empty_lines = [line for line in lines if line.strip()]
    formatted_xml = '\n'.join(non_empty_lines)
    return formatted_xml


def print_formatted_xml(xml_string, indent=2):
    print(format_xml(xml_string, indent))

