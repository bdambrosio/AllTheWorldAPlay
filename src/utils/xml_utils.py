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
    if current is not None:
        new_form = form.replace(keyl+current+keyle, keyl+value+keyle)
    else:
        raise ValueError(f'key {key} not found in form {form}')
    return new_form

def format_xml_formal(xml_string, indent=2):
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

class XMLFormatError(Exception):
    """Error formatting XML string"""
    pass

def format_xml(xml_str, indent=0):
    """Format XML string with minimal assumptions about structure.
    Handles nested tags with same name (like sections/section/sections/section).
    """
    if not xml_str or not isinstance(xml_str, str):
        return ''
        
    result = []
    i = 0
    
    while i < len(xml_str):
        # Skip whitespace
        while i < len(xml_str) and xml_str[i].isspace():
            i += 1
        if i >= len(xml_str):
            break
            
        # Find start of tag
        if xml_str[i] != '<':
            i += 1
            continue
            
        # Skip processing instructions and comments
        if xml_str[i:].startswith('<?') or xml_str[i:].startswith('<!--'):
            end = xml_str.find('>', i)
            if end == -1:
                raise XMLFormatError("Unclosed processing instruction or comment")
            i = end + 1
            continue
            
        # Find tag name
        name_end = xml_str.find('>', i)
        if name_end == -1:
            raise XMLFormatError("Unclosed tag")
        tag_name = xml_str[i+1:name_end].strip()
        if not tag_name:
            raise XMLFormatError("Empty tag name")
            
        # Find closing tag by counting nesting
        stack = []
        pos = i
        while pos < len(xml_str):
            open_pos = xml_str.find('<', pos)
            if open_pos == -1:
                break
                
            close_bracket = xml_str.find('>', open_pos)
            if close_bracket == -1:
                raise XMLFormatError("Unclosed tag")
                
            if xml_str[open_pos+1] == '/':  # Closing tag
                close_tag = xml_str[open_pos+2:close_bracket].strip()
                if stack and stack[-1] == close_tag:
                    stack.pop()
                    if not stack and close_tag == tag_name:
                        end_pos = open_pos
                        break
            else:  # Opening tag
                new_tag = xml_str[open_pos+1:close_bracket].strip()
                stack.append(new_tag)
                
            pos = close_bracket + 1
            
        if not stack and end_pos:
            # Extract and format content
            content = xml_str[name_end+1:end_pos].strip()
            if '<' in content and '>' in content:
                result.append(' ' * indent + f'<{tag_name}>')
                result.append(format_xml(content, indent + 2))
                result.append(' ' * indent + f'</{tag_name}>')
            else:
                result.append(' ' * indent + f'<{tag_name}>{content}</{tag_name}>')
                
            i = end_pos + len(tag_name) + 3  # skip past </tag_name>
        else:
            raise XMLFormatError(f"Missing closing tag for {tag_name}")
            
    return '\n'.join(result)



