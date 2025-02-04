import xml.dom.minidom
# a couple of utils for retrieving data from XML
# ridiculous computationally, but LLMs seem much more robust at producing XML than JSON.

def findall(tag, xml_str):
    """Find all occurrences of tag in xml_str.
    Args:
        tag: Tag to find (with angle brackets)
        xml_str: String containing XML
    Returns:
        List of matching XML strings
    """
    if not xml_str or not isinstance(xml_str, str):
        return []
        
    # Strip angle brackets from tag
    tag_name = tag.strip('<>')
    results = []
    i = 0
    
    while i < len(xml_str):
        # Find start of tag
        start = xml_str.find(f'<{tag_name}', i)
        if start == -1:
            break
            
        # Find tag end
        tag_end = xml_str.find('>', start)
        if tag_end == -1:
            break
            
        # Check for self-closing tag
        tag_content = xml_str[start+1:tag_end].strip()
        if tag_content.endswith('/') or xml_str[tag_end-1] == '/':
            i = tag_end + 1
            continue
            
        # Find closing tag
        end = xml_str.find(f'</{tag_name}>', tag_end)
        if end == -1:
            break
            
        results.append(xml_str[tag_end + 1:end])
        i = end + len(tag_name) + 3
        
    return results


def find(key, xml_str):
    """Find content of first occurrence of key in xml_str.
    Args:
        key: Tag to find (with angle brackets)
        xml_str: String containing XML
    Returns:
        Content of tag if found, empty string if not found or self-closing
    """
    if not key or not xml_str:
        return ""
        
    keyl = key.lower()
    xml_strl = xml_str.lower()
    # Strip angle brackets from search key
    tag = keyl.strip('<>')
    
    # Find start of tag
    start = xml_strl.find(f'<{tag}')
    if start == -1:
        return ""
        
    # Find end of opening tag
    tag_end = xml_strl.find('>', start)
    if tag_end == -1:
        return ""
        
    # Check for self-closing tag
    tag_content = xml_strl[start+1:tag_end].strip()
    if tag_content.endswith('/') or xml_strl[tag_end-1] == '/':
        return ""
        
    # Find closing tag
    end = xml_strl.find(f'</{tag}>', tag_end)
    if end == -1:
        # run on tag_end + 1 to end of string
        return xml_str[tag_end + 1:]
        
    return xml_str[tag_end + 1:end]


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
        raise XMLFormatError("Input must be non-empty string")  # Changed from return ''
        
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
            
        # Find tag name and check for self-closing tag
        name_end = xml_str.find('>', i)
        if name_end == -1:
            raise XMLFormatError("Unclosed tag")
        tag_content = xml_str[i+1:name_end].strip()
        if not tag_content:
            raise XMLFormatError("Empty tag name")
            
        # Get tag name (everything before first space if attributes exist)
        tag_name = tag_content.split()[0]
            
        # Check for self-closing tag (ending with /)
        if tag_content.endswith('/') or xml_str[name_end-1] == '/':
            result.append(' ' * indent + xml_str[i:name_end+1])
            i = name_end + 1
            continue
            
        # Find closing tag by counting nesting
        stack = []
        pos = i
        end_pos = None
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
                new_tag = xml_str[open_pos+1:close_bracket].strip().split()[0]
                if not xml_str[close_bracket-1] == '/':  # Not self-closing
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

def get_text(xml_str):
    """Extract text content from within outermost XML tags.
    
    Args:
        xml_str: String that may contain XML tags
        
    Returns:
        String with outermost tags removed, or original string if no tags
    """
    if not xml_str or not isinstance(xml_str, str):
        return ""
        
    # Find first < and last >
    start = xml_str.find('<')
    end = xml_str.rfind('>')
    
    if start == -1 or end == -1 or start >= end:
        return xml_str.strip()
        
    # Extract content between first > and last <
    first_close = xml_str.find('>', start)
    last_open = xml_str.rfind('<', 0, end)
    
    if first_close == -1 or last_open == -1 or first_close >= last_open:
        return xml_str.strip()
        
    return xml_str[first_close + 1:last_open].strip()



