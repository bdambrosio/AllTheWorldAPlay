import re
from typing import List

def _validate_tag(tag):
    """Validate tag contains only letters a-zA-Z and underscores"""
    if not tag or not isinstance(tag, str):
        raise ValueError("Tag must be a non-empty string")
    if not re.match(r'^[a-zA-Z][a-zA-Z_]*$', tag):
        raise ValueError("Tag must start with a letter and contain only letters and underscores")
    return tag.lower()

def findall(tag, text):
    """Find all occurrences of hash-tagged sections in text.
    Args:
        tag: Tag name (letters a-zA-Z only)
        text: String containing hash-formatted text
    Returns:
        List of matching text sections
    """
    if not text or not isinstance(text, str):
        return []
        
    tag = _validate_tag(tag)
    results = []
    lines = text.split('\n')
    current_section = []
    in_section = False
    
    # First try to find multi-line sections
    for line in lines:
        stripped = line.strip()
        if stripped == f'#{tag}':  # Start of new section
            if in_section:  # Previous section wasn't terminated
                results.append('\n'.join(current_section))
            in_section = True
            current_section = []
        elif stripped == '##':  # End of current section
            if in_section:
                results.append('\n'.join(current_section))
                current_section = []
            in_section = False
        elif in_section:
            current_section.append(line)
            
    # Handle case where last section isn't terminated
    if in_section and current_section:
        results.append('\n'.join(current_section))

    # If no multi-line sections found, look for single-line tags
    if not results:
        tag_marker = f'#{tag} '
        for line in lines:
            stripped = line.strip()
            if stripped.startswith(tag_marker):
                results.append(stripped[len(tag_marker):])
                
    return results

def hasTag(tag, text):
    """Check if tag is present in text.
    Args:
        tag: Tag name (letters a-zA-Z only)
        text: String containing hash-formatted text
    Returns:
        True if tag is present, False otherwise
    """
    if not text or not isinstance(text, str):
        return False
        
    tag = _validate_tag(tag)
    return f'#{tag} ' in text.lower()

def find(tag, text):
    """Find content of first occurrence of tag in text.
    Args:
        tag: Tag name (letters a-zA-Z only)
        text: String containing hash-formatted text
    Returns:
        Content after tag if found, empty string if not found
    """
    if not text or not isinstance(text, str):
        return ""
        
    tag = _validate_tag(tag)
    lines = text.split('\n')
    tag_marker = f'#{tag}'
    
    for line in lines:
        stripped = line.strip().lower()
        if stripped.startswith(tag_marker + ' '):
            # Use original line for result to preserve case of content
            return line.strip()[len(tag_marker)+1:]
    return ""

def set(tag, form, value):
    """Replace content of tag in form with new value.
    Args:
        tag: Tag name (letters a-zA-Z only)
        form: Original hash-formatted text
        value: New value for tag
    Returns:
        Updated text with replaced tag content
    """
    if not form:
        raise ValueError("Form cannot be empty")
        
    tag = _validate_tag(tag)
    lines = form.split('\n')
    tag_marker = f'#{tag}'
    
    for i, line in enumerate(lines):
        if line.strip().lower().startswith(tag_marker + ' '):
            lines[i] = f'#{tag} {value}'
            return '\n'.join(lines)
            
    raise ValueError(f'Tag {tag} not found in form')

def format_hash(text, indent=0):
    """Format hash-marked text with proper indentation.
    Args:
        text: Hash-formatted text to format
        indent: Number of spaces for indentation
    Returns:
        Formatted text with consistent indentation
    """
    if not text or not isinstance(text, str):
        return ""
        
    lines = text.split('\n')
    formatted = []
    
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('#'):
            formatted.append(' ' * indent + stripped)
        else:
            formatted.append(' ' * (indent + 2) + stripped)
            
    return '\n'.join(formatted)

def get_text(text):
    """Extract text content from within hash marks.
    Args:
        text: String that may contain hash-marked sections
    Returns:
        String with hash marks and tags removed
    """
    if not text or not isinstance(text, str):
        return ""
        
    lines = text.split('\n')
    content = []
    
    for line in lines:
        stripped = line.strip()
        if not stripped.startswith('#'):
            content.append(stripped)
            
    return ' '.join(content).strip()

def add(tag, text, value):
    """Add new tag and value to text.
    Args:
        tag: Tag name (letters a-zA-Z only)
        text: Original hash-formatted text
        value: Value for new tag
    Returns:
        Updated text with new tag-value pair
    """
    if not text:
        return f'#{tag} {value}\n##'
        
    tag = _validate_tag(tag)
    
    # If text ends with ##, insert new tag before it
    if text.strip().endswith('##'):
        base = text.strip()[:-2].rstrip()
        return f'{base}\n#{tag} {value}\n##'
    
    # Otherwise just append tag and ##
    return f'{text}\n#{tag} {value}\n##'

def clean(text: str) -> str:
    """Clean up common formatting issues in hash-formatted LLM responses.
    Handles cases like missing newlines before ##, multiple ##, or missing ##.
    Standardizes form separation using ## markers.
    
    Args:
        text: String containing hash-formatted text with potential formatting issues
    Returns:
        Cleaned text with proper hash format, forms separated by ## markers
    """
    if not text or not isinstance(text, str):
        return ""
        
    # Split into lines but preserve empty lines initially
    lines = [line.strip() for line in text.split('\n')]
    
    # Handle ## attached to any line
    cleaned_lines = []
    for line in lines:
        if line.endswith('##'):
            # Split at ## and add parts
            cleaned_lines.append(line[:-2].strip())
            cleaned_lines.append('##')
        else:
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)

def findall_forms(text: str) -> List[str]:
    """Find all hash-formatted forms in text.
    A form starts with a #key and ends with ## or blank line.
    Args:
        text: String containing hash-formatted text
    Returns:
        List of form strings including start tag and ## terminator
    """
    if not text or not isinstance(text, str):
        return []
        
    # First clean the text to standardize format
    text = clean(text)
    
    forms = []
    current_form = []
    in_form = False
    
    lines = text.split('\n')
    for line in lines:
        stripped = line.strip()
        if len(stripped) == 0:
            continue
        if not in_form and stripped.startswith('#') and not stripped == '##':
            # Start of new form
            in_form = True
            current_form = [stripped]  # Keep original line with whitespace
            formTags = {stripped[1:].split(' ')[0]: True}
        elif in_form:
            tag = stripped[1:].split(' ')[0] if stripped.startswith('#') else None
            if stripped == '##':
                # End current form
                current_form.append('##')
                forms.append('\n'.join(current_form))
                current_form = []
                in_form = False
            elif tag and tag in formTags:
                # End current form and start new one
                current_form.append('##')
                forms.append('\n'.join(current_form))
                # Start new form
                current_form = [stripped]
                formTags = {tag: True}
            else:
                current_form.append(stripped)
                if tag:
                    formTags[tag] = True
    if in_form and current_form:
        current_form.append('##')  # Add terminator if missing
        forms.append('\n'.join(current_form))
        
    return forms

if __name__ == "__main__":
    text = """
#goal Balance Work Life
#description Find harmony in life
#otherActorName None
#signalCluster_id sc17
#termination Achieved life balance

#goal Resolve Ambition
#description Weigh career options carefully
#otherActorName Elijah
#signalCluster_id sc11
#termination Made informed decision

#goal Pursue Artistic Goal
#description Showcase art in gallery
#otherActorName None
#signalCluster_id sc14
#termination Exhibition scheduled
"""
    #print(clean(text))
    print('\n\n'.join(findall_forms(text)))
    print(len(findall_forms(text)))
