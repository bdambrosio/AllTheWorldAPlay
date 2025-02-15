import re

def _validate_tag(tag):
    """Validate tag contains only letters a-zA-Z"""
    if not tag or not isinstance(tag, str):
        raise ValueError("Tag must be a non-empty string")
    if not re.match(r'^[a-zA-Z]+$', tag):
        raise ValueError("Tag must contain only letters a-zA-Z")
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
    
    # Handle both full sections (#plan...##) and individual tags (#name etc)
    if tag == 'plan':
        # Look for full plan sections
        for line in lines:
            stripped = line.strip()
            if stripped.lower() == '#' + tag:  # Start of new plan
                if in_section:  # Previous plan wasn't terminated with ##
                    results.append('\n'.join(current_section))
                in_section = True
                current_section = []
            elif stripped == '##':  # End of current plan
                if in_section:
                    results.append('\n'.join(current_section))
                    current_section = []
                in_section = False
            elif in_section:
                current_section.append(line.rstrip())
        # Handle case where last section isn't terminated
        if in_section and current_section:
            results.append('\n'.join(current_section))
    else:
        # Look for specific tags anywhere
        tag_marker = f'#{tag} '
        for line in lines:
            stripped = line.strip().lower()
            if stripped.startswith(tag_marker):
                results.append(line.strip()[len(tag_marker):])
    return results

def hasKey(tag, text):
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
    return tag in text

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