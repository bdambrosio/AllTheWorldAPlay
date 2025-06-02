import sys
import re
sys.path.append('src')

# Read lost.py content
with open('src/plays/lost.py', 'r') as f:
    lost_content = f.read()

print("=== Testing setting extraction ===")

# Try different patterns
patterns = [
    r'W = context\.Context\([^,]+,\s*"""([^"]+?)"""\s*,',
    r'W = context\.Context\([^,]+,\s*"""(.*?)"""\s*,',
    r'W = context\.Context\(\[[^\]]+\],\s*"""(.*?)"""\s*,',
    r'W = context\.Context\(\[[^\]]+\],\s*"""(.*?)""",',
]

for i, pattern in enumerate(patterns):
    print(f"Pattern {i+1}: {pattern}")
    context_match = re.search(pattern, lost_content, re.DOTALL)
    if context_match:
        setting = context_match.group(1).strip()
        print(f"  SUCCESS! Found setting ({len(setting)} chars)")
        print(f"  Setting: {repr(setting[:100])}...")
        break
    else:
        print("  No match")

print("\n=== Context creation section ===")
# Look for Context creation
context_lines = []
in_context = False
for line in lost_content.split('\n'):
    if 'W = context.Context' in line:
        in_context = True
    if in_context:
        context_lines.append(line)
        if 'server_name=server_name)' in line:
            break

print('\n'.join(context_lines))

print("\n=== Testing corrected parse_scenario ===")

def test_parse_scenario(scenario_content):
    if isinstance(scenario_content, list):
        content = '\n'.join(scenario_content)
    else:
        content = scenario_content
    
    # Extract setting from Context creation - CORRECTED PATTERN
    context_pattern = r'W = context\.Context\(\[[^\]]+\],\s*"""(.*?)"""\s*,'
    context_match = re.search(context_pattern, content, re.DOTALL)
    setting = context_match.group(1).strip() if context_match else ""
    
    characters = []
    
    # Find all NarrativeCharacter definitions
    char_pattern = r'(\w+)\s*=\s*NarrativeCharacter\(\s*"([^"]+)"\s*,\s*"""([^"]+?)"""\s*,'
    char_matches = re.finditer(char_pattern, content, re.DOTALL)
    
    for char_match in char_matches:
        var_name = char_match.group(1)
        char_name = char_match.group(2)
        char_description = char_match.group(3).strip()
        
        # Find drives for this character
        drives_pattern = rf'\b{re.escape(var_name)}\.set_drives\(\[\s*((?:"[^"]*",?\s*)+)\]\)'
        drives_match = re.search(drives_pattern, content, re.DOTALL)
        
        drives = ""
        if drives_match:
            drives_text = drives_match.group(1)
            drive_strings = re.findall(r'"([^"]*)"', drives_text)
            drives = '. '.join(drive_strings)
        
        characters.append({
            "name": char_name,
            "description": char_description,
            "drives": drives
        })
    
    return {
        "setting": setting,
        "characters": characters
    }

result = test_parse_scenario(lost_content)

print(f"Setting ({len(result['setting'])} chars):")
print(f"'{result['setting']}'")
print()
print("Characters:")
for char in result['characters']:
    print(f"  Name: {char['name']}")
    print(f"  Drives: {char['drives'][:100]}...")
    print() 