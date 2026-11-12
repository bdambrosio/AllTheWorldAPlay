"""Small judgement calls answered in JSON, after the style of Cognitive_workbench
(src/utils/json_utils.py repair_json_string, the probes in src/chat/).

The prompt asks for JSON in plain words and shows the shapes; the reply is parsed tolerantly;
the caller checks the keys against what it allows and, on a reply it cannot read, takes the
default it chose on purpose (no retry). Meaning is decided by the model, never by matching
words in the text it judged (Bruce, 2026-11-11: "NO KEYWORD MATCHING ... they never work").
"""
from __future__ import annotations

import json
import re


def parse(text):
    """The first JSON object (or list) in `text`, or None. Tolerates code fences, prose around
    it, a trailing comma, and a missing closing brace at the end of a reply."""
    if not text:
        return None
    body = text.replace('```json', '').replace('```', '').strip()
    try:
        return json.loads(body)
    except ValueError:
        pass
    start = min([i for i in (body.find('{'), body.find('[')) if i >= 0], default=-1)
    if start < 0:
        return None
    body = body[start:]
    for attempt in (body, body[:body.rfind('}') + 1] if '}' in body else body):
        candidate = re.sub(r',\s*([}\]])', r'\1', attempt)
        try:
            return json.loads(candidate)
        except ValueError:
            pass
    # a reply cut short: close what is open
    depth_brace = body.count('{') - body.count('}')
    depth_list = body.count('[') - body.count(']')
    if depth_brace >= 0 and depth_list >= 0:
        try:
            return json.loads(re.sub(r',\s*$', '', body.rstrip()) + ']' * depth_list + '}' * depth_brace)
        except ValueError:
            return None
    return None


def items(reply, key, allowed_kinds=None, fields=('kind', 'words')):
    """reply[key] as a list of dicts that have the given fields (strings, non-empty), with 'kind'
    lower-cased and kept only if it is one of allowed_kinds."""
    out = []
    for entry in (reply or {}).get(key) or [] if isinstance(reply, dict) else []:
        if not isinstance(entry, dict):
            continue
        values = {f: str(entry.get(f) or '').strip() for f in fields}
        if not all(values.values()):
            continue
        if 'kind' in values:
            values['kind'] = values['kind'].lower()
            if allowed_kinds is not None and values['kind'] not in allowed_kinds:
                continue
        out.append(values)
    return out
