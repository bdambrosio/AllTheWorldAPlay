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
