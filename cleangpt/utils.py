import re
import json


### https://github.com/Lightning-AI/litgpt/blob/main/litgpt/utils.py
def fix_and_load_json(s):
    s = re.sub(r',(\s*[}\]])', r'\1', s)

    pattern = r'(?<=[}\]0-9truefalsenull"])\s*(\n\s*)"'
    replacement = r',\1"'

    s = re.sub(pattern, replacement, s)
    
    try:
        return json.loads(s)
    except json.JSONDecodingError as e:
        raise ValueError("Failed to parse json after fixing:", e)
