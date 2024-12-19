# -*- coding: UTF-8 -*-
import json 
with open('train.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    with open('train.jsonl', 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


