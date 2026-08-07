# statistics of the patent data, including length of abstract, claims, and detailed description, as well as missing fields.
import sys
from pathlib import Path
import json
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import config

folder_path = Path(config.DATA_DIR)
total_patents = 0
abstract_len =[]
detail_len =[]
claims_len =[]
missing_fields = {"title": 0, "doc_number": 0, "filename": 0, "abstract": 0, "detailed_description": 0, "claims": 0, "bibtex": 0, "classification": 0}

for file_path in folder_path.glob("*.json"):
    with open(file_path,"r") as f:
        data = json.load(f)
        total_patents += len(data)

        for patent in data:
            abstract_curlen = len(patent.get("abstract",""))
            abstract_len.append(abstract_curlen)

            # count length of each claims for embedding
            for claim in patent.get("claims",[]):
                claims_len.append(len(claim))

            desc = " ".join(patent.get("detailed_description", []))
            detail_len.append(len(desc))

            # calculate missing fields
            for field in missing_fields:
                val = patent.get(field)
                if not val or val == [] or val == "":
                    missing_fields[field] += 1
                elif isinstance(val, list) and all(v.strip() == "" for v in val):
                    missing_fields[field] += 1


print(f"Abstract: min={min(abstract_len)}, max={max(abstract_len)}, mean={sum(abstract_len)//len(abstract_len)}")
print(f"Claims: min={min(claims_len)}, max={max(claims_len)}, mean={sum(claims_len)//len(claims_len)}")
print(f"Detailed_description: min={min(detail_len)}, max={max(detail_len)}, mean={sum(detail_len)//len(detail_len)}")
print(f"Total patents: {total_patents}")
for field, count in missing_fields.items():
    print(f"Missing {field}: {count} ({count*100//total_patents}%)")


# Abstract: min=62, max=1768, mean=694
# Claims: min=5, max=4829, mean=292
# Detailed_description: min=0, max=185157, mean=18531
# Total patents: 640
# Missing title: 0 (0%)
# Missing doc_number: 0 (0%)
# Missing filename: 0 (0%)
# Missing abstract: 0 (0%)
# Missing detailed_description: 119 (18%)
# Missing claims: 0 (0%)
# Missing bibtex: 0 (0%)
# Missing classification: 0 (0%)
    



