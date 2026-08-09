# src/scraper.py
import json
import requests
from bs4 import BeautifulSoup
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import config

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        " (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}

def fetch_patent(doc_number):
    # Google Patents URL for US patents
    url = f"https://patents.google.com/patent/US{doc_number}A1/en"
    response = requests.get(url, headers=HEADERS)
    if response.status_code != 200:
        return None
    soup = BeautifulSoup(response.text, "html.parser")

    # title: meta fallback
    title_meta = soup.find("meta", {"name": "DC.title"})
    if title_meta:
        title = title_meta["content"].strip()
    else:
        tag = soup.find("span", itemprop="title")
        title = tag.get_text(strip=True) if tag else ""

    # abstract: section fallback to meta
    abstract_section = soup.find("section", {"itemprop": "abstract"})
    if abstract_section:
        abstract = abstract_section.get_text(separator=" ", strip=True)
    else:
        meta = soup.find("meta", {"name": "DC.description"})
        abstract = meta["content"].strip() if meta else ""

    # claims
    claims = []
    claims_section = soup.find("section", {"itemprop": "claims"})
    if claims_section:
        for claim in claims_section.find_all("div", class_="claim"):
            claims.append(claim.get_text(separator=" ", strip=True))

    # classification
    classification_codes = set()
    for code_elem in soup.select('[itemprop="code"]'):
        code_text = code_elem.get_text(strip=True)
        if code_text:
            classification_codes.add(code_text)

    # return same structure as in the JSON files, with empty detailed_description and bibtex
    return {
        "doc_number": doc_number,
        "title": title,
        "abstract": abstract,
        "claims": claims,
        "detailed_description": [],
        "classification": sorted(list(classification_codes))[0] if classification_codes else "",
        "bibtex": "",
        "filename": "",
    }

def save_patent(patent):
    output_dir = Path(config.DATA_DIR)
    output_file = output_dir / f"new_{patent['doc_number']}.json"

    if output_file.exists():
        with open(output_file, "r") as f:
            existing = json.load(f)
        existing.append(patent)
    else:
        existing = [patent]

    with open(output_file, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"Saved to {output_file}")

