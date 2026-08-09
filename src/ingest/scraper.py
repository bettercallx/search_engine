# Fetch a single patent from Google Patents and save it as new_<doc>.json for loader.py to pick up
import json
import requests
from bs4 import BeautifulSoup
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import config
 
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        " (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}
 
# US application/grant kind codes to try, in order.
# Applications are usually A1; granted patents are B1/B2.
KIND_CODES = ["A1", "B2", "B1", "A2", ""]
 
REQUEST_TIMEOUT = 15
 
 
def _get_soup(doc_number):
    """Try the URL with each kind code until one returns 200; return soup or None."""
    for kind in KIND_CODES:
        url = f"https://patents.google.com/patent/US{doc_number}{kind}/en"
        try:
            response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        except requests.RequestException as e:
            print(f"  request error for {url}: {e}")
            continue
        if response.status_code == 200:
            return BeautifulSoup(response.text, "html.parser")
    return None
 
 
def fetch_patent(doc_number):
    soup = _get_soup(doc_number)
    if soup is None:
        return None
 
    # title: meta first, fall back to itemprop span
    title_meta = soup.find("meta", {"name": "DC.title"})
    if title_meta and title_meta.get("content"):
        title = title_meta["content"].strip()
    else:
        tag = soup.find("span", itemprop="title")
        title = tag.get_text(strip=True) if tag else ""
 
    # abstract: section first, fall back to meta description
    abstract_section = soup.find("section", {"itemprop": "abstract"})
    if abstract_section:
        abstract = abstract_section.get_text(separator=" ", strip=True)
    else:
        meta = soup.find("meta", {"name": "DC.description"})
        abstract = meta["content"].strip() if meta and meta.get("content") else ""
 
    # claims: each claim is a div.claim inside the claims section
    claims = []
    claims_section = soup.find("section", {"itemprop": "claims"})
    if claims_section:
        for claim in claims_section.find_all("div", class_="claim"):
            text = claim.get_text(separator=" ", strip=True)
            if text:
                claims.append(text)
 
    # de-dup: Google Patents lists independent claims twice (overview + claims section). 
    # Dedupe by content, preserving order, so each claim maps to one vector in claim_embeddings and claim_index stays correct.
    seen = set()
    unique_claims = []
    for c in claims:
        if c not in seen:
            seen.add(c)
            unique_claims.append(c)
    claims = unique_claims
 
    # classification: codes render as a hierarchy (B -> B60 -> B60B -> B60B7/00-> B60B7/0013) via itemprop="Code" (capital C). 
    # Keep the deepest (longest) code as the single primary classification
    codes = [
        el.get_text(strip=True)
        for el in soup.select('[itemprop="Code"]')
        if el.get_text(strip=True)
    ]
    classification = max(codes, key=len) if codes else ""
 
    # same structure as the JSON dataset; fields not on the page are left empty
    return {
        "doc_number": doc_number,
        "title": title,
        "abstract": abstract,
        "claims": claims,
        "detailed_description": [],
        "classification": classification,
        "bibtex": "",
        "filename": "",
    }
 
 
def save_patent(patent):
    # always overwrite, re-fetching a doc replaces its file instead of appending
    output_dir = Path(config.NEW_DATA_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"new_{patent['doc_number']}.json"
    with open(output_file, "w") as f:
        json.dump([patent], f, indent=2)
    print(f"Saved to {output_file}")