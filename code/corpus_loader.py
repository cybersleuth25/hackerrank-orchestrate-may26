"""
Corpus Loader — Reads all markdown files from the data/ directory,
parses frontmatter and content, and tags each document with ecosystem
and product_area metadata derived from the file path.
"""

import re
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import List

from config import CORPUS_DIRS


@dataclass
class Document:
    """Represents a single support article from the corpus."""
    content: str
    title: str = ""
    source_url: str = ""
    ecosystem: str = ""          # hackerrank, claude, visa
    product_area: str = ""       # derived from directory structure
    file_path: str = ""          # relative path within data/
    metadata: dict = field(default_factory=dict)


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """
    Extract YAML frontmatter (between --- markers) and return
    (frontmatter_dict, remaining_content).
    """
    fm_match = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, re.DOTALL)
    if fm_match:
        try:
            fm = yaml.safe_load(fm_match.group(1)) or {}
        except yaml.YAMLError:
            fm = {}
        content = text[fm_match.end():]
        return fm, content
    return {}, text


def _derive_product_area(file_path: Path, ecosystem: str) -> str:
    """
    Derive product_area from the directory structure.
    e.g., data/hackerrank/screen/managing-tests/... → 'screen'
          data/claude/privacy-and-legal/... → 'privacy'
          data/visa/support/consumer/... → 'general_support'
    """
    # Get parts relative to the ecosystem directory
    eco_dir = CORPUS_DIRS.get(ecosystem)
    if not eco_dir:
        return "general"
    
    try:
        rel = file_path.relative_to(eco_dir)
    except ValueError:
        return "general"
    
    parts = rel.parts
    
    if len(parts) <= 1:
        # Root-level file like index.md
        return "general"
    
    # First directory level is the product area
    raw_area = parts[0].lower()
    
    # Normalize common patterns
    area_map = {
        # HackerRank
        "screen": "screen",
        "interviews": "interviews",
        "library": "library",
        "settings": "settings",
        "integrations": "integrations",
        "general-help": "general_help",
        "engage": "engage",
        "skillup": "skillup",
        "chakra": "chakra",
        "hackerrank_community": "community",
        "uncategorized": "general",
        # Claude
        "claude": "claude_general",
        "claude-api-and-console": "api",
        "claude-code": "claude_code",
        "claude-desktop": "claude_desktop",
        "claude-for-education": "education",
        "claude-for-government": "government",
        "claude-for-nonprofits": "nonprofits",
        "claude-in-chrome": "chrome_extension",
        "claude-mobile-apps": "mobile",
        "connectors": "connectors",
        "identity-management-sso-jit-scim": "identity_management",
        "privacy-and-legal": "privacy",
        "pro-and-max-plans": "billing",
        "safeguards": "safeguards",
        "team-and-enterprise-plans": "team_enterprise",
        "amazon-bedrock": "bedrock",
        # Visa
        "support": "general_support",
    }
    
    area = area_map.get(raw_area, raw_area.replace("-", "_"))
    
    # For visa, refine based on deeper path
    if ecosystem == "visa" and len(parts) > 1:
        sub = parts[1].lower() if len(parts) > 1 else ""
        if "travel" in sub:
            area = "travel_support"
        elif "consumer" in sub:
            area = "general_support"
        elif "small-business" in sub or "merchant" in sub:
            area = "merchant_support"
        elif "dispute" in sub:
            area = "dispute_resolution"
        elif "fraud" in sub:
            area = "fraud_prevention"
        elif "security" in sub or "data-security" in sub:
            area = "security"

    return area


def load_corpus() -> List[Document]:
    """
    Load all markdown files from the corpus directories.
    Returns a list of Document objects with parsed content and metadata.
    """
    documents = []
    
    for ecosystem, corpus_dir in CORPUS_DIRS.items():
        if not corpus_dir.exists():
            continue
        
        for md_file in sorted(corpus_dir.rglob("*.md")):
            # Skip index files as they're just tables of contents
            if md_file.name == "index.md":
                continue
            
            try:
                raw_text = md_file.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            
            if not raw_text.strip():
                continue
            
            # Parse frontmatter
            frontmatter, content = _parse_frontmatter(raw_text)
            
            # Derive metadata
            title = frontmatter.get("title", "")
            if not title:
                # Try to extract from first heading
                h1_match = re.search(r"^#\s+(.+)", content, re.MULTILINE)
                title = h1_match.group(1).strip() if h1_match else md_file.stem
            
            product_area = _derive_product_area(md_file, ecosystem)
            
            doc = Document(
                content=content.strip(),
                title=title,
                source_url=frontmatter.get("source_url", ""),
                ecosystem=ecosystem,
                product_area=product_area,
                file_path=str(md_file.relative_to(corpus_dir)),
                metadata={
                    "description": frontmatter.get("description", ""),
                    "last_modified": frontmatter.get("last_modified", ""),
                },
            )
            documents.append(doc)
    
    return documents


if __name__ == "__main__":
    docs = load_corpus()
    print(f"Loaded {len(docs)} documents")
    for eco in ["hackerrank", "claude", "visa"]:
        eco_docs = [d for d in docs if d.ecosystem == eco]
        areas = set(d.product_area for d in eco_docs)
        print(f"  {eco}: {len(eco_docs)} docs, areas: {sorted(areas)}")
