"""
Multi-Domain Support Triage Agent - Main Entry Point

Usage:
    cd hackathon
    python code/main.py                           # Process support_tickets.csv
    python code/main.py --sample                  # Process sample_support_tickets.csv (for testing)
    python code/main.py --rebuild-index            # Force rebuild the FAISS index
"""

import csv
import sys
import os
import time
import argparse
from pathlib import Path

# Force UTF-8 output on Windows
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure code/ is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from config import (
    INPUT_CSV, SAMPLE_CSV, OUTPUT_CSV, REPO_ROOT,
    GEMINI_API_KEY, RANDOM_SEED,
)
from corpus_loader import load_corpus
from chunker import chunk_corpus
from embedder import build_index
from retriever import Retriever
from classifier import triage_ticket
from logger import log_session_start, log_agent_run

console = Console()


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Domain Support Triage Agent")
    parser.add_argument("--sample", action="store_true", help="Run on sample_support_tickets.csv instead")
    parser.add_argument("--rebuild-index", action="store_true", help="Force rebuild the FAISS index")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed")
    return parser.parse_args()


def read_tickets(csv_path: Path) -> list[dict]:
    """Read support tickets from CSV."""
    tickets = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tickets.append({
                "issue": row.get("Issue", row.get("issue", "")).strip(),
                "subject": row.get("Subject", row.get("subject", "")).strip(),
                "company": row.get("Company", row.get("company", "")).strip(),
            })
    return tickets


def write_output(results: list[dict], output_path: Path):
    """Write triage results to CSV."""
    fieldnames = ["issue", "subject", "company", "response", "product_area", "status", "request_type", "justification"]
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def normalize_company(company: str) -> str:
    """Normalize company name to ecosystem identifier."""
    if not company or company.lower() in ("none", "nan", ""):
        return None
    
    c = company.strip().lower()
    if "hackerrank" in c:
        return "hackerrank"
    elif "claude" in c:
        return "claude"
    elif "visa" in c:
        return "visa"
    return None


def main():
    args = parse_args()
    
    # ── Banner ──────────────────────────────────────────────────────
    console.print(Panel.fit(
        "[bold cyan]Multi-Domain Support Triage Agent[/bold cyan]\n"
        "[dim]HackerRank | Claude | Visa[/dim]",
        border_style="cyan",
    ))
    
    # ── Validate API key ────────────────────────────────────────────
    if not GEMINI_API_KEY:
        console.print("[red]ERROR: GEMINI_API_KEY not set. Create code/.env with your key.[/red]")
        sys.exit(1)
    
    console.print("[green][OK][/green] Gemini API key loaded")
    
    # ── Log session start ───────────────────────────────────────────
    log_session_start(repo_root=str(REPO_ROOT))
    
    # ── Step 1: Load corpus ─────────────────────────────────────────
    console.print("\n[bold]Step 1:[/bold] Loading support corpus...")
    docs = load_corpus()
    console.print(f"  [green][OK][/green] Loaded {len(docs)} articles")
    
    for eco in ["hackerrank", "claude", "visa"]:
        count = sum(1 for d in docs if d.ecosystem == eco)
        console.print(f"    {eco}: {count} articles")
    
    # ── Step 2: Chunk corpus ────────────────────────────────────────
    console.print("\n[bold]Step 2:[/bold] Chunking documents...")
    chunks = chunk_corpus(docs)
    console.print(f"  [green][OK][/green] Created {len(chunks)} chunks")
    
    # ── Step 3: Build/load FAISS index ──────────────────────────────
    console.print("\n[bold]Step 3:[/bold] Building embedding index...")
    index, chunks_meta = build_index(chunks, force_rebuild=args.rebuild_index)
    retriever = Retriever(index, chunks_meta)
    console.print(f"  [green][OK][/green] Index ready ({index.ntotal} vectors)")
    
    # ── Step 4: Read tickets ────────────────────────────────────────
    input_csv = SAMPLE_CSV if args.sample else INPUT_CSV
    console.print(f"\n[bold]Step 4:[/bold] Reading tickets from {input_csv.name}...")
    tickets = read_tickets(input_csv)
    console.print(f"  [green][OK][/green] Read {len(tickets)} tickets")
    
    # ── Step 5: Process each ticket ─────────────────────────────────
    console.print(f"\n[bold]Step 5:[/bold] Processing tickets with RAG pipeline...")
    results = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("Processing tickets...", total=len(tickets))
        
        for i, ticket in enumerate(tickets):
            # Build search query
            query = ticket["issue"]
            if ticket["subject"]:
                query = f"{ticket['subject']}: {query}"
            
            # Determine ecosystem filter
            ecosystem = normalize_company(ticket["company"])
            
            # Retrieve relevant docs
            search_results = retriever.search(
                query=query,
                ecosystem_filter=ecosystem,
            )
            context = retriever.format_context(search_results)
            
            # Classify and generate response
            triage = triage_ticket(
                issue=ticket["issue"],
                subject=ticket["subject"],
                company=ticket["company"],
                context=context,
            )
            
            results.append({
                "issue": ticket["issue"],
                "subject": ticket["subject"],
                "company": ticket["company"],
                "response": triage.response,
                "product_area": triage.product_area,
                "status": triage.status,
                "request_type": triage.request_type,
                "justification": triage.justification,
            })
            
            progress.update(task, advance=1, description=f"Ticket {i+1}/{len(tickets)}")
            
            # Delay to respect free-tier rate limits (~15 RPM)
            time.sleep(4)
    
    # ── Step 6: Write output ────────────────────────────────────────
    output_path = OUTPUT_CSV
    console.print(f"\n[bold]Step 6:[/bold] Writing results to {output_path.name}...")
    write_output(results, output_path)
    console.print(f"  [green][OK][/green] Wrote {len(results)} predictions to {output_path}")
    
    # ── Log the run ─────────────────────────────────────────────────
    log_agent_run(len(results), str(output_path))
    
    # ── Summary table ───────────────────────────────────────────────
    console.print("\n")
    table = Table(title="Triage Summary", show_lines=True)
    table.add_column("#", style="dim", width=3)
    table.add_column("Company", width=12)
    table.add_column("Status", width=10)
    table.add_column("Type", width=16)
    table.add_column("Product Area", width=20)
    table.add_column("Response Preview", width=50)
    
    for i, r in enumerate(results, 1):
        status_style = "green" if r["status"] == "replied" else "yellow"
        table.add_row(
            str(i),
            r["company"] or "-",
            f"[{status_style}]{r['status']}[/{status_style}]",
            r["request_type"],
            r["product_area"],
            r["response"][:50] + "..." if len(r["response"]) > 50 else r["response"],
        )
    
    console.print(table)
    console.print(f"\n[bold green]Done![/bold green] Output written to: {output_path}")


if __name__ == "__main__":
    main()
