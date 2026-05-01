"""
Logger — Appends structured entries to the hackathon log file
per AGENTS.md §5 format. Redacts secrets automatically.
"""

import re
from datetime import datetime, timezone
from pathlib import Path

from config import LOG_DIR, LOG_FILE


def _ensure_log_file():
    """Create log directory and file if they don't exist."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if not LOG_FILE.exists():
        LOG_FILE.touch()


def _redact_secrets(text: str) -> str:
    """Replace potential API keys and secrets with [REDACTED]."""
    # Common API key patterns
    patterns = [
        r'(AIza[A-Za-z0-9_-]{35})',                    # Google API keys
        r'(sk-[A-Za-z0-9]{48,})',                       # OpenAI keys
        r'(sk-ant-[A-Za-z0-9-]{80,})',                  # Anthropic keys
        r'([A-Za-z0-9]{32,})',                           # Generic long tokens (be careful)
        r'(Bearer\s+[A-Za-z0-9._-]+)',                  # Bearer tokens
        r'(password\s*[:=]\s*\S+)',                     # Passwords
    ]
    
    result = text
    # Only redact things that look like actual keys (long alphanumeric strings)
    result = re.sub(r'AIza[A-Za-z0-9_-]{35}', '[REDACTED]', result)
    result = re.sub(r'sk-[A-Za-z0-9]{20,}', '[REDACTED]', result)
    result = re.sub(r'sk-ant-[A-Za-z0-9-]{20,}', '[REDACTED]', result)
    
    return result


def _timestamp() -> str:
    """ISO-8601 timestamp with timezone."""
    return datetime.now().astimezone().isoformat()


def log_session_start(agent_name: str = "Antigravity", repo_root: str = "", branch: str = "main", language: str = "py"):
    """Log a session start entry per §5.1."""
    _ensure_log_file()
    
    entry = f"""
## [{_timestamp()}] SESSION START

Agent: {agent_name}
Repo Root: {repo_root}
Branch: {branch}
Worktree: main
Parent Agent: none
Language: {language}
Time Remaining: calculating...

"""
    with open(LOG_FILE, "a", encoding="utf-8", newline="\n") as f:
        f.write(entry)


def log_turn(title: str, user_prompt: str, response_summary: str, actions: list[str], 
             agent_name: str = "Antigravity", repo_root: str = "", branch: str = "main"):
    """Log a per-turn entry per §5.2."""
    _ensure_log_file()
    
    # Truncate title to 80 chars
    title = title[:80]
    
    # Redact secrets from user prompt
    safe_prompt = _redact_secrets(user_prompt)
    
    actions_str = "\n".join(f"* {a}" for a in actions) if actions else "* No file changes"
    
    entry = f"""
## [{_timestamp()}] {title}

User Prompt (verbatim, secrets redacted):
{safe_prompt}

Agent Response Summary:
{response_summary}

Actions:
{actions_str}

Context:
tool={agent_name}
branch={branch}
repo_root={repo_root}
worktree=main
parent_agent=none

"""
    with open(LOG_FILE, "a", encoding="utf-8", newline="\n") as f:
        f.write(entry)


def log_agent_run(ticket_count: int, output_path: str):
    """Log the agent's processing run."""
    _ensure_log_file()
    
    entry = f"""
## [{_timestamp()}] Agent Run: Processed {ticket_count} tickets

Agent Response Summary:
Triage agent processed {ticket_count} support tickets and wrote results to {output_path}.

Actions:
* Loaded and indexed support corpus
* Processed {ticket_count} tickets through RAG pipeline
* Wrote predictions to {output_path}

Context:
tool=SupportTriageAgent
branch=main
worktree=main
parent_agent=none

"""
    with open(LOG_FILE, "a", encoding="utf-8", newline="\n") as f:
        f.write(entry)
