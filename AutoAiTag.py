#!/usr/bin/env python3
"""
AutoAiTag.py

- Menu-driven Dry Run / Metadata Write for Obsidian vaults
- Uses a local LM Studio (OpenAI-compatible) endpoint to generate summary+tags for long notes
- Multithreaded processing with configurable LLM concurrency limiting
- Robust extraction of JSON/YAML from messy LLM responses
- Proper unicode handling (no \u2011 escapes)
- Intelligent date detection with AI, filename, and filesystem fallbacks
"""

import os
import sys
import json
import re
import requests
import yaml
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# ===================================================================
# CONFIGURATION SECTION - CUSTOMIZE THESE SETTINGS FOR YOUR ENVIRONMENT
# ===================================================================
#
# LM STUDIO SETTINGS
# ------------------
# API_URL: Your LM Studio endpoint URL
#   - Default: "http://192.168.1.164:1234/v1/chat/completions"
#   - Change this to match your LM Studio server address and port
#   - Example: "http://localhost:1234/v1/chat/completions"
#   - Example: "http://192.168.1.100:8080/v1/chat/completions"
API_URL = "http://192.168.1.164:1234/v1/chat/completions"

# MODEL_NAME: The language model to use for AI processing
#   - Default: "openai/gpt-oss-20b"
#   - Common alternatives:
#     * "llama2:7b" (faster, less accurate)
#     * "llama2:13b" (balanced)
#     * "llama2:70b" (slower, more accurate)
#     * "codellama:7b" (good for technical content)
#   - Check your LM Studio for available models
MODEL_NAME = "openai/gpt-oss-20b"

# REQUEST_TIMEOUT: Maximum time to wait for LLM response (seconds)
#   - Default: 60 seconds
#   - Increase if you have slower models or network issues
#   - Decrease if you want faster failure detection
REQUEST_TIMEOUT = 60

# SYSTEM_PROMPT: Instructions sent to the AI model
#   - Keep this focused on metadata extraction
#   - Don't change unless you understand the implications
SYSTEM_PROMPT = "You are a metadata assistant. Return only JSON or YAML representing metadata (no extra commentary)."

# PROCESSING SETTINGS
# -------------------
# DEFAULT_CHAR_LIMIT: Minimum characters required for AI processing
#   - Default: 1000 characters
#   - Files shorter than this won't get AI-generated summaries/tags
#   - Increase for better AI results, decrease for faster processing
#   - Recommended range: 500-2000 characters
DEFAULT_CHAR_LIMIT = 1000

# DEFAULT_WORKERS: Number of concurrent file processing threads
#   - Default: 4 workers
#   - Increase for faster processing on multi-core systems
#   - Decrease if you experience memory issues
#   - Recommended: 2-8 workers depending on your system
DEFAULT_WORKERS = 4

# DEFAULT_MAX_CONCURRENT_LLM: Maximum simultaneous AI requests
#   - Default: 2 concurrent LLM requests
#   - IMPORTANT: Don't set higher than your LM Studio can handle
#   - Higher values = faster processing but may overwhelm your LLM server
#   - Lower values = slower but more stable
#   - Recommended: 1-4 depending on your LM Studio server capacity
DEFAULT_MAX_CONCURRENT_LLM = 2

# ===================================================================
# END CONFIGURATION SECTION
# ===================================================================

# Global semaphore to limit concurrent LLM requests
llm_semaphore = None


def extract_json_object(s: str):
    """
    Extract the first balanced JSON object from a string.
    
    This function handles messy LLM outputs that might contain extra text
    before or after the JSON response. It finds the first { } pair and
    extracts everything between them.
    
    Args:
        s: Input string that may contain JSON
        
    Returns:
        The JSON substring if found, None otherwise
        
    Example:
        Input: "Here's the result: {'key': 'value'} Thanks!"
        Output: "{'key': 'value'}"
    """
    if not s:
        return None
    start = s.find('{')
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(s)):
        ch = s[i]
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return s[start:i + 1]
    return None


def clean_text_unicode_escapes(s: str) -> str:
    """If string contains literal escape sequences like '\\u2011', decode them to real unicode characters.
    If decode fails, return original string."""
    if not isinstance(s, str):
        return s
    # quick check for an escaped unicode pattern
    if '\\u' in s or '\\U' in s:
        try:
            return s.encode('utf-8').decode('unicode_escape')
        except Exception:
            return s
    return s


def parse_tags_field(raw) -> list:
    """Normalize a tags field to a list of strings."""
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(t).strip() for t in raw if str(t).strip()]
    if isinstance(raw, str):
        # try YAML parse (handles "['a','b']" or "- a\n- b" etc)
        try:
            parsed = yaml.safe_load(raw)
            if isinstance(parsed, list):
                return [str(t).strip() for t in parsed if str(t).strip()]
        except Exception:
            pass
        # fallback comma-separated
        return [t.strip() for t in raw.split(",") if t.strip()]
    # Other types -> convert to single-item list
    return [str(raw).strip()]


def query_llm_for_summary_and_tags(note_text: str, debug: bool = False) -> dict:
    """
    Query LM Studio to generate metadata for a note.
    
    This function sends the note content to your local LLM and requests:
    - A concise summary (max 50 words)
    - Up to 3 relevant tags
    - Date extraction with confidence scoring
    - Date detection from note content
    
    The function handles:
    - Concurrency limiting via semaphore
    - Robust JSON/YAML parsing from LLM responses
    - Error handling and fallbacks
    - Debug output for troubleshooting
    
    Args:
        note_text: The content of the markdown note
        debug: Enable verbose debug output
        
    Returns:
        Dictionary with keys: summary, tags, date, date_confidence
        
    Note:
        This function requires LM Studio to be running and accessible
        at the URL specified in API_URL configuration.
    """
    # Acquire semaphore if it exists (for concurrency limiting)
    if llm_semaphore:
        if debug:
            print(f"🔒 Acquiring LLM semaphore (current limit: {llm_semaphore._value})")
        llm_semaphore.acquire()
        if debug:
            print(f"✅ Acquired LLM semaphore")
    
    try:
        prompt = (
            "Given the following note text, return a JSON or YAML object with these keys:\n"
            "- summary: a concise summary of maximum 50 words\n"
            "- tags: a list (up to 3) of short keyword tags\n"
            "- date: a date found in the text (YYYY-MM-DD format) if present, null if none found\n"
            "- date_confidence: confidence level (0.0 to 1.0) that the extracted date is accurate\n\n"
            "For date detection, look for dates near the beginning of the text. Only return a date if you are very confident (confidence > 0.9).\n"
            "Return only the object (JSON or YAML), no extra commentary.\n\n"
            + note_text
        )

        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2,
            "max_tokens": 400
        }

        r = requests.post(API_URL, json=payload, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        resp = r.json()
        raw_output = resp["choices"][0]["message"]["content"]
        if debug:
            print("\n--- RAW LLM OUTPUT START ---\n")
            print(raw_output)
            print("\n--- RAW LLM OUTPUT END ---\n")

        # 1) Extract balanced {...} if present
        json_sub = extract_json_object(raw_output)
        parsed = None

        if json_sub:
            # Try YAML loader: YAML is more permissive than strict JSON (handles single quotes, etc.)
            try:
                parsed = yaml.safe_load(json_sub)
            except Exception:
                # last resort: try json.loads
                try:
                    parsed = json.loads(json_sub)
                except Exception:
                    parsed = None

        if parsed is None:
            # Try to parse the entire output as YAML (covers YAML outputs without braces)
            try:
                parsed = yaml.safe_load(raw_output)
            except Exception:
                parsed = None

        if not isinstance(parsed, dict):
            if debug:
                print("⚠️ Unable to parse LLM output into dict; returning empty metadata.")
            return {"summary": "", "tags": [], "date": None, "date_confidence": 0.0}

        # Extract and clean summary
        raw_summary = parsed.get("summary", "") or ""
        raw_summary = clean_text_unicode_escapes(str(raw_summary).strip())

        # Truncate to ~50 words if necessary
        words = raw_summary.split()
        if len(words) > 50:
            raw_summary = " ".join(words[:50]) + "..."

        # Extract tags
        raw_tags = parsed.get("tags", parsed.get("tag", []))
        tags = parse_tags_field(raw_tags)
        # Clean tags
        tags = [clean_text_unicode_escapes(t) for t in tags][:3]  # max 3 returned

        # Extract date and confidence
        raw_date = parsed.get("date")
        date_confidence = parsed.get("date_confidence", 0.0)

        return {"summary": raw_summary, "tags": tags, "date": raw_date, "date_confidence": date_confidence}

    except Exception as e:
        if debug:
            print(f"⚠️ LLM query error: {e}")
        return {"summary": "", "tags": [], "date": None, "date_confidence": 0.0}
    finally:
        # Always release the semaphore if we acquired it
        if llm_semaphore:
            llm_semaphore.release()
            if debug:
                print(f"🔓 Released LLM semaphore")


def extract_date_from_filename(filename: str) -> str:
    """Extract date from filename if it's in (YYYY-MM-DD) format."""
    import re
    # Look for pattern like (2024-01-15) or (2024-1-5)
    pattern = r'\((\d{4})-(\d{1,2})-(\d{1,2})\)'
    match = re.search(pattern, filename)
    if match:
        year, month, day = match.groups()
        # Ensure proper formatting
        month = month.zfill(2)
        day = day.zfill(2)
        return f"{year}-{month}-{day}"
    return None


def determine_file_date(ai_date: str, ai_confidence: float, filename: str, file_creation_date: str) -> tuple:
    """
    Determine the file date using a three-tier fallback system.
    
    This function implements intelligent date detection with multiple fallback methods:
    
    Tier 1 (Highest Priority): AI Date Detection
        - Uses LLM to extract dates from note content
        - Only accepts dates with confidence > 0.9
        - Most accurate when dates are mentioned in the text
        
    Tier 2 (Medium Priority): Filename Date Extraction
        - Looks for dates in filename format: (YYYY-MM-DD)
        - Example: "Meeting Notes (2024-01-15).md"
        - Useful for files with consistent naming conventions
        
    Tier 3 (Fallback): File System Creation Date
        - Uses the file's creation timestamp from the OS
        - Least accurate but always available
        - May not reflect the actual content date
        
    Args:
        ai_date: Date string extracted by AI (or None)
        ai_confidence: AI confidence score (0.0 to 1.0)
        filename: Name of the file
        file_creation_date: File system creation date
        
    Returns:
        Tuple of (date_string, date_source)
        date_string: Date in YYYY-MM-DD format
        date_source: One of 'AI', 'Filename', or 'FileSystem'
    """
    # Tier 1: AI date with high confidence (>0.9)
    if ai_date and ai_confidence and ai_confidence > 0.9:
        return ai_date, "AI"
    
    # Tier 2: Date from filename
    filename_date = extract_date_from_filename(filename)
    if filename_date:
        return filename_date, "Filename"
    
    # Tier 3: File system creation date
    return file_creation_date, "FileSystem"


def read_file_content(path: str) -> str:
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def write_file_atomic(path: str, content: str):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(content)
    os.replace(tmp, path)


def parse_frontmatter_and_body(content: str):
    """Return (frontmatter_dict, body_text, had_frontmatter_bool)."""
    if content.startswith("---"):
        # split into ['', fm, rest] if standard frontmatter
        parts = content.split("---", 2)
        if len(parts) >= 3:
            fm_text = parts[1].strip()
            body = parts[2]
            try:
                fm = yaml.safe_load(fm_text) or {}
            except Exception:
                fm = {}
            if not isinstance(fm, dict):
                fm = {}
            return fm, body, True
    # no frontmatter
    return {}, content, False


def build_yaml_frontmatter(data: dict) -> str:
    """Dump frontmatter dict to YAML string (block style), preserving unicode."""
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True, default_flow_style=False)


def process_file(filepath: str, vault_root: str, char_limit: int, force_flag: bool, write_mode: bool, debug: bool):
    """Process a single markdown file. If write_mode True -> writes file; else returns metadata dict for dryrun."""
    try:
        content = read_file_content(filepath)
    except Exception as e:
        print(f"ERR reading {filepath}: {e}")
        return None

    fm, body, had_fm = parse_frontmatter_and_body(content)

    # Skip if autoAiTag True and not forced
    if fm.get("autoAiTag") and not force_flag:
        # skip
        return None

    # Basic fields
    title = fm.get("title") or os.path.splitext(os.path.basename(filepath))[0]
    fm.setdefault("title", title)
    
    # Get file creation date for fallback
    file_creation_date = datetime.fromtimestamp(os.path.getctime(filepath)).strftime("%Y-%m-%d")
    
    # wordCount
    word_count = len(body.split())
    fm["wordCount"] = word_count

    # Date detection and AI processing
    ai_date = None
    ai_confidence = 0.0
    date_source = "FileSystem"  # default
    
    # If long enough, call LLM
    if len(body) >= char_limit:
        ai = query_llm_for_summary_and_tags(body, debug=debug)
        ai_date = ai.get("date")
        ai_confidence = ai.get("date_confidence", 0.0)
        
        if ai.get("summary"):
            fm["summary"] = ai["summary"]
        # Merge tags
        existing_tags = fm.get("tags", [])
        if isinstance(existing_tags, str):
            # try parse into list
            existing_tags = parse_tags_field(existing_tags)
        if not isinstance(existing_tags, list):
            existing_tags = list(existing_tags)
        new_tags = [t for t in ai.get("tags", []) if t not in existing_tags]
        if new_tags:
            # append up to 3 new tags
            fm["tags"] = existing_tags + new_tags[:3]
    else:
        # File too short for LLM processing
        ai_date = None
        ai_confidence = 0.0
        if debug:
            print(f"📏 File too short for LLM processing: {len(body)} chars < {char_limit} limit")
    
    # Determine the final date using our three-tier system
    final_date, date_source = determine_file_date(ai_date, ai_confidence, os.path.basename(filepath), file_creation_date)
    fm["Date"] = final_date

    fm["autoAiTag"] = True

    # Write file if requested
    if write_mode:
        yaml_text = build_yaml_frontmatter(fm)
        # preserve body spacing but ensure single blank line after frontmatter
        new_content = f"---\n{yaml_text}---\n\n{body.lstrip(chr(10))}"
        try:
            write_file_atomic(filepath, new_content)
            return {"updated": True, "path": os.path.relpath(filepath, vault_root)}
        except Exception as e:
            print(f"ERR writing {filepath}: {e}")
            return {"updated": False, "error": str(e), "path": os.path.relpath(filepath, vault_root)}
    else:
        # Dry run: return what we'd write (relative path + metadata) plus date source info
        return {
            "path": os.path.relpath(filepath, vault_root), 
            "metadata": fm,
            "date_source": date_source,
            "ai_date": ai_date,
            "ai_confidence": ai_confidence
        }


def gather_md_files(base_path: str):
    out = []
    for root, _, files in os.walk(base_path):
        for fn in files:
            if fn.lower().endswith(".md"):
                out.append(os.path.join(root, fn))
    return out


def main():
    """
    Main function for AutoAiTag.
    
    Workflow:
    1. Get vault path and processing mode from user
    2. Configure processing parameters (char limit, workers, etc.)
    3. Scan for markdown files recursively
    4. Process files in parallel using ThreadPoolExecutor
    5. Generate metadata with AI assistance for long files
    6. Save results (dry run JSON or modify files directly)
    7. Generate error logs and processing summary
    
    The script can run in two modes:
    - Dry Run: Simulates processing and saves metadata to JSON
    - Write Mode: Actually modifies markdown files with new metadata
    """
    print("=== AutoAiTag - Obsidian Vault AI Metadata Tool ===\n")

    vault_path = input("Enter path to vault (absolute or relative): ").strip()
    if not vault_path:
        print("No path entered, quitting.")
        return
    vault_path = os.path.abspath(vault_path)
    if not os.path.exists(vault_path):
        print("Path does not exist:", vault_path)
        return

    print("\nMode:\n  1) Dry Run (writes metadata_dryrun.json)\n  2) Metadata Write (modifies files)")
    mode = input("Choose 1 or 2: ").strip()
    if mode not in ("1", "2"):
        print("Invalid choice.")
        return
    write_mode = mode == "2"

    # char limit override
    char_limit = DEFAULT_CHAR_LIMIT
    if write_mode:
        ov = input(f"Override default character threshold ({DEFAULT_CHAR_LIMIT})? (y/N): ").strip().lower()
        if ov == "y":
            try:
                char_limit = int(input("Enter new character threshold: ").strip())
            except Exception:
                print("Invalid value, using default.")
    else:
        # For dry-run let user decide char_limit too
        ov = input(f"Set character threshold for Dry Run (default {DEFAULT_CHAR_LIMIT})? (y/N): ").strip().lower()
        if ov == "y":
            try:
                char_limit = int(input("Enter new character threshold: ").strip())
            except Exception:
                print("Invalid value, using default.")

    # force override
    force_flag = False
    if write_mode:
        force_flag = input("Force processing even if autoAiTag: true? (y/N): ").strip().lower() == "y"
    else:
        # allow dry-run to simulate forced run
        force_flag = input("Dry Run: simulate forcing autoAiTag processing? (y/N): ").strip().lower() == "y"

    # workers
    try:
        workers_in = int(input(f"Worker threads (default {DEFAULT_WORKERS}): ").strip() or DEFAULT_WORKERS)
        workers = max(1, workers_in)
    except Exception:
        workers = DEFAULT_WORKERS

    # LLM concurrency limit override
    llm_concurrency_limit = DEFAULT_MAX_CONCURRENT_LLM
    if write_mode:
        ov_llm_concurrency = input(f"Override default LLM concurrency limit ({DEFAULT_MAX_CONCURRENT_LLM})? (y/N): ").strip().lower()
        if ov_llm_concurrency == "y":
            try:
                llm_concurrency_limit = int(input("Enter new LLM concurrency limit: ").strip())
            except Exception:
                print("Invalid value, using default.")
    else:
        # For dry-run let user decide LLM concurrency limit too
        ov_llm_concurrency = input(f"Set LLM concurrency limit for Dry Run (default {DEFAULT_MAX_CONCURRENT_LLM})? (y/N): ").strip().lower()
        if ov_llm_concurrency == "y":
            try:
                llm_concurrency_limit = int(input("Enter new LLM concurrency limit: ").strip())
            except Exception:
                print("Invalid value, using default.")

    debug = input("Enable debug printing of raw LLM output? (y/N): ").strip().lower() == "y"

    print("\nScanning files...")
    md_files = gather_md_files(vault_path)
    if not md_files:
        print("No markdown files found under", vault_path)
        return
    print(f"Found {len(md_files)} markdown files.")
    print(f"Configuration: {workers} worker thread(s), max {llm_concurrency_limit} concurrent LLM request(s)")
    print("Processing...\n")

    results = []
    processed_count = 0
    error_log = []  # Track processing errors

    # Initialize semaphore if LLM concurrency limiting is enabled
    if llm_concurrency_limit > 0:
        global llm_semaphore
        llm_semaphore = threading.Semaphore(llm_concurrency_limit)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        future_to_file = {
            ex.submit(process_file, fp, vault_path, char_limit, force_flag, write_mode, debug): fp
            for fp in md_files
        }
        for fut in as_completed(future_to_file):
            fp = future_to_file[fut]
            try:
                res = fut.result()
                if res:
                    results.append(res)
                    processed_count += 1
            except Exception as e:
                error_info = {
                    "filepath": os.path.relpath(fp, vault_path),
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "timestamp": datetime.now().isoformat()
                }
                error_log.append(error_info)
                print(f"ERR processing {fp}: {e}")

    if write_mode:
        updated = [r for r in results if r.get("updated")]
        print(f"\nWrite complete. Processed: {len(results)} files. Successfully updated: {len(updated)}")
        
        # Save error log if there were any errors
        if error_log:
            timestamp = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
            log_filename = f"({timestamp}) AutoAiTag Log.json"
            log_path = os.path.join(vault_path, log_filename)
            try:
                with open(log_path, "w", encoding="utf-8") as fh:
                    json.dump(error_log, fh, indent=2, ensure_ascii=False)
                print(f"Error log saved to: {log_filename}")
            except Exception as e:
                print("ERR saving error log:", e)
    else:
        # Save dry run JSON
        out_path = os.path.join(vault_path, "metadata_dryrun.json")
        try:
            with open(out_path, "w", encoding="utf-8") as fh:
                json.dump(results, fh, indent=2, ensure_ascii=False)
            print(f"\nDry run complete. Metadata written to: {out_path}")
        except Exception as e:
            print("ERR saving dry run JSON:", e)
        
        # Save error log if there were any errors
        if error_log:
            timestamp = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
            log_filename = f"({timestamp}) AutoAiTag Log.json"
            log_path = os.path.join(vault_path, log_filename)
            try:
                with open(log_path, "w", encoding="utf-8") as fh:
                    json.dump(error_log, fh, indent=2, ensure_ascii=False)
                print(f"Error log saved to: {log_filename}")
            except Exception as e:
                print("ERR saving error log:", e)
        
        # Print processing summary
        print(f"\nProcessing Summary:")
        print(f"  Total files found: {len(md_files)}")
        print(f"  Successfully processed: {len(results)}")
        print(f"  Errors encountered: {len(error_log)}")
        
        # Show date source breakdown
        if results:
            date_sources = {}
            for r in results:
                source = r.get('date_source', 'Unknown')
                date_sources[source] = date_sources.get(source, 0) + 1
            
            print(f"\nDate Source Breakdown:")
            for source, count in date_sources.items():
                print(f"  {source}: {count} files")


if __name__ == "__main__":
    main()
