"""
Report Generation Module.

Builds the Markdown tables and updates the README file on Hugging Face.
"""

import re
import shutil
import pandas as pd
from datetime import datetime
from huggingface_hub import HfApi

from src.config import README_PATH, logger, HF_TOKEN, REPO_ID, REPO_TYPE
from src.analysis import run_statistics

def build_benchmark_table() -> str:
    """Generates a Markdown table from the statistics dataframe."""
    logger.info("Building benchmark table...")
    df = run_statistics()
    
    if df.empty:
        return "No results available."

    header = (
        "## **Benchmark Results**\n\n"
        "| **Model** | **MRE A1 (%)** | **MRE A2 (%)** | **MRE B1 (%)** | **MRE B2 (%)** | **Score** | **Runtime (s)** | **Rank** |\n"
        "|-----------|---------------|---------------|---------------|---------------|----------|----------|--------|"
    )

    rows = []
    for _, row in df.iterrows():
        # Helper to format floats
        fmt = lambda x: f"{x:.3f}" if pd.notna(x) else ""
        
        row_values = [
            str(row["Model"]),
            fmt(row["MRE_A1"]),
            fmt(row["MRE_A2"]),
            fmt(row["MRE_B1"]),
            fmt(row["MRE_B2"]),
            fmt(row["Score"]),
            fmt(row["Runtime"]),
            f"{int(row['Ranking'])}°" if pd.notna(row["Ranking"]) else "-",
        ]
        rows.append("| " + " | ".join(row_values) + " |")

    return header + "\n" + "\n".join(rows)


def update_markdown_file() -> None:
    """
    Updates the local README.md with the latest benchmark table and date.
    """
    logger.info("Updating README at '%s'", README_PATH)

    if not README_PATH.exists():
        logger.error("README file not found at %s", README_PATH)
        return

    # Create Backup
    backup_name = README_PATH.with_name(f"{README_PATH.stem}_backup_{datetime.now():%Y%m%d_%H%M%S}.md")
    shutil.copy2(README_PATH, backup_name)

    content = README_PATH.read_text("utf-8")

    # 1. Update Date
    today_str = datetime.today().strftime("%d-%m-%Y")
    content, n_subs = re.subn(r"Last update: \d{2}-\d{2}-\d{4}", f"Last update: {today_str}", content)

    # 2. Generate and Insert New Table
    new_table = build_benchmark_table()
    
    # Remove old table
    table_regex = re.compile(r"^## \*\*Benchmark Results\*\*.*?(?=^## |\Z)", flags=re.DOTALL | re.MULTILINE)
    content = table_regex.sub("", content)

    # Insert new table before Introduction
    intro_regex = re.compile(r"^## \*\*Introduction\*\*", flags=re.MULTILINE)
    match = intro_regex.search(content)

    if match:
        insert_pos = match.start()
        content = content[:insert_pos] + new_table.strip() + "\n\n" + content[insert_pos:]
    else:
        content = content.rstrip() + "\n\n" + new_table

    README_PATH.write_text(content, "utf-8")
    logger.info("README updated successfully.")


def push_readme_to_hub() -> None:
    """Uploads the updated README.md to the Hugging Face Hub."""
    logger.info("Pushing README to Hugging Face Hub...")
    api = HfApi(token=HF_TOKEN)
    api.upload_file(
        path_or_fileobj=README_PATH,
        path_in_repo="README.md",
        repo_id=REPO_ID,
        token=HF_TOKEN,
        repo_type=REPO_TYPE,
    )
    logger.info("README pushed successfully.")