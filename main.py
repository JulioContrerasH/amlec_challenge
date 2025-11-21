"""
Main Execution Script for AMLEC Benchmark.

Orchestrates the workflow:
1. Merges PRs.
2. Downloads data.
3. Validates submissions.
4. Calculates stats and updates README.
5. Cleans up and syncs with Hub.
"""

from src.config import logger
from src.data_manager import (
    pull_request, download_dataset, validate_results_dir, 
    remove_obsolete_files, squash_commit_history
)
from src.report import update_markdown_file, push_readme_to_hub

# Optional: Upload to Space if needed
def up_spaces() -> None:
    from huggingface_hub import HfApi
    from src.config import HF_TOKEN, REPO_ID
    
    # This seems specific to your setup, kept as requested
    api = HfApi(token=HF_TOKEN)
    try:
        api.upload_file(
            path_or_fileobj="dataset/rtm_emulation/README.md",
            path_in_repo="README.md",
            repo_id=REPO_ID, # Verify if this should be a Space ID or Dataset ID
            repo_type="space",
            commit_message="Sync README to Space",
        )
        logger.info("Uploaded README to Space.")
    except Exception as e:
        logger.warning(f"Failed to upload to Space: {e}")

if __name__ == "__main__":
    logger.info("Starting AMLEC Benchmark Automation.")
    
    try:
        pull_request()
        download_dataset()
        validate_results_dir()
        update_markdown_file()
        push_readme_to_hub()
        remove_obsolete_files()
        squash_commit_history()
        up_spaces()
        logger.info("Automation completed successfully.")
        
    except Exception as e:
        logger.critical(f"Script failed with error: {e}", exc_info=True)