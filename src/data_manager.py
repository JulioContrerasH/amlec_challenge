"""
Data Manager Module.

Handles:
1. Downloading datasets from Hugging Face.
2. Managing Pull Requests.
3. Validating submission file formats.
4. Cleaning up local files.
"""

import shutil
import pathlib
import re
import h5py
import os
from typing import Tuple, List, Dict
from huggingface_hub import HfApi, snapshot_download, DiscussionWithDetails
from huggingface_hub.errors import EntryNotFoundError

from src.config import (
    HF_TOKEN, REPO_ID, REPO_TYPE, LOCAL_DIR, CACHE_DIR, 
    BACK_RESULTS_BASE, BACK_RESULTS_CURRENT, ORI_BACK_RESULTS, 
    ERROR_DIR, REFERENCE_DIR, logger
)

# Global list to track files that need to be removed from the repo
remove_pull_list = []

def _check_file_validity(target: pathlib.Path, reference: pathlib.Path) -> Tuple[bool, List[str]]:
    """
    Internal validator for a single HDF5 submission file.
    """
    errors = []

    # 1. Check existence and extension
    if not target.is_file():
        errors.append("Target file does not exist.")
    elif target.suffix.lower() not in {".h5", ".hdf5"}:
        errors.append("Target file is not .h5/.hdf5.")

    # 2. Check naming convention <model>_<Sx>
    name = target.stem
    if not re.fullmatch(r"^[A-Za-z0-9_]+_(A|B)[12]$", name):
        errors.append("Filename must be <model>_<Sx> with S=A|B and x=1|2.")

    if errors:
        return False, errors

    # 3. Check internal structure (LUTdata)
    try:
        with h5py.File(target, "r") as hf:
            lut_tgt = hf["/LUTdata"][()]
            # Check for runtime attribute
            _ = hf.attrs['runtime']
    except Exception as e:
        return False, errors + [f"File read error or missing keys: {str(e)}"]

    if lut_tgt.ndim != 2:
        errors.append("LUTdata in target is not 2-D.")

    # 4. Check against Reference
    if not reference.is_file():
        # If reference is missing, we can't validate dimensions strictly
        errors.append(f"Reference file not found at {reference}.")
        return False, errors

    try:
        with h5py.File(reference, "r") as hf:
            lut_ref = hf["/LUTdata"][()]
    except Exception:
        return False, errors + ["'/LUTdata' not found in reference."]

    if lut_ref.ndim != 2:
        errors.append("LUTdata in reference is not 2-D.")
    
    # Validate sample count matches (dim 0)
    if lut_tgt.shape[0] != lut_ref.shape[0]:
        errors.append("Second dimension of LUTdata does not match reference.")

    return len(errors) == 0, errors


def pull_request() -> None:
    """
    Merges all open Pull Requests in the Hugging Face repository 
    if there are no conflicts.
    """
    api = HfApi(token=HF_TOKEN)
    open_prs: list[DiscussionWithDetails] = api.get_repo_discussions(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        discussion_type="pull_request",
        discussion_status="open"
    )

    if not open_prs:
        logger.info("No open PRs found — nothing to merge.")
        return

    for pr in open_prs:
        if pr.status == "open":
            try:
                api.merge_pull_request(
                    repo_id=REPO_ID,
                    discussion_num=pr.num,
                    repo_type=REPO_TYPE,
                )
                logger.info("✔️  Merged PR #%d — %s", pr.num, pr.title)
            except Exception as e:
                logger.warning("❌  Could not merge PR #%d → %s", pr.num, e)


def download_dataset() -> None:
    """
    Downloads or updates the dataset locally via `snapshot_download`.
    Manages the backup and results folders.
    """
    # Remove local dir to ensure a fresh sync
    if LOCAL_DIR.exists():
        shutil.rmtree(LOCAL_DIR)
        logger.info("Local dataset folder removed. Re-downloading...")

    logger.info("Downloading dataset from Hugging Face Hub...")
    cache_results = CACHE_DIR / "results"
    
    snapshot_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        local_dir=LOCAL_DIR.as_posix(),
        token=HF_TOKEN,
        local_dir_use_symlinks=False,
        resume_download=True,
        cache_dir=CACHE_DIR.as_posix(),
        local_files_only=False,
    )

    # Initialize backup directories
    if not ORI_BACK_RESULTS.exists():
        if cache_results.exists():
            shutil.copytree(cache_results, ORI_BACK_RESULTS)
            shutil.copytree(ORI_BACK_RESULTS, BACK_RESULTS_BASE)
    else:
        BACK_RESULTS_CURRENT.mkdir(parents=True, exist_ok=True)
        
        # Copy new files
        if cache_results.exists():
            for item in cache_results.iterdir():
                if item.is_file() and not (ORI_BACK_RESULTS / item.name).exists():
                    shutil.copy(item, BACK_RESULTS_CURRENT / item.name)
            
            # Sync current backup to base backup
            for item in BACK_RESULTS_CURRENT.iterdir():
                if item.is_file():
                    shutil.copy(item, BACK_RESULTS_BASE / item.name)
                    logger.info(f"Copied new file {item.name} to BACK_RESULTS_BASE")


def validate_results_dir() -> Dict[str, bool]:
    """
    Validates all *.h5 files in the backup results directory.
    Removes invalid files and logs errors.
    """
    track_ref = {1: "refInterp.h5", 2: "refExtrap.h5"}
    pattern = re.compile(r'^(?P<model>[A-Za-z0-9_]+)_(?P<scenario>[AB])(?P<track>[12])\.h5$', re.I)
    status = {}

    # Iterate over the current backup results
    if not BACK_RESULTS_CURRENT.exists():
        logger.warning("No current results directory found to validate.")
        return {}

    for f in BACK_RESULTS_CURRENT.glob("*.h5"):
        removeh5 = BACK_RESULTS_BASE / f.name
        m = pattern.match(f.name)
        
        # 1. Regex check
        if not m:
            txt = ["Filename incorrect. Must be <model>_<Sx>."]
            (ERROR_DIR / f"{f.stem}.txt").write_text("\n".join(txt))
            
            if removeh5.exists():
                removeh5.unlink()
                remove_pull_list.append("results/" + f.name)
            status[f.name] = False
            continue

        # 2. Content check
        scenario = m["scenario"].upper()
        track = int(m["track"])
        ref_file = (REFERENCE_DIR / f"scenario{scenario}" / "reference" / track_ref[track])
        
        ok, errs = _check_file_validity(target=f, reference=ref_file)
        status[f.name] = ok

        if not ok:
            (ERROR_DIR / f"{f.stem}.txt").write_text("\n".join(errs))
            if removeh5.exists():
                removeh5.unlink()
                remove_pull_list.append("results/" + f.name)
    
    return status


def remove_obsolete_files() -> None:
    """
    Removes files from the remote repository that are deemed invalid or obsolete.
    """
    api = HfApi(token=HF_TOKEN)
    files_to_delete = []

    # Identify files in BASE but not in ORIGINAL (meaning they were added but might be invalid)
    for item in BACK_RESULTS_BASE.iterdir():
        if item.is_file() and not (ORI_BACK_RESULTS / item.name).exists():
            file_path = f"results/{item.name}"
            files_to_delete.append(file_path)

    # Add explicitly flagged invalid files
    files_to_delete.extend(list(set(remove_pull_list)))

    for file in files_to_delete:
        try:
            api.delete_file(
                path_in_repo=file,
                repo_id=REPO_ID,
                repo_type=REPO_TYPE,
                token=HF_TOKEN,
                commit_message="Delete: obsolete or invalid file"
            )
            logger.info("Deleted remote file: %s", file)
        except EntryNotFoundError:
            logger.warning("Remote file not found, skipping delete: %s", file)


def squash_commit_history() -> None:
    """Squashes the commit history of the dataset repository."""
    api = HfApi(token=HF_TOKEN)
    api.super_squash_history(
        repo_id=REPO_ID,
        commit_message="Squashed history: AMLEC Challenge Benchmark Update",
        repo_type=REPO_TYPE,
        token=HF_TOKEN
    )
    logger.info("Commit history has been squashed.")