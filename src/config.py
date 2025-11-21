"""
Configuration module for the AMLEC Challenge Benchmark.

This module handles:
1. Path definitions for local and remote resources.
2. Environment variable retrieval (Hugging Face Token).
3. Logging configuration.
"""

import os
import pathlib
import logging
from datetime import datetime

# ===============================
# PATHS AND DIRECTORIES
# ===============================
# Resolve base directory
try:
    BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
except NameError:
    BASE_DIR = pathlib.Path.cwd()

# Data Directories
LOCAL_DIR = BASE_DIR / "dataset" / "rtm_emulation"
CACHE_DIR = LOCAL_DIR  # Using same dir for cache in this context
RESULTS_DIR = LOCAL_DIR / "results"

# Backup and Logging Directories
BACKUP_DIR = BASE_DIR / "backup"
NOW_DATE = datetime.now().strftime("%Y%m%d_%H%M%S")

ORI_BACK_RESULTS = BACKUP_DIR / "results_base"
BACK_RESULTS_BASE = BACKUP_DIR / "results"
BACK_RESULTS_CURRENT = BACKUP_DIR / f"results_{NOW_DATE}"
ERROR_DIR = BACK_RESULTS_CURRENT / "validation_errors"

# Reference Data (Update this path if running on a new server)
# For portability, we default to a local path, but keep your absolute path as fallback/comment.
REFERENCE_DIR = pathlib.Path("/data/users/julio/rtm_emulation/")
if not REFERENCE_DIR.exists():
    REFERENCE_DIR = LOCAL_DIR  # Fallback to local dataset if absolute path fails

# Readme Path
README_PATH = LOCAL_DIR / "README.md"

# ===============================
# HUGGING FACE CONFIGURATION
# ===============================
# NOTE: Ideally, load this from a .env file or system environment for security.
# Do not hardcode tokens in public repositories.
HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN", "----YOUR_HF_TOKEN_HERE----")
REPO_ID = "isp-uv-es/rtm_emulation"
REPO_TYPE = "dataset"

# ===============================
# LOGGING SETUP
# ===============================
LOG_FILE = f"log_{NOW_DATE}.txt"
LOG_PATH = BACKUP_DIR / LOG_FILE

# Create necessary directories
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
ERROR_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_PATH, mode="w"),
    ],
)
logger = logging.getLogger("AMLEC_Benchmark")