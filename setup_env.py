"""
Filename: setup_env.py
Description: Initializes the project directory structure, verifies hardware acceleration,
             and sets up logging configurations.
"""

import os
import sys
import logging
import platform
import torch
from pathlib import Path

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

PROJECT_STRUCTURE = [
    "data/raw/images",
    "data/raw/text",
    "data/processed/embeddings",
    "data/processed/graph",
    "src/encoders",
    "src/graph",
    "src/fusion",
    "src/utils",
    "models/checkpoints",
    "models/exported",
    "logs",
    "notebooks",
    "config"
]

def check_hardware_acceleration():
    """Detects and logs available hardware acceleration (CUDA/MPS/CPU)."""
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"[CUDA] Detected: {gpu_name} (VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB)")
    elif torch.backends.mps.is_available():
        device = "mps"
        logger.info("[MPS] Apple Metal Performance Shaders Detected")
    else:
        logger.warning("[WARNING] No Hardware Acceleration Detected. Training will be slow on CPU.")
    
    # Save device config for future scripts
    with open("config/device_config.txt", "w") as f:
        f.write(device)
    
    return device

def initialize_directories(root_path: str):
    """Creates the project folder structure."""
    root = Path(root_path)
    
    for folder in PROJECT_STRUCTURE:
        dir_path = root / folder
        try:
            dir_path.mkdir(parents=True, exist_ok=True)
            # Create a .gitkeep to ensure git tracks empty folders
            (dir_path / ".gitkeep").touch()
            logger.info(f"[OK] Created/Verified: {dir_path}")
        except Exception as e:
            logger.error(f"Failed to create {dir_path}: {e}")
            sys.exit(1)

def main():
    logger.info("Starting MMKG Project Initialization...")
    
    # Set root to current working directory
    project_root = os.getcwd()
    
    # 1. Initialize Folders
    initialize_directories(project_root)
    
    # 2. Check Hardware
    device = check_hardware_acceleration()
    
    logger.info(f"[SUCCESS] Phase 0 Setup Complete. Target Device: {device.upper()}")
    logger.info(f"Project Root: {project_root}")

if __name__ == "__main__":
    main()

