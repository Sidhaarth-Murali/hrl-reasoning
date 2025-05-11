#!/usr/bin/env python3
"""
Test runner script for evaluating different RL approaches with minimal configurations.
This script runs the following approaches sequentially:
1. ARCHER
2. SCoRe 
3. SMART-SCoRe
4. RL-Guided SCoRe 
5. Bilevel SCoRe

Each approach uses a minimal configuration for quick testing to verify model saving works correctly.
"""

import os
import sys
import glob
import time
import subprocess
import argparse
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_test(config_path, expected_save_path, name, script_path=None):
    """Run a test with the given configuration and check if models are saved correctly."""
    print(f"\n{'='*80}")
    print(f"RUNNING TEST: {name}")
    print(f"{'='*80}")
    print(f"Config: {config_path}")
    print(f"Expected save path: {expected_save_path}")
    
    # Clear existing test models
    if os.path.exists(expected_save_path):
        existing_files = glob.glob(f"{expected_save_path}/*.pt")
        for file in existing_files:
            print(f"Removing existing test model: {file}")
            os.remove(file)
    
    # Ensure save directory exists
    os.makedirs(expected_save_path, exist_ok=True)
    
    # Determine which script to run based on the agent type
    if script_path is None:
        if "archer" in name.lower() or "score" in name.lower():
            script_path = "scripts/run.py"
        elif "bilevel" in name.lower():
            script_path = "run_bi_level_score.py"
        else:
            script_path = "scripts/run.py"  # Default to run.py
    
    # Build command to run with this config
    cmd = [
        "python", script_path, 
        f"--config-name={os.path.basename(config_path)}",
        f"hydra.searchpath=[{os.path.dirname(config_path)}]"
    ]
    
    # Run the command
    start_time = time.time()
    print(f"Running command: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd, 
            check=True, 
            text=True, 
            capture_output=True,
            env=dict(os.environ, PYTHONPATH=os.getcwd())
        )
        print("STDOUT:")
        print(result.stdout[-1000:])  # Print last 1000 chars of output
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False
    
    elapsed = time.time() - start_time
    
    # Check if models were saved
    saved_models = glob.glob(f"{expected_save_path}/*.pt")
    if not saved_models:
        print(f"❌ ERROR: No models were saved to {expected_save_path}")
        return False
    
    print(f"✅ SUCCESS: Test completed in {elapsed:.2f} seconds")
    print(f"Found {len(saved_models)} saved models:")
    for model in saved_models:
        print(f"  - {os.path.basename(model)}")
    
    return True


def main():
    """Run all the tests sequentially."""
    parser = argparse.ArgumentParser(description='Run tests for all RL approaches')
    parser.add_argument('--test', type=str, help='Run only a specific test (archer, score, smart, rl, bilevel)')
    args = parser.parse_args()
    
    # Base paths
    base_dir = os.path.abspath(os.path.dirname(__file__))
    config_dir = os.path.join(base_dir, "scripts", "test_configs")
    test_models_dir = os.path.join(base_dir, ".test_models")
    
    # Define test configurations
    tests = [
        {
            "name": "ARCHER",
            "config": os.path.join(config_dir, "test_archer.yaml"),
            "save_path": os.path.join(test_models_dir, "archer"),
            "test_id": "archer",
            "script": "scripts/run.py"
        },
        {
            "name": "SCoRe",
            "config": os.path.join(config_dir, "test_score.yaml"),
            "save_path": os.path.join(test_models_dir, "score"),
            "test_id": "score",
            "script": "scripts/run.py"
        },
        {
            "name": "SMART-SCoRe",
            "config": os.path.join(config_dir, "test_smart_score.yaml"),
            "save_path": os.path.join(test_models_dir, "smart_score"),
            "test_id": "smart",
            "script": "scripts/run.py"
        },
        {
            "name": "RL-Guided SCoRe",
            "config": os.path.join(config_dir, "test_rl_guided.yaml"),
            "save_path": os.path.join(test_models_dir, "hrl_score"),
            "test_id": "rl",
            "script": "scripts/run.py"
        },
        {
            "name": "Bilevel SCoRe",
            "config": os.path.join(config_dir, "test_bilevel.yaml"),
            "save_path": os.path.join(test_models_dir, "bilevel_score"),
            "test_id": "bilevel",
            "script": "run_bi_level_score.py"
        }
    ]
    
    # Run all tests or just the specified one
    results = {}
    for test in tests:
        if args.test and args.test.lower() != test["test_id"]:
            continue
            
        # Create the save directory if it doesn't exist
        os.makedirs(test["save_path"], exist_ok=True)
        
        # Run the test
        success = run_test(test["config"], test["save_path"], test["name"], test["script"])
        results[test["name"]] = success
    
    # Print summary
    print("\n\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    all_success = True
    for name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{name}: {status}")
        if not success:
            all_success = False
    
    # Return exit code
    return 0 if all_success else 1


if __name__ == "__main__":
    sys.exit(main()) 