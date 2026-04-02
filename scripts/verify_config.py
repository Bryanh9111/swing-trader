#!/usr/bin/env python3
"""
Configuration Verification Script

Verifies that all required configuration and secrets are properly set up.
Run this before starting development or deployment.

Usage:
    python scripts/verify_config.py [--env dev|paper|live]
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# ANSI color codes
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"


def print_success(msg: str) -> None:
    """Print success message in green."""
    print(f"{GREEN}✓{RESET} {msg}")


def print_error(msg: str) -> None:
    """Print error message in red."""
    print(f"{RED}✗{RESET} {msg}")


def print_warning(msg: str) -> None:
    """Print warning message in yellow."""
    print(f"{YELLOW}⚠{RESET}  {msg}")


def check_python_version() -> bool:
    """Verify Python version >= 3.11."""
    version = sys.version_info
    if version.major == 3 and version.minor >= 11:
        print_success(f"Python version: {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print_error(f"Python version {version.major}.{version.minor} < 3.11 (required)")
        return False


def check_file_exists(filepath: Path, required: bool = True) -> bool:
    """Check if a file exists."""
    if filepath.exists():
        print_success(f"Found: {filepath}")
        return True
    else:
        if required:
            print_error(f"Missing (required): {filepath}")
        else:
            print_warning(f"Missing (optional): {filepath}")
        return required is False


def check_config_files(env: str) -> bool:
    """Check that all required config files exist."""
    print("\n=== Configuration Files ===")

    project_root = Path(__file__).parent.parent
    config_dir = project_root / "config"

    files_to_check = [
        (config_dir / "config.yaml", True),
        (config_dir / "secrets.env.example", True),
        (config_dir / "secrets.env", True),
    ]

    # Add environment-specific config if not dev
    if env != "dev":
        files_to_check.append((config_dir / f"config.{env}.yaml", True))

    all_exist = True
    for filepath, required in files_to_check:
        if not check_file_exists(filepath, required):
            all_exist = False

    return all_exist


def check_secrets_env(env: str) -> Tuple[bool, List[str]]:
    """Check that secrets.env has required variables."""
    print("\n=== Secrets Environment Variables ===")

    project_root = Path(__file__).parent.parent
    secrets_file = project_root / "config" / "secrets.env"

    if not secrets_file.exists():
        print_error("secrets.env not found. Copy from secrets.env.example")
        return False, []

    # Read secrets file
    with open(secrets_file, "r") as f:
        content = f.read()

    # Required secrets based on environment
    required_secrets = [
        "ENCRYPTION_KEY",
    ]

    # Add environment-specific secrets
    if env == "paper":
        required_secrets.extend([
            "IBKR_PAPER_USERNAME",
            "IBKR_PAPER_PASSWORD",
            "POLYGON_API_KEY",
        ])
    elif env == "live":
        required_secrets.extend([
            "IBKR_LIVE_USERNAME",
            "IBKR_LIVE_PASSWORD",
            "POLYGON_API_KEY",
            "TELEGRAM_BOT_TOKEN",
            "TELEGRAM_CHAT_ID",
        ])
    else:  # dev
        # Dev can use free data sources
        pass

    missing_secrets = []
    for secret in required_secrets:
        if secret in content and f"{secret}=your_" not in content:
            print_success(f"{secret} is set")
        else:
            print_error(f"{secret} is missing or has placeholder value")
            missing_secrets.append(secret)

    return len(missing_secrets) == 0, missing_secrets


def check_directory_structure() -> bool:
    """Check that required directories exist."""
    print("\n=== Directory Structure ===")

    project_root = Path(__file__).parent.parent

    required_dirs = [
        "config",
        "docs",
        "docs/adr",
        "docs/sessions",
        "docs/schemas",
        "docs/patterns",
        "docs/runbooks",
        "scripts",
    ]

    all_exist = True
    for dir_name in required_dirs:
        dir_path = project_root / dir_name
        if dir_path.exists() and dir_path.is_dir():
            print_success(f"Directory: {dir_name}/")
        else:
            print_error(f"Missing directory: {dir_name}/")
            all_exist = False

    return all_exist


def check_git_status() -> bool:
    """Check git repository status."""
    print("\n=== Git Repository ===")

    project_root = Path(__file__).parent.parent
    git_dir = project_root / ".git"

    if not git_dir.exists():
        print_error("Not a git repository")
        return False

    print_success("Git repository initialized")

    # Check if secrets.env is in .gitignore
    gitignore = project_root / ".gitignore"
    if gitignore.exists():
        with open(gitignore, "r") as f:
            gitignore_content = f.read()

        if "config/secrets.env" in gitignore_content:
            print_success("secrets.env is in .gitignore")
        else:
            print_error("secrets.env NOT in .gitignore (security risk!)")
            return False

    return True


def print_summary(checks: Dict[str, bool]) -> None:
    """Print summary of all checks."""
    print("\n" + "=" * 50)
    print("=== VERIFICATION SUMMARY ===")
    print("=" * 50)

    all_passed = all(checks.values())

    for check_name, passed in checks.items():
        status = f"{GREEN}PASS{RESET}" if passed else f"{RED}FAIL{RESET}"
        print(f"{check_name:.<40} {status}")

    print("=" * 50)

    if all_passed:
        print(f"\n{GREEN}✓ All checks passed!{RESET}")
        print("You're ready to start development.")
    else:
        print(f"\n{RED}✗ Some checks failed!{RESET}")
        print("Please fix the issues above before proceeding.")

    return all_passed


def main():
    parser = argparse.ArgumentParser(
        description="Verify configuration and secrets setup"
    )
    parser.add_argument(
        "--env",
        choices=["dev", "paper", "live"],
        default="dev",
        help="Environment to verify (default: dev)",
    )
    args = parser.parse_args()

    print("=" * 50)
    print(f"Configuration Verification - Environment: {args.env.upper()}")
    print("=" * 50)

    # Run all checks
    checks = {
        "Python Version": check_python_version(),
        "Configuration Files": check_config_files(args.env),
        "Secrets Environment": check_secrets_env(args.env)[0],
        "Directory Structure": check_directory_structure(),
        "Git Repository": check_git_status(),
    }

    # Print summary
    all_passed = print_summary(checks)

    # Exit with appropriate code
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
