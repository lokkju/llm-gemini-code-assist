#!/usr/bin/env python3
"""
Sync models from gemini-cli's canonical model configuration.

Fetches the model list from:
https://github.com/google-gemini/gemini-cli/blob/main/packages/core/src/config/defaultModelConfigs.ts

Compares with GEMINI_CODE_ASSIST_MODELS and reports differences.

Usage:
    uv run scripts/sync_models.py
"""

import re
import sys
from pathlib import Path

import httpx


GEMINI_CLI_CONFIG_URL = (
    "https://raw.githubusercontent.com/google-gemini/gemini-cli/main/"
    "packages/core/src/config/defaultModelConfigs.ts"
)

# User-facing model patterns (excludes internal configs like 'base', 'classifier', etc.)
# Matches: gemini-2.5-pro, gemini-3-flash-preview, gemini-2.5-flash-lite
# Excludes: gemini-2.5-flash-base (internal config)
USER_FACING_MODEL_PATTERN = re.compile(r"^gemini-[\d.]+-[\w-]+$|^gemini-\d+-[\w-]+$")

# Internal config suffixes to exclude
INTERNAL_SUFFIXES = {"-base"}


def fetch_gemini_cli_config() -> str:
    """Fetch the TypeScript config file from gemini-cli repo."""
    response = httpx.get(GEMINI_CLI_CONFIG_URL, timeout=30.0)
    response.raise_for_status()
    return response.text


def parse_model_ids(config_text: str) -> set[str]:
    """
    Parse model IDs from the TypeScript config.

    Looks for patterns like:
    - 'gemini-2.5-pro': { ... }
    - 'gemini-3-flash-preview': { ... }
    """
    # Match quoted strings that look like model IDs at the start of object definitions
    model_pattern = re.compile(r"['\"]([^'\"]+)['\"]:\s*\{")
    all_keys = set(model_pattern.findall(config_text))

    # Filter to only user-facing models (gemini-X.X-name or gemini-X-name patterns)
    user_models = {key for key in all_keys if USER_FACING_MODEL_PATTERN.match(key)}

    # Exclude internal configs
    user_models = {
        m for m in user_models if not any(m.endswith(suffix) for suffix in INTERNAL_SUFFIXES)
    }

    return user_models


def get_local_models() -> set[str]:
    """Import and return the local GEMINI_CODE_ASSIST_MODELS set."""
    # Add parent directory to path to import the module
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from llm_gemini_code_assist import GEMINI_CODE_ASSIST_MODELS

    return GEMINI_CODE_ASSIST_MODELS


def main() -> int:
    print("Fetching gemini-cli model configuration...")
    try:
        config_text = fetch_gemini_cli_config()
    except httpx.HTTPError as e:
        print(f"Error fetching config: {e}")
        return 1

    remote_models = parse_model_ids(config_text)
    local_models = get_local_models()

    print(f"\nRemote models (gemini-cli): {sorted(remote_models)}")
    print(f"Local models (this plugin): {sorted(local_models)}")

    new_models = remote_models - local_models
    removed_models = local_models - remote_models

    if new_models:
        print(f"\n+ New models available: {sorted(new_models)}")
    if removed_models:
        print(f"\n- Models no longer in gemini-cli: {sorted(removed_models)}")

    if not new_models and not removed_models:
        print("\n✓ Models are in sync!")
        return 0

    print("\nTo update, edit GEMINI_CODE_ASSIST_MODELS in llm_gemini_code_assist.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
