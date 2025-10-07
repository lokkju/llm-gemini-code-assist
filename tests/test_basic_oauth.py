#!/usr/bin/env python3
"""Quick test to verify OAuth credential loading works."""

from llm_gemini_code_assist import get_oauth_credentials, get_code_assist_project, OAuthError

try:
    print("Testing OAuth credential loading...")
    credentials = get_oauth_credentials()

    if credentials:
        print(f"✓ Loaded credentials")
        print(f"  Token valid: {credentials.valid}")
        print(f"  Has refresh token: {bool(credentials.refresh_token)}")
        print(f"  Scopes: {credentials.scopes[:2] if credentials.scopes else None}...")

        print("\nTesting Code Assist project assignment...")
        project_id, user_tier = get_code_assist_project(credentials)

        if project_id:
            print(f"✓ Got project assignment")
            print(f"  Project ID: {project_id}")
            print(f"  User tier: {user_tier}")
        else:
            print("✗ Failed to get project assignment")
    else:
        print("✗ No OAuth credentials found at ~/.gemini/oauth_creds.json")

except OAuthError as e:
    print(f"✗ OAuth Error: {e}")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
