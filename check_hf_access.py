#!/usr/bin/env python
"""
Check HF gated repo access and provide next steps.
Usage: python check_hf_access.py
"""

import os
from huggingface_hub import HfApi, login

token = os.getenv("HUGGINGFACE_TOKEN", "")

try:
    api = HfApi()
    repo_info = api.repo_info("facebook/sam3", token=token)
    print("✓ SUCCESS: You have access to facebook/sam3!")
    print(f"  Repo: {repo_info.id}")
    print(f"  Private: {repo_info.private}")
    print(f"  Last modified: {repo_info.last_modified}")
except Exception as e:
    error_msg = str(e)
    if "gated" in error_msg.lower() or "401" in error_msg:
        print("✗ BLOCKED: facebook/sam3 is gated and you don't have access yet.")
        print("\nNext steps:")
        print("1. Go to: https://huggingface.co/facebook/sam3")
        print("2. Click 'Request access' button")
        print("3. Fill out form (name, organization, use case)")
        print("4. Submit and wait for approval email (1-48 hours)")
        print("5. Once approved, rerun: python check_hf_access.py")
    else:
        print(f"✗ ERROR: {error_msg}")
