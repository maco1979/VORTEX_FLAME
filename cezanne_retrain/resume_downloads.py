#!/usr/bin/env python3
"""Resume incomplete HuggingFace downloads for Cezanne training data"""
import os, sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

E_DRIVE = "E:/AI_Data"

DOWNLOADS = [
    {
        "name": "Capybara",
        "repo": "LDJnr/Capybara",
        "local": os.path.join(E_DRIVE, "Capybara"),
    },
    {
        "name": "Puffin",
        "repo": "LDJnr/Puffin",
        "local": os.path.join(E_DRIVE, "Puffin"),
    },
    {
        "name": "Pure-Dove",
        "repo": "LDJnr/Pure-Dove",
        "local": os.path.join(E_DRIVE, "Pure-Dove"),
    },
    {
        "name": "LessWrong-Amplify-Instruct",
        "repo": "LDJnr/LessWrong-Amplify-Instruct",
        "local": os.path.join(E_DRIVE, "LessWrong-Amplify-Instruct"),
    },
    {
        "name": "rStar-Coder-seed_testcase",
        "repo": "microsoft/rStar-Coder",
        "local": os.path.join(E_DRIVE, "rStar-Coder"),
        "subset": "seed_testcase",
    },
]


def main():
    from huggingface_hub import snapshot_download

    for d in DOWNLOADS:
        name = d["name"]
        repo = d["repo"]
        local = d["local"]
        subset = d.get("subset")

        print(f"\n{'='*50}", flush=True)
        print(f"  Downloading: {name}", flush=True)
        print(f"  Repo: {repo}", flush=True)
        print(f"  Local: {local}", flush=True)
        print(f"{'='*50}", flush=True)

        try:
            if subset:
                snapshot_download(
                    repo_id=repo,
                    repo_type="dataset",
                    local_dir=local,
                    allow_patterns=[f"{subset}/**"],
                    resume_download=True,
                )
            else:
                snapshot_download(
                    repo_id=repo,
                    repo_type="dataset",
                    local_dir=local,
                    resume_download=True,
                )
            print(f"  [DONE] {name} completed!", flush=True)
        except Exception as e:
            print(f"  [ERROR] {name}: {e}", flush=True)

    print(f"\n{'#'*50}", flush=True)
    print(f"  All downloads finished!", flush=True)
    print(f"{'#'*50}", flush=True)


if __name__ == "__main__":
    main()
