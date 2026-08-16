#!/usr/bin/env python
"""Print all branches of a HF repo, one per line (manual checkpoint inspection)."""

import argparse


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", required=True, help="HF repo id, e.g. allenai/Olmo-3-1025-7B")
    ap.add_argument("--tags", action="store_true", help="also print tags, prefixed 'tag:'")
    args = ap.parse_args()

    from huggingface_hub import list_repo_refs
    refs = list_repo_refs(args.repo)
    for branch in sorted(refs.branches, key=lambda b: b.name):
        print(branch.name)
    if args.tags:
        for tag in sorted(refs.tags, key=lambda t: t.name):
            print(f"tag:{tag.name}")


if __name__ == "__main__":
    main()
