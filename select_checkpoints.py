#!/usr/bin/env python
"""Pick ~log-spaced checkpoint branches from a HF repo (pure selection fn + CLI).

The CLI prints one revision per line in training order, for run_all.sh to
mapfile. Exits non-zero with a stderr message when nothing matches.
"""

import argparse
import math
import re
import sys


def _last_int(name):
    nums = re.findall(r"\d+", name)
    return int(nums[-1]) if nums else None


def pick_spaced(names, pattern, n):
    """Filter `names` by regex `pattern`, sort by the last integer in each name,
    and return ~log-spaced picks that always include the first and the last.

    n=1 returns just the last; n >= number of matches returns all matches.
    Names without any integer are dropped. Raises ValueError if nothing matches.
    """
    rx = re.compile(pattern)
    matched = [nm for nm in names if rx.search(nm) and _last_int(nm) is not None]
    if not matched:
        raise ValueError(f"no branch names with an integer match pattern {pattern!r}")
    matched.sort(key=lambda nm: (_last_int(nm), nm))
    if n <= 1:
        return [matched[-1]]
    if n >= len(matched):
        return list(matched)

    values = [_last_int(nm) for nm in matched]
    # Log-spaced targets need a positive start; a step-0 first checkpoint is
    # still force-included below.
    positives = [v for v in values if v > 0]
    lo = float(positives[0]) if positives else 1.0
    hi = float(max(values[-1], lo))
    chosen = {0, len(matched) - 1}
    for i in range(1, n - 1):
        target = lo * (hi / lo) ** (i / (n - 1)) if hi > lo else lo
        remaining = [j for j in range(len(matched)) if j not in chosen]
        best = min(remaining, key=lambda j: abs(
            math.log(max(values[j], 1)) - math.log(max(target, 1))))
        chosen.add(best)
    return [matched[j] for j in sorted(chosen)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", required=True, help="HF repo id, e.g. allenai/Olmo-3-1025-7B")
    ap.add_argument("--pattern", required=True,
                    help="regex matched against branch names, e.g. '^stage1-step'")
    ap.add_argument("--n", type=int, default=6)
    args = ap.parse_args()

    from huggingface_hub import list_repo_refs
    refs = list_repo_refs(args.repo)
    names = [b.name for b in refs.branches]
    try:
        picked = pick_spaced(names, args.pattern, args.n)
    except ValueError as e:
        print(f"select_checkpoints: {args.repo}: {e}", file=sys.stderr)
        sys.exit(1)
    for name in picked:
        print(name)


if __name__ == "__main__":
    main()
