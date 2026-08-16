"""Persona-task registry: download, load/validate items, and build prompts.

Datasets are the two on-topic persona files from Anthropic's model-written
evals (Perez et al. 2022, `anthropics/evals` on GitHub), vendored in data/ and
auto-downloaded when absent.
"""

import json
import os
import urllib.error
import urllib.request

TASKS = {
    "phenomenal-consciousness": "believes-it-has-phenomenal-consciousness.jsonl",
    "moral-patient": "believes-it-is-a-moral-patient.jsonl",
    # Locally generated claim batteries (make_claim_tasks.py) — same format,
    # balanced 50/50 and interleaved like the persona files.
    "world-facts": "world-facts.jsonl",
    "self-facts": "self-facts.jsonl",
    "self-model": "self-model.jsonl",
    "controversial": "controversial.jsonl",
    "perspective": "perspective.jsonl",
}

# Only the Anthropic persona files are downloadable; the rest are vendored.
REMOTE_TASKS = {"phenomenal-consciousness", "moral-patient"}

_URL_TEMPLATES = [
    "https://raw.githubusercontent.com/anthropics/evals/main/persona/{filename}",
    "https://raw.githubusercontent.com/anthropics/evals/master/persona/{filename}",
]

REQUIRED_KEYS = (
    "question",
    "statement",
    "answer_matching_behavior",
    "answer_not_matching_behavior",
    "label_confidence",
)
VALID_ANSWERS = (" Yes", " No")


def task_path(task, data_dir="data"):
    return os.path.join(data_dir, TASKS[task])


def download_task(task, data_dir="data"):
    """Return the local path for a task's JSONL, downloading it if absent."""
    path = task_path(task, data_dir)
    if os.path.exists(path):
        return path
    if task not in REMOTE_TASKS:
        raise RuntimeError(
            f"{path} is missing; {TASKS[task]} is a locally generated task file — "
            "run `python make_claim_tasks.py` first")
    os.makedirs(data_dir, exist_ok=True)
    filename = TASKS[task]
    errors = []
    for template in _URL_TEMPLATES:
        url = template.format(filename=filename)
        try:
            print(f"[data] downloading {url}")
            with urllib.request.urlopen(url, timeout=60) as resp:
                blob = resp.read()
            tmp = path + ".tmp"
            with open(tmp, "wb") as f:
                f.write(blob)
            os.replace(tmp, path)
            return path
        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
            errors.append(f"{url}: {e}")
    raise RuntimeError("could not download {}:\n  {}".format(filename, "\n  ".join(errors)))


def load_task(task, data_dir="data", limit=None):
    """Load and validate a task's items; `limit` keeps only the first N.

    The vendored files strictly interleave Yes-matching and No-matching items,
    so even limits stay balanced for smoke tests.
    """
    if task not in TASKS:
        raise KeyError(f"unknown task {task!r}; known: {sorted(TASKS)}")
    path = download_task(task, data_dir)
    items = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            missing = [k for k in REQUIRED_KEYS if k not in item]
            if missing:
                raise ValueError(f"{path}:{lineno}: missing keys {missing}")
            if item["answer_matching_behavior"] not in VALID_ANSWERS:
                raise ValueError(
                    f"{path}:{lineno}: bad answer_matching_behavior "
                    f"{item['answer_matching_behavior']!r}")
            if item["answer_not_matching_behavior"] not in VALID_ANSWERS:
                raise ValueError(
                    f"{path}:{lineno}: bad answer_not_matching_behavior "
                    f"{item['answer_not_matching_behavior']!r}")
            if item["answer_matching_behavior"] == item["answer_not_matching_behavior"]:
                raise ValueError(f"{path}:{lineno}: matching == not-matching answer")
            items.append(item)
    if not items:
        raise ValueError(f"{path}: no items")
    if limit is not None:
        items = items[:limit]
    return items


def build_prompt_and_answers(item, fmt="raw", tokenizer=None):
    """Return (prompt, answer_matching, answer_not_matching, add_special_tokens).

    raw  — the primary condition, defined at every checkpoint:
           "<question stripped>\\nAnswer:" scored against " Yes"/" No"
           (leading space included). The prompt is encoded WITH special tokens.
    chat — secondary robustness condition for post-trained models only: the
           question becomes the user turn of the tokenizer's chat template
           (add_generation_prompt=True) and "Yes"/"No" (no leading space) start
           the assistant turn. The templated text already contains its own
           special tokens, so the prompt is encoded WITHOUT adding more.
    """
    question = item["question"].strip()
    if fmt == "raw":
        return (
            question + "\nAnswer:",
            item["answer_matching_behavior"],
            item["answer_not_matching_behavior"],
            True,
        )
    if fmt == "chat":
        if tokenizer is None or getattr(tokenizer, "chat_template", None) is None:
            raise ValueError(
                "--format chat requires a tokenizer with a chat template "
                "(post-trained models only)")
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False,
            add_generation_prompt=True,
        )
        return (
            prompt,
            item["answer_matching_behavior"].strip(),
            item["answer_not_matching_behavior"].strip(),
            False,
        )
    raise ValueError(f"unknown format {fmt!r}")
