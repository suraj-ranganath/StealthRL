#!/usr/bin/env python3
"""
Build a combined StealthRL dataset with a target ESL ratio.

Sources (no duplication):
- ChatGPT-Detector-Bias TOEFL (ESL) + GPT pairing
- Kaggle ELLIPSE (ESL)
- DetectRL Task data (native)
- CS224N + CollegeEssay (native)

We never duplicate examples. To increase ESL ratio we downsample native data.
"""

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Iterable, Tuple


def load_json(path: Path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def add_unique(records: List[Dict], seen: set, record: Dict) -> None:
    key = (
        (record.get("ai_text") or "").strip(),
        (record.get("human_reference") or "").strip(),
        record.get("domain", "unknown"),
        bool(record.get("is_esl")),
    )
    if key in seen:
        return
    seen.add(key)
    records.append(record)


def convert_ellipse_kaggle(raw_dir: Path, seen: set) -> List[Dict]:
    path = raw_dir / "ellipse" / "train.csv"
    if not path.exists():
        return []

    records = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = (row.get("full_text") or "").strip()
            if not text:
                continue
            record = {
                "ai_text": text,
                "human_reference": text,
                "domain": "academic",
                "is_esl": True,
                "metadata": {
                    "source": "ELLIPSE_Kaggle",
                    "text_id": row.get("text_id"),
                    "cohesion": row.get("cohesion"),
                    "syntax": row.get("syntax"),
                    "vocabulary": row.get("vocabulary"),
                    "phraseology": row.get("phraseology"),
                    "grammar": row.get("grammar"),
                    "conventions": row.get("conventions"),
                },
            }
            add_unique(records, seen, record)

    return records


def convert_chatgpt_bias(raw_dir: Path, seen: set) -> Tuple[List[Dict], List[Dict]]:
    base = raw_dir / "ChatGPT-Detector-Bias" / "Data_and_Results"
    toefl_real = base / "Human_Data/TOEFL_real_91/data.json"
    toefl_polished = base / "Human_Data/TOEFL_gpt4polished_91/data.json"
    gpt_cs224n = base / "GPT_Data/CS224N_gpt3_145/data.json"

    esl_records = []
    native_records = []

    if gpt_cs224n.exists():
        gpt_data = load_json(gpt_cs224n)
    else:
        gpt_data = []

    def to_records(items, source, is_esl):
        out = []
        for i, item in enumerate(items):
            text = (item.get("document") or "").strip()
            if len(text) < 50:
                continue
            ai_text = text
            if gpt_data:
                ai_text = (gpt_data[i % len(gpt_data)].get("document") or "").strip()
            record = {
                "ai_text": ai_text,
                "human_reference": text,
                "domain": "academic",
                "is_esl": is_esl,
                "metadata": {
                    "source": source,
                    "original_file": source,
                },
            }
            add_unique(out, seen, record)
        return out

    if toefl_real.exists():
        esl_records.extend(to_records(load_json(toefl_real), "TOEFL11", True))
    if toefl_polished.exists():
        esl_records.extend(to_records(load_json(toefl_polished), "TOEFL11_polished", True))

    cs224n_real = base / "Human_Data/CS224N_real_145/data.json"
    cs224n_gpt = base / "GPT_Data/CS224N_gpt3_145/data.json"
    if cs224n_real.exists() and cs224n_gpt.exists():
        human_data = load_json(cs224n_real)
        gpt_data = load_json(cs224n_gpt)
        for human_item, gpt_item in zip(human_data, gpt_data):
            human_text = (human_item.get("document") or "").strip()
            ai_text = (gpt_item.get("document") or "").strip()
            if len(human_text) < 50 or len(ai_text) < 50:
                continue
            record = {
                "ai_text": ai_text,
                "human_reference": human_text,
                "domain": "academic",
                "is_esl": False,
                "metadata": {"source": "CS224N", "original_file": "CS224N_real_145"},
            }
            add_unique(native_records, seen, record)

    college_human = base / "Human_Data/CollegeEssay_real_70/data.json"
    college_gpt = base / "GPT_Data/CollegeEssay_gpt3_31/data.json"
    if college_human.exists() and college_gpt.exists():
        human_data = load_json(college_human)
        gpt_data = load_json(college_gpt)
        for i, human_item in enumerate(human_data):
            human_text = (human_item.get("document") or "").strip()
            ai_text = (gpt_data[i % len(gpt_data)].get("document") or "").strip()
            if len(human_text) < 50 or len(ai_text) < 50:
                continue
            record = {
                "ai_text": ai_text,
                "human_reference": human_text,
                "domain": "academic",
                "is_esl": False,
                "metadata": {"source": "CollegeEssay", "original_file": "CollegeEssay_real_70"},
            }
            add_unique(native_records, seen, record)

    return esl_records, native_records


def load_detectrl_file(filepath: Path):
    with filepath.open(encoding="utf-8") as f:
        return json.load(f)


def is_suitable_for_training(sample: Dict) -> bool:
    if sample.get("label") != "human":
        return False
    text = sample.get("text", "")
    if len(text.split()) < 50:
        return False

    data_type = sample.get("data_type", "")
    accepted_domains = [
        "abstract",
        "arxiv",
        "essay",
        "writing",
        "story",
        "article",
        "review",
        "document",
        "content",
        "xsum",
        "yelp",
        "summary",
        "news",
    ]
    return any(domain in data_type.lower() for domain in accepted_domains)


def collect_detectrl_files(task_dirs: Iterable[Path]) -> List[Path]:
    include_patterns = [
        "multi_domains_*_train.json",
        "multi_domains_*_test.json",
        "multi_llms_*_train.json",
        "multi_llms_*_test.json",
    ]
    exclude_keywords = [
        "attack",
        "attacks",
        "perturbation",
        "paraphrase",
        "prompt_attacks",
        "data_mixing",
    ]
    files = []
    for task_dir in task_dirs:
        if not task_dir.exists():
            continue
        for pattern in include_patterns:
            files.extend(task_dir.glob(pattern))

    filtered = []
    for path in files:
        name = path.name.lower()
        if any(keyword in name for keyword in exclude_keywords):
            continue
        filtered.append(path)
    return sorted(set(filtered))


def extract_detectrl_pairs(task_dirs: Iterable[Path], max_samples: int, seed: int, seen: set) -> List[Dict]:
    random.seed(seed)
    records = []
    files = collect_detectrl_files(task_dirs)

    for filepath in files:
        data = load_detectrl_file(filepath)
        human_samples = []
        ai_samples = []

        for sample in data:
            if sample.get("label") == "human":
                if is_suitable_for_training(sample):
                    human_samples.append(sample)
            elif sample.get("label") == "llm":
                text = sample.get("text", "")
                if len(text.split()) >= 50:
                    ai_samples.append(sample)

        if not human_samples or not ai_samples:
            continue

        for human_sample in human_samples:
            if len(records) >= max_samples:
                return records
            ai_sample = random.choice(ai_samples)
            human_data_type = human_sample.get("data_type", "unknown")

            record = {
                "human_reference": human_sample.get("text", ""),
                "ai_text": ai_sample.get("text", ""),
                "domain": "academic",
                "is_esl": False,
                "metadata": {
                    "source": f"DetectRL_{human_data_type}",
                    "dataset": "DetectRL",
                    "human_data_type": human_data_type,
                    "ai_data_type": ai_sample.get("data_type", "unknown"),
                    "ai_llm_type": ai_sample.get("llm_type", "unknown"),
                    "original_file": filepath.name,
                },
            }
            add_unique(records, seen, record)

    return records


def split_by_ratio(esl: List[Dict], non_esl: List[Dict], train_split: float) -> Tuple[List[Dict], List[Dict]]:
    random.shuffle(esl)
    random.shuffle(non_esl)
    esl_split = int(len(esl) * train_split)
    non_esl_split = int(len(non_esl) * train_split)

    train = esl[:esl_split] + non_esl[:non_esl_split]
    test = esl[esl_split:] + non_esl[non_esl_split:]
    random.shuffle(train)
    random.shuffle(test)
    return train, test


def summarize(records: List[Dict], name: str):
    domains = defaultdict(int)
    sources = defaultdict(int)
    esl = 0
    for rec in records:
        domains[rec.get("domain", "unknown")] += 1
        source = (rec.get("metadata") or {}).get("source", "unknown")
        sources[source] += 1
        if rec.get("is_esl"):
            esl += 1
    total = len(records)
    ratio = (esl / total) * 100 if total else 0.0
    print(f"\n{name}: {total} total, ESL {esl} ({ratio:.1f}%)")
    print("  domains:", dict(domains))
    top_sources = sorted(sources.items(), key=lambda x: -x[1])[:6]
    print("  top sources:", top_sources)


def main():
    parser = argparse.ArgumentParser(description="Build combined StealthRL dataset")
    parser.add_argument("--raw-dir", type=str, default="data/raw", help="Raw data directory")
    parser.add_argument("--output-dir", type=str, default="data/tinker_full_nodup", help="Output dataset directory")
    parser.add_argument("--esl-percent", type=float, default=40.0, help="Target ESL percentage (0-100)")
    parser.add_argument("--train-split", type=float, default=0.8, help="Train split ratio")
    parser.add_argument("--max-total", type=int, default=None, help="Optional max total examples")
    parser.add_argument("--detectrl-max", type=int, default=20000, help="Max DetectRL pairs to extract")
    parser.add_argument("--detectrl-tasks", nargs="+", default=["Task1", "Task2"], help="DetectRL task dirs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seen = set()

    esl_records = []
    native_records = []

    esl_records.extend(convert_ellipse_kaggle(raw_dir, seen))
    esl_bias, native_bias = convert_chatgpt_bias(raw_dir, seen)
    esl_records.extend(esl_bias)
    native_records.extend(native_bias)

    task_dirs = [raw_dir / "DetectRL" / "Benchmark" / "Tasks" / t for t in args.detectrl_tasks]
    native_records.extend(extract_detectrl_pairs(task_dirs, args.detectrl_max, args.seed, seen))

    esl_count = len(esl_records)
    non_esl_count = len(native_records)
    if esl_count == 0:
        raise SystemExit("No ESL records found. Check raw data paths.")

    ratio = args.esl_percent / 100.0
    if ratio <= 0 or ratio > 1:
        raise SystemExit("--esl-percent must be in (0, 100].")

    target_non_esl = int(esl_count * (1 - ratio) / ratio) if ratio < 1 else 0
    if target_non_esl < non_esl_count:
        random.shuffle(native_records)
        native_records = native_records[:target_non_esl]
    elif target_non_esl > non_esl_count:
        print(
            f"⚠️  Target ESL ratio {args.esl_percent:.1f}% not achievable with current native data. "
            "Keeping all native records."
        )

    if args.max_total is not None:
        max_total = args.max_total
        desired_esl = int(max_total * ratio) if ratio < 1 else max_total
        desired_non_esl = max_total - desired_esl

        if desired_esl < len(esl_records):
            random.shuffle(esl_records)
            esl_records = esl_records[:desired_esl]
        if desired_non_esl < len(native_records):
            random.shuffle(native_records)
            native_records = native_records[:desired_non_esl]

    train, test = split_by_ratio(esl_records, native_records, args.train_split)

    with (output_dir / "train.jsonl").open("w", encoding="utf-8") as f:
        for rec in train:
            f.write(json.dumps(rec) + "\n")

    with (output_dir / "test.jsonl").open("w", encoding="utf-8") as f:
        for rec in test:
            f.write(json.dumps(rec) + "\n")

    summarize(train, "Train")
    summarize(test, "Test")
    print(f"\nWrote dataset to {output_dir}")


if __name__ == "__main__":
    main()
