#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

ANATOMIES = ["AB", "HN", "TH"]
SPLIT_NAMES = ("train", "val", "test")
RATIOS = {"train": 0.7, "val": 0.1, "test": 0.2}
CASE_PATTERN = re.compile(r"^([12])(AB|HN|TH)([A-Za-z])(.+)$")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=".", help="SynthRAD2025 根目录")
    parser.add_argument("--seed", type=int, default=3407, help="随机种子")
    parser.add_argument("--force", action="store_true", help="覆盖已有输出目录")
    return parser.parse_args()


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def safe_remove(path: Path):
    if not path.exists() and not path.is_symlink():
        return
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def create_symlink(src: Path, dst: Path):
    if dst.exists() or dst.is_symlink():
        safe_remove(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.symlink_to(src.resolve(), target_is_directory=True)


def expected_files(task_name: str):
    if task_name == "task1":
        return ["mr.mha", "ct.mha", "mask.mha"]
    elif task_name == "task2":
        return ["cbct.mha", "ct.mha", "mask.mha"]
    else:
        raise ValueError(f"未知任务: {task_name}")


def parse_case_name(case_name: str):
    m = CASE_PATTERN.match(case_name)
    if m is None:
        return None
    task_digit, anatomy, center, patient_id = m.groups()
    return {
        "task_digit": task_digit,
        "anatomy": anatomy,
        "center": center.upper(),
        "patient_id": patient_id,
    }


def should_skip_dir(case_dir: Path):
    name = case_dir.name.lower()
    if name == "overviews":
        return True
    if not case_dir.is_dir():
        return True
    return False


def discover_cases(task_name: str, task_root: Path):
    task_digit_expected = "1" if task_name == "task1" else "2"
    req_files = expected_files(task_name)

    cases = []
    warnings = []

    for anatomy in ANATOMIES:
        anatomy_dir = task_root / anatomy
        if not anatomy_dir.exists():
            warnings.append(f"[WARN] 缺少 anatomy 目录: {anatomy_dir}")
            continue

        for case_dir in sorted(anatomy_dir.iterdir()):
            if should_skip_dir(case_dir):
                continue

            parsed = parse_case_name(case_dir.name)
            if parsed is None:
                warnings.append(f"[WARN] 跳过非标准病例目录: {case_dir}")
                continue

            if parsed["task_digit"] != task_digit_expected:
                warnings.append(
                    f"[WARN] 任务号不匹配，已跳过: {case_dir.name}, "
                    f"parsed_task={parsed['task_digit']}, expected_task={task_digit_expected}"
                )
                continue

            if parsed["anatomy"] != anatomy:
                warnings.append(
                    f"[WARN] anatomy 不一致，已跳过: {case_dir.name}, "
                    f"name_anatomy={parsed['anatomy']}, dir_anatomy={anatomy}"
                )
                continue

            missing = [f for f in req_files if not (case_dir / f).exists()]
            if missing:
                warnings.append(
                    f"[WARN] 跳过缺少文件的病例目录: {case_dir} -> missing={missing}"
                )
                continue

            cases.append({
                "case_id": case_dir.name,
                "task": task_name,
                "task_digit": parsed["task_digit"],
                "anatomy": parsed["anatomy"],
                "center": parsed["center"],
                "patient_id": parsed["patient_id"],
                "src": str(case_dir.resolve())
            })

    return cases, warnings


def apportion_counts(n, ratios):
    raw = {
        "train": n * ratios["train"],
        "val": n * ratios["val"],
        "test": n * ratios["test"],
    }
    counts = {k: math.floor(v) for k, v in raw.items()}
    remain = n - sum(counts.values())

    frac_order = sorted(
        SPLIT_NAMES,
        key=lambda k: raw[k] - counts[k],
        reverse=True
    )
    for i in range(remain):
        counts[frac_order[i % len(frac_order)]] += 1

    if n >= 5 and counts["val"] == 0 and counts["train"] > 1:
        counts["train"] -= 1
        counts["val"] += 1
    if n >= 3 and counts["test"] == 0 and counts["train"] > 1:
        counts["train"] -= 1
        counts["test"] += 1

    total = sum(counts.values())
    if total != n:
        counts["train"] += (n - total)

    return counts


def split_within_anatomy_by_center(cases, seed):
    """
    对单个 task 内的数据：
    按 anatomy 分组；
    每个 anatomy 内再按 center 分层；
    然后做 7:1:2。
    """
    rng = random.Random(seed)

    result = {}
    all_task_splits = {"train": [], "val": [], "test": []}

    for anatomy in ANATOMIES:
        anatomy_cases = [x for x in cases if x["anatomy"] == anatomy]
        center_buckets = defaultdict(list)

        for item in anatomy_cases:
            center_buckets[item["center"]].append(item)

        split_map = {"train": [], "val": [], "test": []}
        summary = {}

        for center in sorted(center_buckets.keys()):
            items = center_buckets[center][:]
            rng.shuffle(items)

            n = len(items)
            counts = apportion_counts(n, RATIOS)

            n_train = counts["train"]
            n_val = counts["val"]
            n_test = counts["test"]

            train_items = items[:n_train]
            val_items = items[n_train:n_train + n_val]
            test_items = items[n_train + n_val:n_train + n_val + n_test]

            split_map["train"].extend(train_items)
            split_map["val"].extend(val_items)
            split_map["test"].extend(test_items)

            summary[center] = {
                "total": n,
                "train": len(train_items),
                "val": len(val_items),
                "test": len(test_items),
            }

        result[anatomy] = {
            "summary_by_center": summary,
            "train": split_map["train"],
            "val": split_map["val"],
            "test": split_map["test"],
        }

        all_task_splits["train"].extend(split_map["train"])
        all_task_splits["val"].extend(split_map["val"])
        all_task_splits["test"].extend(split_map["test"])

    return result, all_task_splits


def write_split_json(out_json: Path, seed: int, task_results: dict):
    data = {
        "seed": seed,
        "ratios": RATIOS,
        "tasks": task_results
    }
    ensure_dir(out_json.parent)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def prepare_output_dirs(root: Path, force: bool):
    managed_paths = [
        root / "raw",
        root / "task1",
        root / "task2",
        root / "splits",
    ]
    exists_any = any(p.exists() or p.is_symlink() for p in managed_paths)

    if exists_any and not force:
        raise FileExistsError(
            "检测到输出目录已存在。若要覆盖，请加 --force\n"
            f"将处理的目录: {[str(p) for p in managed_paths]}"
        )

    if exists_any and force:
        for p in managed_paths:
            safe_remove(p)


def build_layout(root: Path, seed: int, force: bool):
    task1_src_root = root / "synthRAD2025_Task1_Train" / "Task1"
    task2_src_root = root / "synthRAD2025_Task2_Train" / "Task2"

    if not task1_src_root.exists():
        raise FileNotFoundError(f"未找到目录: {task1_src_root}")
    if not task2_src_root.exists():
        raise FileNotFoundError(f"未找到目录: {task2_src_root}")

    prepare_output_dirs(root, force=force)

    raw_dir = root / "raw"
    split_json = root / "splits" / f"synthrad_split_seed{seed}.json"

    # 扫描病例
    task1_cases, task1_warnings = discover_cases("task1", task1_src_root)
    task2_cases, task2_warnings = discover_cases("task2", task2_src_root)

    # 按 anatomy + center 划分
    task1_by_anatomy, task1_all = split_within_anatomy_by_center(task1_cases, seed)
    task2_by_anatomy, task2_all = split_within_anatomy_by_center(task2_cases, seed)

    # 写 split.json
    task_results = {
        "task1": {
            "source_root": str(task1_src_root.resolve()),
            "total_cases": len(task1_cases),
            "anatomies": task1_by_anatomy,
            "all": task1_all,
        },
        "task2": {
            "source_root": str(task2_src_root.resolve()),
            "total_cases": len(task2_cases),
            "anatomies": task2_by_anatomy,
            "all": task2_all,
        }
    }
    write_split_json(split_json, seed, task_results)

    # raw/task1/{AB,HN,TH}, raw/task2/{AB,HN,TH}
    for anatomy in ANATOMIES:
        create_symlink(task1_src_root / anatomy, raw_dir / "task1" / anatomy)
        create_symlink(task2_src_root / anatomy, raw_dir / "task2" / anatomy)

    # 创建 task1/HN/train... 这种结构
    for task_name, anatomy_result in [("task1", task1_by_anatomy), ("task2", task2_by_anatomy)]:
        for anatomy in ANATOMIES:
            for split_name in SPLIT_NAMES:
                split_dir = root / task_name / anatomy / split_name
                ensure_dir(split_dir)
                for item in anatomy_result[anatomy][split_name]:
                    src = Path(item["src"])
                    dst = split_dir / item["case_id"]
                    create_symlink(src, dst)

    # 打印结果
    print("=" * 80)
    print("处理完成")
    print(f"root:       {root.resolve()}")
    print(f"split json: {split_json.resolve()}")
    print("-" * 80)

    for task_name, anatomy_result, warnings in [
        ("task1", task1_by_anatomy, task1_warnings),
        ("task2", task2_by_anatomy, task2_warnings),
    ]:
        print(f"[{task_name}]")
        for anatomy in ANATOMIES:
            train_n = len(anatomy_result[anatomy]["train"])
            val_n = len(anatomy_result[anatomy]["val"])
            test_n = len(anatomy_result[anatomy]["test"])
            total_n = train_n + val_n + test_n
            print(f"  [{anatomy}] total={total_n}, train={train_n}, val={val_n}, test={test_n}")
            for center, v in anatomy_result[anatomy]["summary_by_center"].items():
                print(
                    f"    center={center}: total={v['total']}, "
                    f"train={v['train']}, val={v['val']}, test={v['test']}"
                )

        if warnings:
            print("  warnings:")
            for w in warnings[:20]:
                print(f"    {w}")
            if len(warnings) > 20:
                print(f"    ... 还有 {len(warnings) - 20} 条 warning")
        print("-" * 80)

    # 交叉检查
    for task_name in ["task1", "task2"]:
        print(f"[检查 {task_name} 各 anatomy 无交叉]")
        for anatomy in ANATOMIES:
            train_set = set(p.name for p in (root / task_name / anatomy / "train").iterdir())
            val_set = set(p.name for p in (root / task_name / anatomy / "val").iterdir())
            test_set = set(p.name for p in (root / task_name / anatomy / "test").iterdir())
            print(f"  {anatomy}: train∩val={len(train_set & val_set)}, "
                  f"train∩test={len(train_set & test_set)}, val∩test={len(val_set & test_set)}")
        print("-" * 80)


def main():
    args = parse_args()
    root = Path(args.root).resolve()
    build_layout(root=root, seed=args.seed, force=args.force)


if __name__ == "__main__":
    main()