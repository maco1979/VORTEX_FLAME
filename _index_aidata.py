"""
Index E:\\AI_Data datasets into VORTEX_FLAME soul knowledge bases.

Datasets:
  1. Capybara (16K multi-turn conversations, STEM/logic/roleplay)
     → einstein (science/logic), guizhu (philosophy/roleplay)
  2. Opus-4.6-Reasoning (2.16K math/reasoning problems with CoT)
     → einstein (mathematical reasoning patterns)
  3. rStar-Coder (418K competitive code problems + solutions)
     → cezanne (code reasoning patterns)

Strategy:
  - Capybara: extract Q&A pairs, tag by domain (physics/math/philosophy/etc)
  - Opus-4.6: extract problem+thinking+solution chains
  - rStar-Coder: extract problem+solution pairs from seed_sft (20 shards)
  - Batch write with rate limiting
"""

import os
import sys
import time

import pyarrow.ipc as ipc
import pyarrow.parquet as pq

sys.path.insert(0, r"D:\VORTEX_FLAME")
from soul_memory import write

BATCH_SIZE = 500


def index_capybara():
    print("\n" + "=" * 60)
    print("  Indexing Capybara Dataset")
    print("=" * 60)

    arrow_path = r"E:\AI_Data\Capybara\LDJnr___capybara\default\0.0.0\c2bc39ac72f24748f60f5fb55b77e08fb0660ba6\capybara-train.arrow"

    with open(arrow_path, "rb") as f:
        reader = ipc.RecordBatchStreamReader(f)
        table = reader.read_all()

    rows = len(table)
    print(f"Total conversations: {rows}")

    STEM_KEYWORDS = {
        "physics": "einstein",
        "math": "einstein",
        "chemistry": "einstein",
        "biology": "einstein",
        "logic": "einstein",
        "programming": "cezanne",
        "code": "cezanne",
        "algorithm": "cezanne",
        "philosophy": "guizhu",
        "ethics": "guizhu",
        "moral": "guizhu",
        "culture": "guizhu",
        "history": "herodotus",
        "art": "monet",
        "music": "beethoven",
    }

    stats = {}
    count = 0

    for i in range(rows):
        source = str(table.column("source")[i].as_py())
        conv = table.column("conversation")[i].as_py()

        if not conv or len(conv) == 0:
            continue

        first_turn = conv[0] if isinstance(conv, list) else conv
        input_text = first_turn.get("input", "") if isinstance(first_turn, dict) else str(first_turn)
        output_text = first_turn.get("output", "") if isinstance(first_turn, dict) else ""

        if not input_text:
            continue

        input_lower = input_text.lower()

        target_soul = "einstein"
        domain_tag = "general"
        for keyword, soul in STEM_KEYWORDS.items():
            if keyword in input_lower:
                target_soul = soul
                domain_tag = keyword
                break

        content = {
            "topic": f"Capybara: {input_text[:100]}",
            "source": "capybara_dataset",
            "conversation_source": source,
            "question": input_text[:4000],
            "answer": output_text[:4000],
            "turns": len(conv) if isinstance(conv, list) else 1,
            "domain": domain_tag,
        }

        if len(output_text) > 4000:
            content["answer_extended"] = output_text[4000:8000]

        try:
            write(
                soul=target_soul,
                category="domain_memory",
                content=content,
                importance=0.6,
                tags=["capybara", domain_tag, "multi_turn"],
            )
            stats[target_soul] = stats.get(target_soul, 0) + 1
        except Exception as e:
            pass

        count += 1
        if count % BATCH_SIZE == 0:
            print(f"  Progress: {count}/{rows}")
            time.sleep(0.1)

    print(f"  Capybara indexed: {count} conversations")
    for soul, n in sorted(stats.items()):
        print(f"    {soul}: {n}")


def index_opus_reasoning():
    print("\n" + "=" * 60)
    print("  Indexing Opus-4.6-Reasoning Dataset")
    print("=" * 60)

    parquet_path = r"E:\AI_Data\Opus-4.6-Reasoning\data\train-00000-of-00001.parquet"
    table = pq.read_table(parquet_path)
    rows = len(table)
    print(f"Total problems: {rows}")

    indexed = 0
    errors = 0

    for i in range(rows):
        problem = str(table.column("problem")[i].as_py() or "")
        thinking = str(table.column("thinking")[i].as_py() or "")
        solution = str(table.column("solution")[i].as_py() or "")
        difficulty = str(table.column("difficulty")[i].as_py() or "")
        category = str(table.column("category")[i].as_py() or "")
        problem_id = str(table.column("id")[i].as_py() or "")

        if not problem or not solution:
            continue

        content = {
            "topic": f"Reasoning: {problem[:100]}",
            "source": "opus_4.6_reasoning",
            "problem_id": problem_id,
            "problem": problem[:4000],
            "thinking_chain": thinking[:6000],
            "solution": solution[:4000],
            "difficulty": difficulty,
            "category": category,
        }

        if len(problem) > 4000:
            content["problem_extended"] = problem[4000:]
        if len(thinking) > 6000:
            content["thinking_extended"] = thinking[6000:12000]
        if len(solution) > 4000:
            content["solution_extended"] = solution[4000:8000]

        try:
            write(
                soul="einstein",
                category="domain_memory",
                content=content,
                importance=0.75 if difficulty == "hard" else 0.65,
                tags=["reasoning", "math", difficulty, category, "cot"],
            )
            indexed += 1
        except Exception as e:
            errors += 1

        if (i + 1) % 200 == 0:
            print(f"  Progress: {i+1}/{rows}")
            time.sleep(0.05)

    print(f"  Opus-4.6 indexed: {indexed}, errors: {errors}")


def index_rstar_coder():
    print("\n" + "=" * 60)
    print("  Indexing rStar-Coder seed_sft Dataset")
    print("=" * 60)

    seed_dir = r"E:\AI_Data\rStar-Coder\seed_sft"
    shard_files = sorted([f for f in os.listdir(seed_dir) if f.endswith(".parquet")])
    print(f"Found {len(shard_files)} shards")

    indexed = 0
    errors = 0
    total_rows = 0

    for shard_idx, shard_file in enumerate(shard_files):
        shard_path = os.path.join(seed_dir, shard_file)
        table = pq.read_table(shard_path)
        rows = len(table)
        total_rows += rows

        for i in range(rows):
            try:
                row = {}
                for col_name in table.column_names:
                    val = table.column(col_name)[i].as_py()
                    row[col_name] = val
            except Exception:
                errors += 1
                continue

            problem_text = str(row.get("problem", row.get("input", "")) or "")
            solution_text = str(row.get("solution", row.get("output", row.get("answer", ""))) or "")

            if not problem_text:
                continue

            content = {
                "topic": f"Code: {problem_text[:80]}",
                "source": "rstar_coder",
                "problem": problem_text[:4000],
                "solution": solution_text[:6000],
            }

            if len(problem_text) > 4000:
                content["problem_extended"] = problem_text[4000:]
            if len(solution_text) > 6000:
                content["solution_extended"] = solution_text[6000:12000]

            for key in ["difficulty", "language", "category", "tags", "verified", "is_passed"]:
                if key in row and row[key] is not None:
                    content[key] = str(row[key])[:200]

            try:
                write(
                    soul="cezanne",
                    category="code_memory",
                    content=content,
                    importance=0.7,
                    tags=["rstar_coder", "competitive_programming", "code_reasoning"],
                )
                indexed += 1
            except Exception as e:
                errors += 1

        print(f"  Shard {shard_idx+1}/{len(shard_files)} ({shard_file}): {rows} rows")
        time.sleep(0.2)

    print(f"  rStar-Coder indexed: {indexed}/{total_rows}, errors: {errors}")


def main():
    print("=" * 60)
    print("  E:\\AI_Data → VORTEX_FLAME Soul Knowledge Bases")
    print("=" * 60)

    index_capybara()
    index_opus_reasoning()
    index_rstar_coder()

    print("\n" + "=" * 60)
    print("  All E:\\AI_Data indexing complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
