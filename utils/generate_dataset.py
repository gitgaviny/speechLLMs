#!/usr/bin/env python3
"""
Created by Hao at 2025-07-01

dataset_builder.py – Build a Hugging Face Dataset from LibriMix‑style files.

Usage examples
--------------
Default (mini subset, "wav.scp")::

    python dataset_builder.py --number 3

Mini subset but different scp name::

    python dataset_builder.py --number 3 --wav_scp_name wav_clean.scp

No suffix (train_3mix/)::

    python dataset_builder.py --number 3 --suffix ''

Suffix "_clean" and scp "wav_clean.scp"::

    python dataset_builder.py --number 2 --suffix _clean --wav_scp_name wav_clean.scp

If ``suffix`` (after stripping a leading underscore) equals ``"mini"``, the script
uses the *train* directory for all three splits (train/validation/test). In all
other cases it follows the standard LibriMix directory scheme (*train* →
``train_<N>mix{suffix}``, *validation* → ``dev_<N>mix{suffix}``, *test* →
``test_<N>mix{suffix}``).
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Dict

import pandas as pd
from datasets import Audio, Dataset, DatasetDict

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Build a Hugging Face Dataset from LibriMix-style files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--base_data_path",
        type=str,
        default="/lustre/users/shi/toolkits/espnet/egs2/librimix/sot_asr1/data",
        help="Root directory of LibriMix data.",
    )
    parser.add_argument(
        "--number",
        type=str,
        choices=["2", "3"],
        default="3",
        help="Number of overlapping speakers (2 or 3).",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="_mini",
        help="Suffix appended after `{number}mix`, e.g. `_mini`, `_clean`, or ''",
    )
    parser.add_argument(
        "--wav_scp_name",
        type=str,
        default="wav.scp",
        help="Name of the SCP file (e.g. `wav.scp`, `wav_clean.scp`).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory; defaults to `libri{number}mix{suffix}` if omitted.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="TRANSCRIBE THE PROVIDED AUDIO INTO ACCURATE TEXT",
        help="The prompt for systems.",
    )

    return parser.parse_args()

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def build_file_paths(
    *,
    base: str,
    num: str,
    suffix: str,
    wav_name: str,
) -> Dict[str, Dict[str, str]]:
    """Construct wav.scp & text paths for each split with custom rules."""

    suffix_stripped = suffix.lstrip("_")
    use_train_for_all = suffix_stripped == "mini"

    # Suffix is NOT appended for noisy/clean; otherwise append if not empty.
    append_suffix = suffix and suffix_stripped not in {"noisy", "clean"}

    prefix_map = {
        "train": "train",
        "validation": "train" if use_train_for_all else "dev",
        "test": "train" if use_train_for_all else "test",
    }

    paths: Dict[str, Dict[str, str]] = {}
    for split, prefix in prefix_map.items():
        dir_name = f"{prefix}_{num}mix{suffix if append_suffix else ''}"
        paths[split] = {
            "wav_scp": os.path.join(base, dir_name, wav_name),
            "text": os.path.join(base, dir_name, "text"),
        }
        LOGGER.info("[%s] wav_scp → %s", split, paths[split]["wav_scp"])
        LOGGER.info("[%s] text    → %s", split, paths[split]["text"])

    return paths

# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def parse_line(line: str):
    parts = line.strip().split(" ", 1)
    return parts[0], parts[1] if len(parts) > 1 else ""


def process_split(wav_scp_path: str, text_path: str, prompt: str) -> Dataset:
    """Convert a pair of wav.scp & text files into a HF *Dataset*."""

    with open(wav_scp_path, "r") as f:
        audio_df = pd.DataFrame([parse_line(l) for l in f], columns=["id", "path"])

    with open(text_path, "r") as f:
        text_df = pd.DataFrame([parse_line(l) for l in f], columns=["id", "text"])

    merged = pd.merge(audio_df, text_df, on="id", how="inner")
    ds = Dataset.from_list(
        [
            {"id": row.id, "audio": row.path, "text": row.text, "prompt": prompt}
            for row in merged.itertuples(index=False)
        ]
    )

    return ds.cast_column("audio", Audio(sampling_rate=16_000))

# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s — %(levelname)s — %(message)s",
    )

    output_dir = args.output_dir or f"libri{args.number}mix{args.suffix or ''}"

    # Resolve all paths
    file_paths = build_file_paths(
        base=args.base_data_path,
        num=args.number,
        suffix=args.suffix,
        wav_name=args.wav_scp_name,
    )

    # Build DatasetDict
    ds_dict = DatasetDict(
        {
            split: process_split(paths["wav_scp"], paths["text"], args.prompt)
            for split, paths in file_paths.items()
        }
    )

    # Persist & report
    ds_dict.save_to_disk(output_dir)
    logging.info("Dataset saved to %s", output_dir)
    logging.info(ds_dict)
    logging.info("First training sample: %s", ds_dict["train"][0])


if __name__ == "__main__":
    main()

