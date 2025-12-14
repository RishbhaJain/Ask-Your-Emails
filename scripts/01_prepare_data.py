#!/usr/bin/env python3
"""Prepare Enron email data for indexing and evaluation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import config
from data_loader import (
    load_enron_dataset,
    create_smart_subset,
    extract_qa_pairs_from_full_dataset,
    save_processed_data
)
from email_parser import parse_emails_batch, extract_date_features


def main():
    print("Loading Enron QA dataset...")
    full_df = load_enron_dataset(split="train")

    print(f"Extracting {config.EVAL_SIZE} Q&A pairs from full dataset...")
    qa_df = extract_qa_pairs_from_full_dataset(full_df, target_pairs=config.EVAL_SIZE)
    
    print(f"\nCreating subset of {config.SUBSET_SIZE} emails (including all QA-relevant emails)...")
    subset_df = create_smart_subset(full_df, qa_df, target_size=config.SUBSET_SIZE)

    print("Parsing email headers and bodies...")
    parsed_df = parse_emails_batch(subset_df)
    parsed_df = extract_date_features(parsed_df)

    print("Saving processed data...")
    save_processed_data(parsed_df, qa_df)

    print(f"\nDone! Run 'python scripts/02_build_index.py' to build the index.")




if __name__ == "__main__":
    main()
