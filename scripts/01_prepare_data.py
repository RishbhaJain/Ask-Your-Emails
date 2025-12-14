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
    full_df = load_enron_dataset(split="train")

    qa_df = extract_qa_pairs_from_full_dataset(full_df, target_pairs=config.EVAL_SIZE)
    
    subset_df = create_smart_subset(full_df, qa_df, target_size=config.SUBSET_SIZE)

    parsed_df = parse_emails_batch(subset_df)
    parsed_df = extract_date_features(parsed_df)

    save_processed_data(parsed_df, qa_df)




if __name__ == "__main__":
    main()
