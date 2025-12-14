from datasets import load_dataset
import pandas as pd
from typing import Tuple
from tqdm import tqdm
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import config

def load_enron_dataset(split="train"):
    """Load Enron QA dataset from HuggingFace"""
    dataset = load_dataset("MichaelR207/enron_qa_0922", split=split)
    df = pd.DataFrame(dataset)
    return df


def extract_qa_pairs_from_full_dataset(full_df, target_pairs=500):
    """Extract Q&A pairs from the full dataset, stratified by user"""
    print(f"\nExtracting Q&A pairs from full dataset")

    qa_pairs = []

    for idx, row in tqdm(full_df.iterrows(), total=len(full_df), desc="Extracting Q&A pairs"):
        questions = row['questions']
        gold_answers = row['gold_answers']

        if not isinstance(questions, list):
            questions = [questions]
        if not isinstance(gold_answers, list):
            gold_answers = [gold_answers]

        for q, a in zip(questions, gold_answers):
            qa_pairs.append({
                'path': row['path'],
                'user': row['user'],
                'question': q,
                'answer': a,
                'email': row['email']
            })

    qa_df = pd.DataFrame(qa_pairs)
    print(f"Extracted {len(qa_df)} individual Q&A pairs")

    # Stratified sampling by user to maintain diversity
    user_groups = qa_df.groupby('user')
    sampled_pairs = []
    
    per_user = max(1, target_pairs // len(user_groups))
    
    for user, group in user_groups:
        n_samples = min(per_user, len(group))
        sampled = group.sample(n=n_samples, random_state=42)
        sampled_pairs.append(sampled)
    
    qa_df = pd.concat(sampled_pairs, ignore_index=True)
    
    if len(qa_df) > target_pairs:
        qa_df = qa_df.sample(n=target_pairs, random_state=42)
    
    print(f"Selected {len(qa_df)} Q&A pairs for evaluation")
    return qa_df


def create_smart_subset(full_df, qa_df, target_size=15000):
    """Create subset that INCLUDES all QA-relevant emails, then fills remainder with random sampling"""
    print(f"\nCreating smart subset with Q&A coverage ")

    # Step 1: Get all emails that have Q&A pairs
    qa_paths = set(qa_df['path'].unique())
    qa_emails = full_df[full_df['path'].isin(qa_paths)].drop_duplicates(subset=['path'])
    
    print(f"Q&A-relevant emails: {len(qa_emails)}")

    # Step 2: Get remaining emails to fill the subset
    remaining_emails = full_df[~full_df['path'].isin(qa_paths)].drop_duplicates(subset=['path'])
    remaining_needed = max(0, target_size - len(qa_emails))
    
    if remaining_needed > 0 and len(remaining_emails) > 0:
        print(f"Filling subset with {remaining_needed} random emails ")
        # Stratified sampling by user for the remainder
        additional = []
        user_counts = remaining_emails['user'].value_counts()
        sample_fraction = min(remaining_needed / len(remaining_emails), 1.0)
        
        for user in user_counts.index:
            user_emails = remaining_emails[remaining_emails['user'] == user]
            n_samples = max(0, int(len(user_emails) * sample_fraction))
            if n_samples > 0:
                user_sample = user_emails.sample(n=min(n_samples, len(user_emails)), random_state=42)
                additional.append(user_sample)
        
        if additional:
            additional_df = pd.concat(additional, ignore_index=True)
            if len(additional_df) > remaining_needed:
                additional_df = additional_df.sample(n=remaining_needed, random_state=42)
            subset_df = pd.concat([qa_emails, additional_df], ignore_index=True)
        else:
            subset_df = qa_emails
    else:
        subset_df = qa_emails

    # Remove duplicates by path
    subset_df = subset_df.drop_duplicates(subset=['path'])
    
    print(f"Subset size: {len(subset_df)} emails")
    print(f"Q&A coverage: {len(qa_emails)} / {len(qa_df)} pairs")
    print(f"Unique users: {subset_df['user'].nunique()}")

    subset_df['folder'] = subset_df['path'].apply(extract_folder_from_path)
    
    return subset_df


def extract_folder_from_path(path):
    """Extract folder name from path like 'phanis-s/sent_items/4' -> 'sent_items'"""
    parts = path.split('/')
    if len(parts) >= 2:
        return parts[1]
    return "unknown"


def save_processed_data(emails_df, qa_df):
    """Save processed data to parquet and CSV"""
    emails_path = config.PROCESSED_DIR / "emails_subset.parquet"
    qa_path = config.PROCESSED_DIR / "eval_qa_pairs.parquet"

    print(f"\nSaving processed data ")
    print(f"  Emails (parquet): {emails_path}")
    print(f"  Q&A pairs (parquet): {qa_path}")

    emails_df.to_parquet(emails_path, index=False)
    qa_df.to_parquet(qa_path, index=False)

    # save CSV too for easy viewing
    emails_csv_path = config.PROCESSED_DIR / "emails_subset.csv"
    qa_csv_path = config.PROCESSED_DIR / "eval_qa_pairs.csv"
    emails_df.to_csv(emails_csv_path, index=False)
    qa_df.to_csv(qa_csv_path, index=False)
    print(f"  Emails (CSV): {emails_csv_path}")
    print(f"  Q&A pairs (CSV): {qa_csv_path}")

    print("\nData saved successfully")

    print(f"\nDataset Summary:")
    print(f"  Total emails: {len(emails_df)}")
    print(f"  Unique users: {emails_df['user'].nunique()}")
    print(f"  Evaluation Q&A pairs: {len(qa_df)}")
    print(f"  Email columns: {list(emails_df.columns)}")
