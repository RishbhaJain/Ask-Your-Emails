#!/usr/bin/env python3
"""
Check how many QA pairs have answers that exist in the email dataset
"""
import pandas as pd
import re

# Load data
emails_df = pd.read_csv('data/processed/emails_subset.csv')
qa_df = pd.read_csv('data/processed/eval_qa_pairs.csv')

print(f"Checking {len(qa_df)} Q&A pairs against {len(emails_df)} emails...")
print()

# Create searchable text
emails_df['searchable'] = (
    emails_df['subject'].fillna('') + ' ' + 
    emails_df['body'].fillna('')
).str.lower()

covered = 0
not_covered = []

for idx, row in qa_df.iterrows():
    answer = row['answer'].lower()
    user = row['user']
    
    # Extract key terms from answer
    numbers = re.findall(r'\b\d{4,}\b', answer)
    
    # Check if user's emails contain the answer or key terms
    user_emails = emails_df[emails_df['user'] == user]['searchable']
    
    found = False
    if numbers:
        for num in numbers:
            if user_emails.str.contains(num, na=False).any():
                found = True
                break
    
    # Also check for substantial phrase matches (5+ words)
    words = answer.split()
    if len(words) >= 5:
        for i in range(len(words) - 4):
            phrase = ' '.join(words[i:i+5])
            if user_emails.str.contains(re.escape(phrase), na=False).any():
                found = True
                break
    
    if found:
        covered += 1
    else:
        not_covered.append(idx)

print(f"Coverage: {covered}/{len(qa_df)} ({100*covered/len(qa_df):.1f}%)")
print(f"Not covered: {len(not_covered)} Q&A pairs")
print()
print("Sample of uncovered questions:")
for idx in not_covered[:10]:
    print(f"  #{idx}: {qa_df.iloc[idx]['question'][:80]}...")
