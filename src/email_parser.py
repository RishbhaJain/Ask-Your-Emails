import email
from email import policy
from email.parser import Parser
import email.utils
import re
from datetime import datetime
from typing import Dict, Optional
from bs4 import BeautifulSoup
import pandas as pd
import ast


def parse_enron_format(email_text):
    """Parse Enron dataset custom format (Sender/Recipients fields)"""
    # check if this is actually Enron format
    if 'Sender:' not in email_text and 'Recipients:' not in email_text:
        return None

    result = {
        'subject': '',
        'from': '',
        'to': '',
        'cc': '',
        'date': None,
        'body': ''
    }

    subject_match = re.search(r'Subject:\s*(.+?)(?:\n|$)', email_text, re.IGNORECASE)
    if subject_match:
        result['subject'] = subject_match.group(1).strip()

    sender_match = re.search(r'Sender:\s*(.+?)(?:\n|$)', email_text, re.IGNORECASE)
    if sender_match:
        result['from'] = sender_match.group(1).strip()

    # recipients are in format: ['email1', 'email2',  ]
    recipients_match = re.search(r'Recipients:\s*(\[.+?\])', email_text, re.IGNORECASE | re.DOTALL)
    if recipients_match:
        try:
            recipients_str = recipients_match.group(1)
            recipients_list = eval(recipients_str)
            if isinstance(recipients_list, list):
                result['to'] = ', '.join(recipients_list)
        except:
            result['to'] = recipients_match.group(1).strip()

    # try multiple date patterns
    sent_match = re.search(r'Sent:\s*(.+?)(?:\n|$)', email_text, re.IGNORECASE)
    if sent_match:
        result['date'] = parse_date(sent_match.group(1).strip())
    else:
        date_match = re.search(r'^Date:\s*(.+?)(?:\n|$)', email_text, re.IGNORECASE | re.MULTILINE)
        if date_match:
            result['date'] = parse_date(date_match.group(1).strip())
        else:
            forwarded_date = re.search(r'-+\s*Forwarded by.*?on\s+(\d{1,2}/\d{1,2}/\d{4}\s+\d{1,2}:\d{2}\s+[AP]M)', email_text, re.IGNORECASE)
            if forwarded_date:
                result['date'] = parse_date(forwarded_date.group(1).strip())
            else:
                timestamp_match = re.search(r'\n\s*(\d{1,2}/\d{1,2}/\d{4}\s+\d{1,2}:\d{2}\s+[AP]M)', email_text, re.IGNORECASE)
                if timestamp_match:
                    result['date'] = parse_date(timestamp_match.group(1).strip())

    # extract body after metadata
    body_match = re.search(r'File:.*?\n=+\n(.+)', email_text, re.DOTALL | re.IGNORECASE)
    if body_match:
        result['body'] = clean_body(body_match.group(1))
    else:
        body_match = re.search(r'Recipients:.*?\n=+\n(.+)', email_text, re.DOTALL | re.IGNORECASE)
        if body_match:
            result['body'] = clean_body(body_match.group(1))

    return result


def parse_email_text(email_text):
    """Parse email text and extract metadata"""
    # try Enron format first
    enron_result = parse_enron_format(email_text)
    if enron_result is not None:
        return enron_result

    try:
        # use Python's email library for standard format
        msg = email.message_from_string(email_text, policy=policy.default)

        subject = clean_header(msg.get('subject', ''))
        from_addr = clean_header(msg.get('from', ''))
        to_addr = clean_header(msg.get('to', ''))
        cc_addr = clean_header(msg.get('cc', ''))
        date_str = clean_header(msg.get('date', ''))

        parsed_date = parse_date(date_str)
        body = extract_body(msg)

        return {
            'subject': subject,
            'from': from_addr,
            'to': to_addr,
            'cc': cc_addr,
            'date': parsed_date,
            'body': body
        }

    except Exception as e:
        return parse_email_with_regex(email_text)


def clean_header(header_value: str) -> str:
    """Clean and decode email header values."""
    if not header_value:
        return ""

    # Remove extra whitespace
    cleaned = ' '.join(header_value.split())
    return cleaned.strip()


def parse_date(date_str):
    """Parse various date formats to ISO format"""
    if not date_str:
        return None

    date_str = ' '.join(date_str.split())

    try:
        # try Enron "Sent:" format
        from datetime import datetime
        sent_patterns = [
            '%A, %B %d, %Y %I:%M %p',
            '%A, %B %d, %Y %I:%M:%S %p',
        ]
        for pattern in sent_patterns:
            try:
                dt = datetime.strptime(date_str, pattern)
                return dt.strftime('%Y-%m-%d')
            except:
                continue
    except:
        pass

    try:
        # try MM/DD/YYYY format
        enron_match = re.match(r'(\d{1,2})/(\d{1,2})/(\d{4})', date_str)
        if enron_match:
            month, day, year = enron_match.groups()
            from datetime import datetime
            dt = datetime(int(year), int(month), int(day))
            return dt.strftime('%Y-%m-%d')
    except:
        pass

    try:
        # standard RFC 822
        dt = email.utils.parsedate_to_datetime(date_str)
        return dt.strftime('%Y-%m-%d')
    except:
        # just extract year if we can
        year_match = re.search(r'\b(19|20)\d{2}\b', date_str)
        if year_match:
            return f"{year_match.group(0)}-01-01"
        return None


def extract_body(msg) -> str:
    """Extract clean body text from email message."""
    body = ""

    # Handle multipart messages
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()

            if content_type == "text/plain":
                try:
                    body = part.get_payload(decode=True).decode('utf-8', errors='ignore')
                    break
                except:
                    continue
            elif content_type == "text/html" and not body:
                try:
                    html_content = part.get_payload(decode=True).decode('utf-8', errors='ignore')
                    body = strip_html(html_content)
                except:
                    continue
    else:
        # Single part message
        try:
            body = msg.get_payload(decode=True).decode('utf-8', errors='ignore')
        except:
            body = str(msg.get_payload())

    # Clean the body
    body = clean_body(body)

    return body


def strip_html(html_text: str) -> str:
    """Strip HTML tags from text using BeautifulSoup."""
    try:
        soup = BeautifulSoup(html_text, 'html.parser')
        return soup.get_text()
    except:
        # Fallback: simple regex
        return re.sub(r'<[^>]+>', '', html_text)


def clean_body(body):
    """Clean email body - remove forwarding chains and signatures"""
    if not body:
        return ""

    # remove forwarded content
    forwarding_patterns = [
        r'-+\s*Original Message\s*-+',
        r'-+\s*Forwarded by.*?-+',
        r'From:.*?Sent:.*?To:',
    ]

    for pattern in forwarding_patterns:
        match = re.search(pattern, body, re.IGNORECASE | re.DOTALL)
        if match:
            body = body[:match.start()]
            break

    # remove signatures
    signature_patterns = [
        r'\n--\s*\n',
        r'\nRegards,.*',
        r'\nBest,.*',
        r'\nThanks,.*',
    ]

    for pattern in signature_patterns:
        body = re.split(pattern, body, maxsplit=1, flags=re.IGNORECASE)[0]

    body = re.sub(r'\n\s*\n', '\n\n', body)
    body = re.sub(r'[ \t]+', ' ', body)
    body = body.strip()

    return body


def parse_email_with_regex(email_text: str) -> Dict[str, str]:
    """Fallback parser using regex when email.parser fails.
    Handles both standard email headers and Enron format."""
    result = {
        'subject': '',
        'from': '',
        'to': '',
        'cc': '',
        'date': None,
        'body': ''
    }

    # Extract subject
    subject_match = re.search(r'Subject:\s*(.+?)(?:\n[A-Z]|\n\n)', email_text, re.IGNORECASE)
    if subject_match:
        result['subject'] = subject_match.group(1).strip()

    # Extract from - try both 'From:' and 'Sender:'
    from_match = re.search(r'From:\s*(.+?)(?:\n[A-Z]|\n\n)', email_text, re.IGNORECASE)
    if from_match:
        result['from'] = from_match.group(1).strip()
    else:
        # Try Enron format 'Sender:'
        sender_match = re.search(r'Sender:\s*(.+?)(?:\n|$)', email_text, re.IGNORECASE)
        if sender_match:
            result['from'] = sender_match.group(1).strip()

    # Extract to - try both 'To:' and 'Recipients:'
    to_match = re.search(r'To:\s*(.+?)(?:\n[A-Z]|\n\n)', email_text, re.IGNORECASE)
    if to_match:
        result['to'] = to_match.group(1).strip()
    else:
        # Try Enron format 'Recipients:'
        recipients_match = re.search(r'Recipients:\s*(\[.+?\])', email_text, re.IGNORECASE | re.DOTALL)
        if recipients_match:
            try:
                recipients_str = recipients_match.group(1)
                recipients_list = eval(recipients_str)
                if isinstance(recipients_list, list):
                    result['to'] = ', '.join(recipients_list)
            except:
                result['to'] = recipients_match.group(1).strip()

    # Extract date
    date_match = re.search(r'Date:\s*(.+?)(?:\n[A-Z]|\n\n)', email_text, re.IGNORECASE)
    if date_match:
        result['date'] = parse_date(date_match.group(1).strip())

    # Extract body (everything after headers)
    body_match = re.search(r'\n\n(.+)', email_text, re.DOTALL)
    if body_match:
        result['body'] = clean_body(body_match.group(1))

    return result


def parse_emails_batch(df):
    """Parse email text for all records"""
    from tqdm import tqdm

    print("Parsing email headers and bodies ")

    parsed_data = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Parsing emails"):
        email_text = row['email']
        parsed = parse_email_text(email_text)

        parsed_row = {
            'path': row['path'],
            'user': row['user'],
            'subject': parsed['subject'],
            'from': parsed['from'],
            'to': parsed['to'],
            'cc': parsed['cc'],
            'date': parsed['date'],
            'body': parsed['body'],
            'full_text': email_text
        }

        parsed_data.append(parsed_row)

    parsed_df = pd.DataFrame(parsed_data)

    print(f"Parsed {len(parsed_df)} emails")

    print(f"\nParsing statistics:")
    print(f"  Emails with subject: {parsed_df['subject'].notna().sum()}")
    print(f"  Emails with from: {parsed_df['from'].notna().sum()}")
    print(f"  Emails with date: {parsed_df['date'].notna().sum()}")
    print(f"  Average body length: {parsed_df['body'].str.len().mean():.0f} chars")

    return parsed_df


def extract_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract year and month from parsed dates for faceting."""
    print("\nExtracting date features for faceting ")

    # Parse dates
    df['date_parsed'] = pd.to_datetime(df['date'], errors='coerce')

    # Extract year and month
    df['date_year'] = df['date_parsed'].dt.year
    df['date_month'] = df['date_parsed'].dt.month

    # Fill missing years with median (for emails with unparseable dates)
    median_year = df['date_year'].median()
    df['date_year'] = df['date_year'].fillna(median_year).astype('Int64')

    print(f"  Date range: {df['date_year'].min()} - {df['date_year'].max()}")

    return df
