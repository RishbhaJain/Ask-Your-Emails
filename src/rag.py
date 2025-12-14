import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from anthropic import Anthropic
import config
from typing import List, Dict


class EmailRAG:
    def __init__(self, api_key=None):
        self.api_key = api_key or config.ANTHROPIC_API_KEY
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not found")

        self.client = Anthropic(api_key=self.api_key)
        self.model = config.LLM_MODEL
        self.max_context_emails = config.MAX_CONTEXT_EMAILS

    def answer_question(self, question, retrieved_emails, max_emails=None):
        max_emails = max_emails or self.max_context_emails
        context_emails = retrieved_emails[:max_emails]

        context_text = self._format_emails_for_context(context_emails)
        prompt = self._build_prompt(question, context_text)

        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )

        answer = response.content[0].text

        return {
            'answer': answer,
            'source_emails': context_emails,
            'num_sources': len(context_emails)
        }

    def _format_emails_for_context(self, emails):
        formatted = []
        for i, email in enumerate(emails, 1):
            subject = email.get('subject', 'No subject')
            sender = email.get('from', 'Unknown')
            body = email.get('body', '')

            # give more context to top results
            if i <= 3:
                body_preview = body[:1500]
            else:
                body_preview = body[:800]

            formatted.append(
                f"[Email {i}]\n"
                f"From: {sender}\n"
                f"Subject: {subject}\n"
                f"Body: {body_preview}\n"
            )

        return "\n---\n".join(formatted)

    def _build_prompt(self, question: str, context: str) -> str:
        return f"""You are analyzing emails from the Enron dataset to answer questions.

Context emails:
{context}

Question: {question}

Provide a clear, concise answer based only on the emails above. If the emails don't contain enough information, say so. Include specific details from the emails when relevant."""
