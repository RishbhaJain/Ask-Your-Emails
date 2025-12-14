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

            # Give more context to top results; include both head and tail so endings aren't lost
            if i <= 3:
                body_preview = self._preview_text(body, 3200)
            else:
                body_preview = self._preview_text(body, 2200)

            formatted.append(
                f"[Email {i}]\n"
                f"From: {sender}\n"
                f"Subject: {subject}\n"
                f"Body: {body_preview}\n"
            )

        return "\n---\n".join(formatted)

    def _preview_text(self, body: str, max_chars: int) -> str:
        """Return a head+tail preview to capture endings of long emails."""
        if not body:
            return ""

        if len(body) <= max_chars:
            return body

        head_len = max_chars // 2
        tail_len = max_chars - head_len - len("\n...[trimmed]...\n")
        return body[:head_len] + "\n...[trimmed]...\n" + body[-tail_len:]

    def _build_prompt(self, question: str, context: str) -> str:
        return f'''You are extracting a precise answer from the provided emails.

    Read the emails carefully. If you find the answer, quote the relevant phrase and then give a concise final answer. Focus on names, dates, who did what, and any explicit statements.

    Context emails:
    {context}

    Question: {question}

    Instructions:
    - If the answer is stated, respond with: "Answer: <concise answer>" and optionally one short supporting quote.
    - If multiple names/actors are mentioned, pick the one explicitly tied to the action in the question.
    - If not found in the context, say: "Answer: Not found in provided emails."'''
