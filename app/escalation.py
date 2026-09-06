"""HR escalation workflow.

When the RAG pipeline cannot retrieve a confident answer, the system does not
allow the language model to guess and risk producing incorrect HR, benefits,
or compliance information. Instead, it logs a ticket and returns a clear
message directing the employee to a human contact.
"""

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

from app import config


@dataclass
class EscalationTicket:
    ticket_id: str
    employee_id: str
    employee_name: str
    question: str
    reason: str
    confidence: float
    created_at: str


def _next_ticket_id() -> str:
    if not config.ESCALATIONS_LOG.exists():
        return "ESC-1001"
    with open(config.ESCALATIONS_LOG, encoding="utf-8") as f:
        count = sum(1 for _ in f)
    return f"ESC-{1001 + count}"


def raise_escalation(
    employee_id: str,
    employee_name: str,
    question: str,
    confidence: float,
    reason: str = "low_retrieval_confidence",
) -> EscalationTicket:
    ticket = EscalationTicket(
        ticket_id=_next_ticket_id(),
        employee_id=employee_id,
        employee_name=employee_name,
        question=question,
        reason=reason,
        confidence=round(confidence, 4),
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    config.ESCALATIONS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(config.ESCALATIONS_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(ticket)) + "\n")
    return ticket


def list_escalations() -> list[dict]:
    if not config.ESCALATIONS_LOG.exists():
        return []
    with open(config.ESCALATIONS_LOG, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def escalation_message(ticket: EscalationTicket) -> str:
    return (
        "I don't have a confident, sourced answer to that in the onboarding knowledge base, "
        f"so I've escalated it to HR (ticket **{ticket.ticket_id}**). "
        "An HR team member will follow up directly. In the meantime, you can also reach "
        "HR at hr-general@northlightsystems.example."
    )
