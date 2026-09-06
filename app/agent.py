"""Top-level onboarding agent: RAG-grounded Q&A with citations, a confidence
gate, and HR escalation fallback.
"""

from dataclasses import dataclass

from langchain_core.prompts import ChatPromptTemplate

from app import citations, escalation
from app.llm import invoke_with_fallback
from app.rag_pipeline import FAISS, RetrievalResult, retrieve

ANSWER_SYSTEM_PROMPT = """You are an intelligent new-hire onboarding assistant for Northlight Systems.
Answer the employee's question using ONLY the provided context documents. Be concise and specific.
Personalize your answer using the employee's role, department, location, and start date when relevant.
Always ground claims in the context — do not add policy details, numbers, or dates that are not in it.
If the context is insufficient to answer confidently, say so plainly instead of guessing.
"""

ANSWER_USER_PROMPT = """Employee context:
- Name: {full_name}
- Role: {role}
- Department: {department}
- Location: {location} ({work_location_type})
- Start date: {start_date}

Context documents:
{context}

Employee question: {question}"""


@dataclass
class AgentResponse:
    answer: str
    confidence: float
    escalated: bool
    citations: list[dict]
    ticket_id: str | None = None


def answer_question(vectorstore: FAISS, workday_context: dict, question: str) -> AgentResponse:
    retrieval: RetrievalResult = retrieve(vectorstore, question, role=workday_context.get("role"))

    if not retrieval.is_confident:
        ticket = escalation.raise_escalation(
            employee_id=workday_context["employee_id"],
            employee_name=workday_context["full_name"],
            question=question,
            confidence=retrieval.confidence,
        )
        return AgentResponse(
            answer=escalation.escalation_message(ticket),
            confidence=retrieval.confidence,
            escalated=True,
            citations=citations.format_citations(retrieval.chunks),
            ticket_id=ticket.ticket_id,
        )

    context_text = citations.format_context_for_prompt(retrieval.chunks)
    prompt = ChatPromptTemplate.from_messages(
        [("system", ANSWER_SYSTEM_PROMPT), ("user", ANSWER_USER_PROMPT)]
    )
    prompt_value = prompt.invoke(
        {
            "full_name": workday_context["full_name"],
            "role": workday_context["role"],
            "department": workday_context["department"],
            "location": workday_context["location"],
            "work_location_type": workday_context["work_location_type"],
            "start_date": workday_context["start_date"],
            "context": context_text,
            "question": question,
        }
    )
    result, _model_used = invoke_with_fallback(prompt_value, temperature=0.1)

    return AgentResponse(
        answer=result.content,
        confidence=retrieval.confidence,
        escalated=False,
        citations=citations.format_citations(retrieval.chunks),
    )
