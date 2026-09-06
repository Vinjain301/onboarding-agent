"""Generates structured 30/60/90-day plans, checklists, and learning paths
grounded in the retrieved role-specific and policy documents.
"""

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, ValidationError

from app import citations
from app.llm import build_chat_model, invoke_with_fallback
from app.rag_pipeline import FAISS, RetrievalResult, retrieve


class ChecklistItem(BaseModel):
    task: str = Field(description="A single concrete onboarding action item")
    category: str = Field(description="e.g. IT Setup, Compliance, Benefits, Training, Team Integration")
    due_by: str = Field(description="e.g. 'Day 1', 'Week 1', 'Day 30', 'Day 60', 'Day 90'")


class PlanPhase(BaseModel):
    phase_name: str = Field(description="e.g. 'First 30 Days'")
    goals: list[str] = Field(description="2-4 high-level goals for this phase")
    milestones: list[str] = Field(description="3-6 concrete, measurable milestones for this phase")


class TrainingItem(BaseModel):
    order: int
    title: str
    description: str = Field(description="One sentence on what this training covers and why it matters now")


class OnboardingPlan(BaseModel):
    role: str
    department: str
    phases: list[PlanPhase] = Field(description="Exactly 3 phases: first 30, 60, and 90 days")
    checklist: list[ChecklistItem] = Field(
        description="6-12 onboarding checklist items across the full 90 days"
    )
    training_path: list[TrainingItem] = Field(description="Ordered recommended learning path, 3-6 items")


PLAN_SYSTEM_PROMPT = """You are an HR onboarding assistant generating a personalized onboarding plan.
Base every phase, milestone, checklist item, and training recommendation ONLY on the provided
context documents. Do not invent policies, tools, or timelines that are not supported by the context.
If the context does not mention something relevant, omit it rather than guessing.
Tailor the plan to the employee's specific role, department, location, and work arrangement.
"""

PLAN_USER_PROMPT = """Employee context:
- Name: {full_name}
- Role: {role}
- Department: {department}
- Location: {location} ({work_location_type})
- Start date: {start_date}
- Day 30 / 60 / 90 target dates: {day_30} / {day_60} / {day_90}

Context documents (role guide + relevant policies):
{context}

Generate a complete 30/60/90-day onboarding plan, checklist, and recommended learning path for this employee.

Respond with ONLY valid JSON matching this schema — no markdown fences, no commentary before or after:
{format_instructions}"""

REPAIR_PROMPT = """The following text was supposed to be valid JSON matching this schema:
{format_instructions}

It failed to parse with this error:
{error}

Here is the text to fix:
{broken_output}

Return ONLY the corrected, valid JSON — no markdown fences, no commentary."""


def _repair_json(parser: PydanticOutputParser, broken_text: str, error: Exception, model: str):
    """One follow-up LLM call asking the model to fix its own malformed JSON."""
    prompt_value = ChatPromptTemplate.from_messages([("user", REPAIR_PROMPT)]).invoke(
        {
            "format_instructions": parser.get_format_instructions(),
            "error": str(error),
            "broken_output": broken_text,
        }
    )
    repaired = build_chat_model(model, temperature=0).invoke(prompt_value)
    return parser.parse(repaired.content)


def generate_onboarding_plan(
    vectorstore: FAISS, workday_context: dict
) -> tuple[OnboardingPlan, RetrievalResult]:
    role = workday_context["role"]
    query = (
        f"30 60 90 day onboarding plan, checklist, required training, and learning path "
        f"for a {role} in {workday_context['department']}"
    )
    retrieval = retrieve(vectorstore, query, role=role, k=6)
    context_text = citations.format_context_for_prompt(retrieval.chunks)

    # Free OpenRouter models have inconsistent tool-calling support, so plans are
    # requested as plain JSON text and parsed, rather than relying on function
    # calling. If the first response does not parse cleanly, one repair call
    # asks the same model to fix its own JSON.
    parser = PydanticOutputParser(pydantic_object=OnboardingPlan)

    prompt = ChatPromptTemplate.from_messages([("system", PLAN_SYSTEM_PROMPT), ("user", PLAN_USER_PROMPT)])
    prompt_value = prompt.invoke(
        {
            "full_name": workday_context["full_name"],
            "role": role,
            "department": workday_context["department"],
            "location": workday_context["location"],
            "work_location_type": workday_context["work_location_type"],
            "start_date": workday_context["start_date"],
            "day_30": workday_context["milestones"]["day_30"],
            "day_60": workday_context["milestones"]["day_60"],
            "day_90": workday_context["milestones"]["day_90"],
            "context": context_text,
            "format_instructions": parser.get_format_instructions(),
        }
    )
    raw_response, model_used = invoke_with_fallback(prompt_value, temperature=0.2)

    try:
        plan = parser.parse(raw_response.content)
    except (ValidationError, ValueError) as e:
        plan = _repair_json(parser, raw_response.content, e, model_used)
    return plan, retrieval
