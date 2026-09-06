import streamlit as st

from app import config, escalation
from app.agent import answer_question
from app.plan_generator import generate_onboarding_plan
from app.rag_pipeline import get_or_build_vectorstore
from mock_workday import MockWorkdayClient

st.set_page_config(page_title="New Hire Onboarding Assistant", layout="wide")

workday = MockWorkdayClient()


@st.cache_resource(show_spinner="Loading knowledge base (embedding HR documents)...")
def load_vectorstore():
    return get_or_build_vectorstore()


def require_api_key():
    if not config.OPENROUTER_API_KEY:
        st.error(
            "No OPENROUTER_API_KEY found. Copy `.env.example` to `.env` and add your "
            "OpenRouter key (openrouter.ai/keys), then restart the app."
        )
        st.stop()


def render_sidebar() -> dict:
    st.sidebar.title("New Hire Onboarding")
    employees = workday.list_employees()
    labels = [f"{e['full_name']} — {e['role']} ({e['employee_id']})" for e in employees]
    choice = st.sidebar.selectbox("Select employee (mock Workday record)", labels)
    employee_id = employees[labels.index(choice)]["employee_id"]

    context = workday.get_onboarding_context(employee_id)

    st.sidebar.markdown("### Profile (from mock Workday)")
    st.sidebar.write(f"**Role:** {context['role']}")
    st.sidebar.write(f"**Department:** {context['department']}")
    st.sidebar.write(f"**Location:** {context['location']} ({context['work_location_type']})")
    st.sidebar.write(f"**Start date:** {context['start_date']}")
    st.sidebar.write(f"**Manager:** {context['manager']}")
    st.sidebar.write(f"**Onboarding phase:** {context['onboarding_phase'].replace('_', ' ')}")

    with st.sidebar.expander("HR escalation queue (demo)"):
        tickets = escalation.list_escalations()
        if not tickets:
            st.caption("No escalations yet.")
        else:
            for t in reversed(tickets[-10:]):
                st.markdown(f"**{t['ticket_id']}** — {t['employee_name']}")
                st.caption(f"{t['question']}  \nconfidence={t['confidence']}")

    return context


def render_chat_tab(vectorstore, context: dict):
    st.subheader("Ask the onboarding assistant")
    st.caption(
        "Answers are grounded in HR policies, benefits guides, and SOPs, with citations. "
        "Low-confidence questions are escalated to HR instead of guessed."
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("meta"):
                st.caption(msg["meta"])

    question = st.chat_input("e.g. How much PTO do I accrue, and when can I start using it?")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = answer_question(vectorstore, context, question)
            st.markdown(response.answer)

            if response.escalated:
                st.warning(f"Escalated to HR — ticket {response.ticket_id}")
            else:
                st.progress(min(response.confidence, 1.0), text=f"Confidence: {response.confidence:.2f}")
                if response.citations:
                    with st.expander("Sources"):
                        for c in response.citations:
                            st.markdown(f"- {c['title']} (`{c['source']}`) — relevance {c['confidence']:.2f}")

        meta = f"confidence={response.confidence:.2f}" + (" · escalated" if response.escalated else "")
        st.session_state.messages.append({"role": "assistant", "content": response.answer, "meta": meta})


def render_plan_tab(vectorstore, context: dict):
    st.subheader("Personalized 30/60/90-Day Onboarding Plan")
    st.caption(f"Generated for {context['full_name']} — {context['role']}, {context['department']}")

    if st.button("Generate my onboarding plan", type="primary"):
        with st.spinner("Generating plan from role guides and policies..."):
            plan, retrieval = generate_onboarding_plan(vectorstore, context)
        st.session_state.plan = plan
        st.session_state.plan_retrieval = retrieval

    plan = st.session_state.get("plan")
    retrieval = st.session_state.get("plan_retrieval")
    if not plan:
        st.info("Click the button above to generate a plan grounded in the onboarding knowledge base.")
        return

    if not retrieval.is_confident:
        st.warning("Retrieval confidence for this plan was low — treat it as a draft and confirm with HR.")

    phase_tabs = st.tabs([p.phase_name for p in plan.phases])
    for tab, phase in zip(phase_tabs, plan.phases, strict=True):
        with tab:
            st.markdown("**Goals**")
            for g in phase.goals:
                st.markdown(f"- {g}")
            st.markdown("**Milestones**")
            for m in phase.milestones:
                st.markdown(f"- {m}")

    st.markdown("---")
    st.markdown("### Onboarding Checklist")
    for item in plan.checklist:
        st.checkbox(f"[{item.due_by}] {item.task} ({item.category})", key=f"chk-{item.task}")

    st.markdown("### Recommended Learning Path")
    for item in sorted(plan.training_path, key=lambda t: t.order):
        st.markdown(f"**{item.order}. {item.title}** — {item.description}")

    with st.expander("Sources used for this plan"):
        for c in retrieval.chunks:
            st.markdown(f"- {c.title} (`{c.source}`) — relevance {c.score:.2f}")


def main():
    require_api_key()
    context = render_sidebar()
    vectorstore = load_vectorstore()

    tab1, tab2 = st.tabs(["Chat Assistant", "30/60/90 Plan and Checklist"])
    with tab1:
        render_chat_tab(vectorstore, context)
    with tab2:
        render_plan_tab(vectorstore, context)


if __name__ == "__main__":
    main()
