import sys
import os
import streamlit as st

# Add app/ to path so we can import directly
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'app'))

# ── Secrets: works locally (.env) AND on Streamlit Cloud (st.secrets) ──
def get_secret(key: str, default: str = "") -> str:
    try:
        return st.secrets[key]
    except Exception:
        return os.getenv(key, default)

# Inject all Azure secrets into environment before importing Azure modules
os.environ["AZURE_OPENAI_ENDPOINT"]             = get_secret("AZURE_OPENAI_ENDPOINT")
os.environ["AZURE_OPENAI_API_KEY"]              = get_secret("AZURE_OPENAI_API_KEY")
os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"]      = get_secret("AZURE_OPENAI_CHAT_DEPLOYMENT")
os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"] = get_secret("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
os.environ["AZURE_OPENAI_API_VERSION"]          = get_secret("AZURE_OPENAI_API_VERSION")
os.environ["AZURE_SEARCH_ENDPOINT"]             = get_secret("AZURE_SEARCH_ENDPOINT")
os.environ["AZURE_SEARCH_KEY"]                  = get_secret("AZURE_SEARCH_KEY")
os.environ["AZURE_SEARCH_INDEX"]                = get_secret("AZURE_SEARCH_INDEX")
os.environ["AZURE_CONTENT_SAFETY_ENDPOINT"]     = get_secret("AZURE_CONTENT_SAFETY_ENDPOINT")
os.environ["AZURE_CONTENT_SAFETY_KEY"]          = get_secret("AZURE_CONTENT_SAFETY_KEY")

# Now import the app modules (they read from os.environ)
from rag import rag_query, get_specialist_recommendation
from guardrails import check_content
from function_calling import route_query

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="Medical Q&A Assistant",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 Medical Q&A Assistant")
st.caption(
    "Powered by **Azure OpenAI** + **Azure AI Search** · "
    "Knowledge base: NIH MedQuAD (47,457 medical Q&A pairs)"
)

st.warning(
    "⚠️ **For informational purposes only.** "
    "This assistant is not a substitute for professional medical advice. "
    "Always consult a qualified healthcare provider."
)

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🏗️ Architecture")
    st.markdown("""
    - **LLM:** Azure OpenAI GPT-4o-mini
    - **Retrieval:** Azure AI Search
      (vector + BM25 + semantic reranking)
    - **Safety:** Azure Content Safety
    - **Routing:** Function calling
    - **Dataset:** MedQuAD — NIH
    """)

    st.markdown("---")
    st.markdown("### 📊 Eval Scores (RAGAS)")
    col1, col2 = st.columns(2)
    col1.metric("Faithfulness", "0.88")
    col2.metric("Relevancy", "0.90")
    col1.metric("Precision", "0.84")
    col2.metric("Recall", "0.81")
    st.caption("n=20 · MedQuAD (NIH)")

    st.markdown("---")
    top_k = st.slider("Sources to retrieve (k)", 1, 10, 5)
    use_fc = st.checkbox("Function calling routing", value=True)

# ── Chat state ────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


def handle_question(question: str):
    """Core function — runs the full RAG pipeline and displays result."""
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching NIH knowledge base..."):

            # 1. Input safety
            safety = check_content(question)
            if not safety["safe"]:
                answer = f"🚫 Query blocked by content safety: {safety['reason']}"
                st.error(answer)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )
                return

            routed_action = "search_knowledge_base"
            topic_type = None

            # 2. Function calling router
            if use_fc:
                try:
                    routing = route_query(question)
                    routed_action = routing.get("action", "search_knowledge_base")
                    topic_type = routing.get("topic_type")

                    if routed_action == "emergency_redirect":
                        answer = (
                            "🚨 **This sounds like a medical emergency.**\n\n"
                            "**Please call 000 (Australia) immediately.**\n\n"
                            "Go to your nearest emergency department now."
                        )
                        st.error(answer)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": answer}
                        )
                        return

                    if routed_action == "recommend_specialist":
                        answer = get_specialist_recommendation(
                            routing.get("condition_area", "general"),
                            routing.get("urgency", "routine")
                        )
                        st.write(answer)
                        st.caption("🔀 Routed → **👨‍⚕️ Specialist Recommendation**")
                        st.session_state.messages.append(
                            {"role": "assistant", "content": answer}
                        )
                        return
                except Exception as e:
                    st.warning(f"Routing unavailable, using direct search: {e}")

            # 3. RAG query
            try:
                result = rag_query(question, top_k=top_k, topic_type=topic_type)

                # 4. Output safety
                out_safety = check_content(result["answer"])
                if not out_safety["safe"]:
                    result["answer"] = (
                        "Response filtered by content safety. "
                        "Please consult a healthcare professional."
                    )

                st.write(result["answer"])

                # Metrics
                c1, c2, c3 = st.columns(3)
                c1.metric("⏱ Latency", f"{result.get('latency_ms', 0)}ms")
                c2.metric("🔍 Retrieval", f"{result.get('retrieval_ms', 0)}ms")
                c3.metric("🪙 Tokens", result.get("tokens_used", 0))

                action_labels = {
                    "search_knowledge_base": "🔍 KB Search",
                    "recommend_specialist": "👨‍⚕️ Specialist",
                    "emergency_redirect": "🚨 Emergency",
                }
                st.caption(
                    f"🔀 Routed → **{action_labels.get(routed_action, routed_action)}**"
                )

                if result.get("sources"):
                    with st.expander("📚 Related questions from knowledge base"):
                        for src in result["sources"][:3]:
                            st.caption(f"• {src}")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "meta": result
                })

            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_msg}
                )


# ── Display chat history ──────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ── Suggestion buttons ────────────────────────────────────
if not st.session_state.messages:
    st.markdown("### 💡 Example questions")
    suggestions = [
        "What are the symptoms of Type 2 diabetes?",
        "What is the treatment for high blood pressure?",
        "What causes asthma and how is it managed?",
        "What are the side effects of metformin?",
        "What are the early signs of Alzheimer's disease?",
        "How is depression diagnosed and treated?"
    ]
    cols = st.columns(3)
    for i, s in enumerate(suggestions):
        if cols[i % 3].button(s, key=f"sug_{i}"):
            handle_question(s)

# ── Chat input ────────────────────────────────────────────
if question := st.chat_input("Ask a medical question..."):
    handle_question(question)