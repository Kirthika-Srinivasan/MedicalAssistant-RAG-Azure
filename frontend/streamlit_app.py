import streamlit as st
import requests
import os

def get_secret(key, default=""):
    try:
        return st.secrets[key]
    except Exception:
        return os.getenv(key, default) 
    
API_URL = get_secret("API_URL", "http://localhost:8000")

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

# Important disclaimer banner
st.warning(
    "⚠️ **For informational purposes only.** "
    "This assistant is not a substitute for professional medical advice. "
    "Always consult a qualified healthcare provider."
)

with st.sidebar:
    st.markdown("### 🏗️ Architecture")
    st.markdown("""
    - **LLM:** Azure OpenAI GPT-5.4-mini 
    - **Retrieval:** Azure AI Search  
      (vector + BM25 + semantic reranking)  
    - **Safety:** Azure Content Safety  
    - **Routing:** Function calling  
      (search / specialist / emergency)  
    - **Dataset:** MedQuAD — NIH  
    """)

    st.markdown("---")
    st.markdown("### 📊 Eval Scores (RAGAS)")
    try:
        r = requests.get(f"{API_URL}/eval-results", timeout=3)
        if r.status_code == 200:
            scores = r.json()
            if "faithfulness" in scores:
                col1, col2 = st.columns(2)
                col1.metric("Faithfulness", scores["faithfulness"])
                col2.metric("Relevancy", scores["answer_relevancy"])
                col1.metric("Precision", scores["context_precision"])
                col2.metric("Recall", scores["context_recall"])
                st.caption(f"n={scores.get('n_samples')} · {scores.get('dataset')}")
    except Exception:
        st.caption("Start API to see eval scores")

    st.markdown("---")
    top_k = st.slider("Sources to retrieve (k)", 1, 10, 5)
    use_fc = st.checkbox("Function calling routing", value=True)

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Suggested questions
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
        if cols[i % 3].button(s, key=f"s{i}"):
            st.session_state.messages.append(
                {"role": "user", "content": s}
            )
            st.rerun()

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("meta"):
            m = msg["meta"]
            c1, c2, c3 = st.columns(3)
            c1.metric("⏱ Latency", f"{m.get('latency_ms', 0)}ms")
            c2.metric("🔍 Retrieval", f"{m.get('retrieval_ms', 0)}ms")
            c3.metric("🪙 Tokens", m.get("tokens_used", 0))

            action = m.get("routed_action", "")
            action_labels = {
                "search_knowledge_base": "🔍 KB Search",
                "recommend_specialist": "👨‍⚕️ Specialist Rec.",
                "emergency_redirect": "🚨 Emergency",
            }
            st.caption(f"Routed → **{action_labels.get(action, action)}**")

            if m.get("sources"):
                with st.expander("📚 Related questions from knowledge base"):
                    for src in m["sources"][:3]:
                        st.caption(f"• {src}")

# Chat input
if question := st.chat_input("Ask a medical question..."):
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching NIH knowledge base..."):
            try:
                resp = requests.post(
                    f"{API_URL}/query",
                    json={
                        "question": question,
                        "top_k": top_k,
                        "use_function_calling": use_fc
                    },
                    timeout=30
                )

                if resp.status_code == 400:
                    st.error(
                        f"🚫 {resp.json().get('detail', 'Blocked by safety policy')}"
                    )
                elif resp.status_code == 200:
                    data = resp.json()
                    st.write(data["answer"])

                    c1, c2, c3 = st.columns(3)
                    c1.metric("⏱ Latency", f"{data['latency_ms']}ms")
                    c2.metric("🔍 Retrieval", f"{data['retrieval_ms']}ms")
                    c3.metric("🪙 Tokens", data["tokens_used"])

                    action_labels = {
                        "search_knowledge_base": "🔍 KB Search",
                        "recommend_specialist": "👨‍⚕️ Specialist Rec.",
                        "emergency_redirect": "🚨 Emergency",
                    }
                    st.caption(
                        f"Routed → **{action_labels.get(data['routed_action'], data['routed_action'])}**"
                    )

                    if data.get("sources"):
                        with st.expander("📚 Related questions from knowledge base"):
                            for src in data["sources"][:3]:
                                st.caption(f"• {src}")

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": data["answer"],
                        "meta": data
                    })

            except requests.exceptions.ConnectionError:
                st.error(
                    "Cannot connect to API. "
                    "Run: `uvicorn app.main:app --reload`"
                )