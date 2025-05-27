import os
import textwrap
import streamlit as st
from openai import OpenAI
from pinecone import Pinecone

# --------------------
# Configuration
# --------------------
# Prefer Streamlit secrets, fallback to environment variables
PINECONE_API_KEY = st.secrets.get("PINECONE_API_KEY", os.getenv("PINECONE_API_KEY"))
OPENAI_API_KEY   = st.secrets.get("OPENAI_API_KEY",   os.getenv("OPENAI_API_KEY"))
INDEX_NAME       = st.secrets.get("INDEX_NAME",       os.getenv("INDEX_NAME", "aquestia"))
NAMESPACE        = st.secrets.get("NAMESPACE",        os.getenv("NAMESPACE", "v1"))
EMBED_MODEL      = st.secrets.get("EMBED_MODEL",      os.getenv("EMBED_MODEL", "text-embedding-3-large"))
TOP_K            = int(st.secrets.get("TOP_K",        os.getenv("TOP_K", 20)))

if not PINECONE_API_KEY or not OPENAI_API_KEY:
    st.error("⚠️  Missing API keys for Pinecone and/or OpenAI. Add them to Streamlit secrets or environment variables and restart the app.")
    st.stop()

# --------------------
# Clients
# --------------------
pc     = Pinecone(api_key=PINECONE_API_KEY)
index  = pc.Index(INDEX_NAME)
client = OpenAI(api_key=OPENAI_API_KEY)

# --------------------
# Helper Function
# --------------------

def ask(question: str) -> str:
    """Takes a user question, retrieves relevant context from Pinecone,
    and returns an answer from GPT‑4o constrained to that context."""
    # 1) embed the question
    q_vec = client.embeddings.create(
        model = EMBED_MODEL,
        input = question
    ).data[0].embedding

    # 2) vector search
    res = index.query(
        vector      = q_vec,
        top_k       = TOP_K,
        namespace   = NAMESPACE,
        include_metadata = True
    )

    # 3) build the context for GPT‑4o
    context_chunks = [m.metadata.get("text", "") for m in res.matches]
    context_text   = "\n\n---\n\n".join(context_chunks)

    prompt = textwrap.dedent(f"""
        You are Pipe‑Spec Assistant.
        Answer the user's question **only** with information in the context.
        Try to give the best answer possible. If you really don't know say you don't know, but still try to provide some value.

        ### Context
        {context_text}

        ### Question
        {question}

        ### Answer
    """)

    # 4) call GPT‑4o
    chat_resp = client.chat.completions.create(
        model  = "gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return chat_resp.choices[0].message.content.strip()

# --------------------
# Streamlit UI
# --------------------

def main():
    st.set_page_config(page_title="Pipe‑Spec Assistant", page_icon="🛠️", layout="centered")
    st.title("🛠️ Pipe‑Spec Assistant")

    st.markdown(
        "Ask any question about AQUESTIA pipe specifications or related docs.\n"
        "The assistant will answer **only** using information retrieved from the knowledge base."
    )

    question = st.text_area("Your question:", height=150, placeholder="e.g. What is the maximum operating pressure for the S100‑AL valve?")

    if st.button("Ask") and question.strip():
        with st.spinner("Thinking..."):
            try:
                answer = ask(question)
                st.markdown("### Answer")
                st.write(answer)
            except Exception as e:
                st.error(f"❌ An error occurred: {e}")

    # Optional: expandable debug panel
    with st.expander("ℹ️  Show advanced settings / debug"):
        st.text_input("Pinecone Index", INDEX_NAME, disabled=True)
        st.text_input("Namespace", NAMESPACE, disabled=True)
        st.number_input("Top‑K", value=TOP_K, disabled=True)
        st.text_input("Embedding Model", EMBED_MODEL, disabled=True)


if __name__ == "__main__":
    main()
