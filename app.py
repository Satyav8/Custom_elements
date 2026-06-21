import streamlit as st
from assistant.agent import DealerAgent

st.set_page_config(page_title="VIKMO Dealer Assistant", page_icon="🔧", layout="wide")

st.title("🔧 VIKMO Dealer Assistant")
st.caption("Find parts, check stock, and place orders — powered by AI")

# Initialise session state
if "agent" not in st.session_state:
    st.session_state.agent = DealerAgent()
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar
with st.sidebar:
    st.header("Controls")
    if st.button("🔄 New Conversation"):
        st.session_state.agent.reset()
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.markdown("**Sample queries:**")
    samples = [
        "Do you have brake pads for a Bajaj Pulsar 150?",
        "What's the cheapest chain lube you stock?",
        "I need tyres.",
        "Check stock for BRK-1042",
        "Place an order for 5 units of BRK-1042 for ABC Motors",
        "What's the weather today?",
    ]
    for s in samples:
        if st.button(s, key=s):
            st.session_state._prefill = s
            st.rerun()

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle prefill from sidebar button
prefill = st.session_state.pop("_prefill", None)

user_input = st.chat_input("Ask about parts, stock, or place an order...") or prefill

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.agent.chat(user_input)
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
