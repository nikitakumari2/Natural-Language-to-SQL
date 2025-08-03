import streamlit as st
import requests
import os
import re

# Configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000/query")

st.set_page_config(page_title="NL2SQL Query Assistant", layout="wide")
st.title("📊 NL2SQL Query Assistant")
st.caption("Ask a question about your data in natural language.")

# Input
user_question = st.text_area("Enter your question here:", height=100)

# Helper to extract and clean output
def format_output(raw_output):
    # Clean multi-line formatted response using stricter pattern
    pattern = (
        r"Input Question:\s*(.*?)\s*SQL Query:\s*(.*?)\s*SQL Output:\s*(.*?)\s*Answer:\s*(.*)"
    )
    match = re.search(pattern, raw_output, re.DOTALL)
    if not match:
        return {
            "Input Question": "",
            "SQL Query": "",
            "SQL Output": "",
            "Answer": raw_output.strip(),  # fallback
        }
    return {
        "Input Question": match.group(1).strip(),
        "SQL Query": match.group(2).strip(),
        "SQL Output": match.group(3).strip(),
        "Answer": match.group(4).strip(),
    }

# Button
if st.button("Get Answer"):
    if user_question:
        with st.status("🔄 Processing your question...", expanded=True) as status:
            try:
                response = requests.post(BACKEND_URL, json={"question": user_question})
                response.raise_for_status()
                result_data = response.json()
                answer = result_data.get("result", "❌ Error: No result field in response.")

                # Format and display clean blocks
                blocks = format_output(answer)

                status.update(label="✅ Processing Complete!", state="complete", expanded=True)
                st.markdown("---")

                with st.container():
                    if blocks["Input Question"]:
                        st.subheader("📝 Input Question")
                        st.markdown(f"**{blocks['Input Question']}**")

                    if blocks["SQL Query"]:
                        st.subheader("💡 SQL Query")
                        st.code(blocks["SQL Query"], language="sql")

                    if blocks["SQL Output"]:
                        st.subheader("📄 SQL Output")
                        st.code(blocks["SQL Output"], language="text")

                    if blocks["Answer"]:
                        st.subheader("✅ Final Answer")
                        st.success(blocks["Answer"])

            except requests.exceptions.RequestException as e:
                status.update(label="❌ Could not connect to the backend.", state="error")
                st.error(f"{e}")
            except Exception as e:
                status.update(label="❌ Unexpected error occurred.", state="error")
                st.error(f"{e}")
    else:
        st.warning("Please enter a question before submitting.")

# Sidebar Info
st.sidebar.header("How it Works")
st.sidebar.markdown("""
1. Enter your question in natural language.
2. Click **Get Answer**.
3. The backend:
   - Finds relevant tables.
   - Generates a SQL query.
   - Executes the query.
   - Interprets the results.
4. The answer is returned and displayed clearly.
""")
st.sidebar.header("Project Files")
st.sidebar.markdown("""
- 🧠 Backend Logic: `app/main.py`, `app/sql_functions.py`
- 🔌 Backend Server: `backend.py`
- 🎨 Frontend UI: `frontend.py`
""")
