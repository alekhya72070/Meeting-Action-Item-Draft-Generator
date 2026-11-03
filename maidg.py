
import os
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"

import re
import pandas as pd
import streamlit as st
from transformers import pipeline

st.set_page_config(
    page_title="Meeting Action-Item Draft Generator",
    page_icon="📝",
    layout="wide"
)

st.title("Meeting Action-Item Draft Generator")
st.write("Convert meeting notes into draft action items using AI.")


@st.cache_resource
def load_model():
    return pipeline(
        "text2text-generation",
        model="google/flan-t5-small",
        framework="pt"
    )


def extract_action_items(notes):
    items = []
    sentences = re.split(r"(?<=[.!?])\s+", notes.strip())

    for sentence in sentences:
        match = re.search(
            r"\b([A-Z][a-z]+)\s+will\s+(.+)",
            sentence.strip()
        )

        if match:
            owner = match.group(1)
            task_text = match.group(2).strip().rstrip(".")

        elif re.search(r"\bthe team will\b", sentence, re.I):
            owner = "Team"
            task_text = re.split(
                r"\bwill\b", sentence, maxsplit=1, flags=re.I
            )[-1].strip().rstrip(".")
        else:
            continue

        deadline_match = re.search(
            r"\b(?:by|on|before|next)\s+"
            r"(?:Monday|Tuesday|Wednesday|Thursday|Friday|"
            r"Saturday|Sunday|week|month)",
            task_text,
            re.I
        )

        if deadline_match:
            deadline = deadline_match.group(0)
            task = (
                task_text[:deadline_match.start()]
                + task_text[deadline_match.end():]
            ).strip(" ,.")
        else:
            deadline = "Not specified"
            task = task_text

        items.append({
            "Owner": owner,
            "Action Item": task if task else task_text,
            "Deadline": deadline
        })

    return items


notes = st.text_area(
    "Enter Meeting Notes",
    value=(
        "Ravi will prepare the project report by Friday. "
        "Priya will schedule the client meeting next Monday. "
        "The team will review the design."
    ),
    height=180
)

if st.button("Generate Action Items", type="primary"):
    if not notes.strip():
        st.warning("Please enter meeting notes.")
    else:
        with st.spinner("AI is generating your draft..."):
            st.session_state["items"] = extract_action_items(notes)

            try:
                model = load_model()
                result = model(
                    "Rewrite these meeting notes clearly without "
                    "changing names or deadlines: " + notes[:1000],
                    max_new_tokens=150,
                    do_sample=False
                )
                st.session_state["draft"] = result[0]["generated_text"]
                st.session_state["ai_failed"] = False

            except Exception as error:
                st.session_state["draft"] = notes
                st.session_state["ai_failed"] = True
                st.session_state["ai_error"] = str(error)

if "draft" in st.session_state:
    st.subheader("Generated Draft")
    st.write(st.session_state["draft"])

    if st.session_state.get("ai_failed"):
        st.error("AI model could not load. See technical details below.")
        with st.expander("Technical error details", expanded=True):
            st.code(st.session_state["ai_error"])

if "items" in st.session_state:
    st.subheader("Action Items Table")

    if st.session_state["items"]:
        df = pd.DataFrame(st.session_state["items"])
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.download_button(
            "Download Action Items (CSV)",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="meeting_action_items.csv",
            mime="text/csv",
            key="download_action_items_csv"
        )
    else:
        st.info("No action items detected in these notes.")

st.divider()
st.caption("Review AI-generated drafts before sharing.")
