import os
import streamlit as st
import wandb
from pages.chatbot import (
    find_relevant_markdown_chunks,
    format_markdown_context,
    format_context,
    generate_sql_from_query,
    find_relevant_attributes_with_sql,
    llm_choose_tool,
    get_groq_chat_response,
    leoni_attributes_schema_for_main_loop,
    llm
)
from sentence_transformers import SentenceTransformer, util
from supabase import create_client
import pandas as pd
import re

# --- Ground truth ---
ground_truth = [
  {
    "question": "Number Of Fuse-Circuits",
    "answer": "For fuses with 1 fuse circuit it's 1. A higher number of fuse circuits is only related to multifuses. The number describes the amount of fuse circuits in a multifuse (e.g. 5 circuits)."
  },
  {
    "question": "What are the rules for the 'Strip Length [mm]' attribute for electric contacts?",
    "answer": "- Use the value specified on the supplier drawing.\n- If only max and min are given, enter their average (e.g. (3.75 + 4.25)/2 = 4.00 mm).\n- If only max or min is given, use that value.\n- If no supplier data exists, calculate approximately: ≤ 1 mm² → S = X + 1 mm; 1 mm² < X ≤ 16 mm² → S = X + 2 mm; > 16 mm² → S = X + 3 mm.\n- If the wire size spans a boundary (e.g. 0.5–1.5 mm²), choose the average."
  },
  {
  "question": "What is the Type Of Inductor?",
  "answer": "Chip inductor: designed for SMD mounting on a PCB\nCoil: wire-wound inductor\nOne core double choke: single core with two independent wire coils\nRF inductor: spaced windings for high-frequency applications\nRing choke: (not defined in the doc, placeholder)\nFilter, Ferrit, CAN-choke: listed under RF inductors without detailed definitions"
  },
  {
    "question": "The grease consists of which material?",
    "answer": "Grease is a semisolid lubricant, generally a soap emulsified with mineral or vegetable oil."
  },
  {
    "question": "The sealant is used for what?",
    "answer": "A sealant is used to ensure sealing of the harness on specific spots: as additional material within a grommet; kneaded between cables/wires of a harness."
  },
  {
    "question": "What is the second name of Elongation at break?",
    "answer": "Elongation at break, also known as fracture strain, is the ratio between changed length and initial length after breakage."
  },
  {
    "question": "What are the common attributes for Cavity plug?",
    "answer": "Shape: Round, Oval or Rectangular; External Diameter/Length & Width: dimensions critiques pour l'étanchéité; Material Name & Material Filling: définissent la résistance mécanique/environnementale; All Cavities Closed: implicite par le nom, mais non listé comme attribut distinct."
  },
  {
    "question": "Define LED.",
    "answer": "A light-emitting diode (LED) is a two-lead semiconductor light source, functioning like a pn-junction diode that emits light when forward-biased. Mounting Technology: THT, SMD."
  },
  {
    "question": "What is the connection type of relay?",
    "answer": "Plug-in: The relay is inserted into a relay holder; male terminals mate with a holder's female terminals.\nScrewed: Contacts secured via screws, typically for high-current applications.\nSoldering SMD: Surface-mounted device (SMD): glued to the PCB first, then soldered en masse."
  }
]

# Helper: get chatbot answer (wraps existing logic, does not change it)
def get_chatbot_answer(question):
    # Use the same logic as the chatbot UI
    tool_choice = llm_choose_tool(question, llm)
    relevant_attribute_rows = []
    relevant_markdown_chunks = []
    context_was_found = False
    if tool_choice == "sql":
        generated_sql = generate_sql_from_query(question, leoni_attributes_schema_for_main_loop)
        if generated_sql:
            relevant_attribute_rows = find_relevant_attributes_with_sql(generated_sql)
            context_was_found = bool(relevant_attribute_rows)
    relevant_markdown_chunks = find_relevant_markdown_chunks(question, limit=3)
    if relevant_markdown_chunks:
        context_was_found = True
    attribute_context = format_context(relevant_attribute_rows)
    markdown_context = format_markdown_context(relevant_markdown_chunks)
    combined_context = ""
    if relevant_attribute_rows:
        combined_context += f"**Database Attributes Information:**\n{attribute_context}\n\n"
    if relevant_markdown_chunks:
        combined_context += f"**Documentation/Standards Information:**\n{markdown_context}\n\n"
    if not combined_context:
        combined_context = "No relevant information found in the knowledge base (attributes or documentation)."
    prompt_for_llm = f"Context:\n{combined_context}\n\nUser Question: {question}\n"
    return get_groq_chat_response(prompt_for_llm, context_provided=context_was_found)

# Authenticate wandb using Streamlit secrets
os.environ["WANDB_API_KEY"] = st.secrets["WANDB_API_KEY"]

st.title("Document Search Evaluation with wandb")
st.write("This page evaluates your document search using the provided ground truth and logs results to wandb.")

# Move the chatbot vs ground truth evaluation button to the very top
if st.button("Run Chatbot vs Ground Truth Evaluation"):
    try:
        wandb.init(project="chatbot-vs-gt-eval")
        wandb_initialized = True
    except Exception as e:
        st.error(f"Failed to initialize wandb: {e}")
        wandb_initialized = False
    
    st.subheader("🤖 Chatbot Output vs Ground Truth Evaluation")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    hits = 0
    results = []
    progress = st.progress(0)
    # Metrics setup
    context_precisions = []
    context_recalls = []
    context_f1s = []
    SIMILARITY_THRESHOLD = 0.7  # Use a variable for easy adjustment
    for idx, item in enumerate(ground_truth):
        question = item["question"]
        expected_answer = item["answer"]
        try:
            chatbot_answer = get_chatbot_answer(question)
            emb_gt = model.encode(expected_answer, convert_to_tensor=True)
            emb_cb = model.encode(chatbot_answer, convert_to_tensor=True)
            similarity = util.pytorch_cos_sim(emb_gt, emb_cb).item()
            hit = similarity > SIMILARITY_THRESHOLD
            hits += int(hit)
            # Standard Context Precision: compare chatbot answer to ground truth in chunks
            # For chatbot, treat the answer as a single chunk, but for standard, you would use multiple chunks
            # Here, let's simulate by splitting the chatbot answer into sentences (as pseudo-chunks)
            answer_chunks = [s.strip() for s in re.split(r'[.!?]', chatbot_answer) if s.strip()]
            relevant_chunks = 0
            for chunk in answer_chunks:
                emb_chunk = model.encode(chunk, convert_to_tensor=True)
                sim_chunk = util.pytorch_cos_sim(emb_gt, emb_chunk).item()
                if sim_chunk > SIMILARITY_THRESHOLD:
                    relevant_chunks += 1
            context_precision = relevant_chunks / len(answer_chunks) if answer_chunks else 0
            context_precisions.append(context_precision)
            # Context Recall and F1 remain as before (since we have only one answer per question)
            context_recall = context_precision
            context_recalls.append(context_recall)
            context_f1 = context_precision
            context_f1s.append(context_f1)
            results.append({
                "question": question,
                "expected_answer": expected_answer,
                "chatbot_answer": chatbot_answer,
                "similarity": similarity,
                "hit": hit,
                "context_precision": context_precision,
                "context_recall": context_recall,
                "context_f1": context_f1
            })
            if wandb_initialized:
                wandb.log({
                    "question": question,
                    "expected_answer": expected_answer,
                    "chatbot_answer": chatbot_answer,
                    "similarity": similarity,
                    "hit": hit,
                    "context_precision": context_precision,
                    "context_recall": context_recall,
                    "context_f1": context_f1
                })
            with st.expander(f"{'✅' if hit else '❌'} Q{idx+1}: {question[:50]}...", expanded=False):
                st.markdown(f"**Question:** {question}")
                st.markdown(f"**Expected Answer:** {expected_answer}")
                st.markdown(f"**Chatbot Answer:** {chatbot_answer[:500]}{'...' if len(chatbot_answer) > 500 else ''}")
                st.markdown(f"**Similarity Score:** {similarity:.3f}")
                st.markdown(f"**Hit:** {'✅ Yes' if hit else '❌ No'}")
                st.markdown(f"**Standard Context Precision:** {context_precision:.3f}")
                st.markdown(f"**Context Recall:** {context_recall:.3f}")
                st.markdown(f"**Context F1:** {context_f1:.3f}")
        except Exception as e:
            st.error(f"Error for question '{question}': {e}")
            if wandb_initialized:
                wandb.log({"error": f"{question}: {e}"})
        progress.progress((idx + 1) / len(ground_truth))
    accuracy = hits / len(ground_truth)
    context_precisions = [r["context_precision"] for r in results]
    context_recalls = [r["context_recall"] for r in results]
    context_f1s = [r["context_f1"] for r in results]
    similarities = [r["similarity"] for r in results]
    avg_context_precision = sum(context_precisions) / len(context_precisions) if context_precisions else 0.0
    avg_context_recall = sum(context_recalls) / len(context_recalls) if context_recalls else 0.0
    avg_context_f1 = sum(context_f1s) / len(context_f1s) if context_f1s else 0.0
    avg_answer_correctness = sum(similarities) / len(similarities) if similarities else 0.0
    st.success(f"🤖 **Chatbot Evaluation Complete!** Final Accuracy: {accuracy:.1%} ({hits}/{len(ground_truth)} questions answered correctly)")
    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
    with col1:
        st.metric("Accuracy", f"{accuracy:.1%}")
    with col2:
        st.metric("Avg Standard Context Precision", f"{avg_context_precision:.3f}")
    with col3:
        st.metric("Avg Context Recall", f"{avg_context_recall:.3f}")
    with col4:
        st.metric("Avg Context F1", f"{avg_context_f1:.3f}")
    with col5:
        st.metric("Avg Answer Correctness Score", f"{avg_answer_correctness:.3f}")
    with col6:
        st.metric("Total Questions", f"{len(ground_truth)}")
    with col7:
        st.metric("Hits", f"{hits}")
    if wandb_initialized:
        wandb.log({
            "accuracy": accuracy,
            "avg_context_precision": avg_context_precision,
            "avg_context_recall": avg_context_recall,
            "avg_context_f1": avg_context_f1,
            "avg_answer_correctness": avg_answer_correctness,
            "total_questions": len(ground_truth),
            "hits": hits
        })
        wandb.finish()
    st.info("📊 **Results logged to wandb** - Check your dashboard for detailed analytics and charts.")

with st.sidebar:
    st.markdown("<h2 style='color:white;'>Navigation</h2>", unsafe_allow_html=True)
    if st.button("🏠 Home", key="home_btn"):
        st.switch_page("app.py")
    if st.button("💬 Chat with Leoparts", key="chat_btn"):
        st.switch_page("pages/chatbot.py")
    if st.button("📄 Extract a new Part", key="extract_btn"):
        st.switch_page("pages/extraction_attributs.py")
    if st.button("🆕 New conversation", key="new_conv_btn"):
        st.session_state.messages = []
        st.session_state.last_part_number = None
        st.rerun()
    if st.button("📊 Evaluate Doc Search", key="eval_btn"):
        st.switch_page("pages/evaluate_doc_search.py")
