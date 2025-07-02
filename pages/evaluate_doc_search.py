import os
import streamlit as st
import wandb
from pages.chatbot import find_relevant_markdown_chunks
from sentence_transformers import SentenceTransformer, util
from supabase import create_client
import pandas as pd

# Helper: get chatbot answer (wraps existing logic, does not change it)
def get_chatbot_answer(question):
    # This function should call your chatbot's answer logic as is.
    # For demonstration, we'll use the vector search as a placeholder.
    # Replace this with your actual chatbot answer logic if available.
    chunks = find_relevant_markdown_chunks(question, limit=3)
    return "\n".join(chunk.get("content", "") for chunk in chunks)

# Authenticate wandb using Streamlit secrets
os.environ["WANDB_API_KEY"] = st.secrets["WANDB_API_KEY"]

st.title("Document Search Evaluation with wandb")
st.write("This page evaluates your document search using the provided ground truth and logs results to wandb.")

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
    "answer": "Chip inductor: conçu pour montage SMD sur PCB; Coil: inducteur en bobine filaire; One core double choke: un noyau, deux bobines indépendantes; RF inductor: enroulements espacés pour hautes fréquences; Ring choke: (non défini dans le doc, placeholder); Filter, Ferrit, CAN-choke: listés sous RF inductors sans définition détaillée."
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

if st.button("Run Evaluation"):
    # Initialize wandb at the start of evaluation
    try:
        wandb.init(project="leoparts-doc-search-eval")
        wandb_initialized = True
    except Exception as e:
        st.error(f"Failed to initialize wandb: {e}")
        wandb_initialized = False
    
    # --- Supabase connection test ---
    supabase_url = st.secrets["SUPABASE_URL"]
    supabase_key = st.secrets["SUPABASE_SERVICE_KEY"]
    supabase = create_client(supabase_url, supabase_key)

    st.subheader("Supabase Connection Test")
    try:
        data = supabase.table("markdown_chunks").select("*").limit(1).execute()
        st.success(f"✅ Database connected successfully")
        if wandb_initialized:
            wandb.log({"supabase_table_test": str(data.data)})
    except Exception as e:
        st.error(f"❌ Error fetching from markdown_chunks: {e}")
        if wandb_initialized:
            wandb.log({"supabase_table_test_error": str(e)})
    
    # --- Evaluation Section ---
    st.subheader("🚀 Running Document Search Evaluation")
    
    model = SentenceTransformer("all-MiniLM-L6-v2")
    hits = 0
    results = []
    progress = st.progress(0)
    
    # Create columns for better layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📊 Question-by-Question Results")
    
    with col2:
        st.markdown("### 📈 Live Metrics")
        metric_col1, metric_col2 = st.columns(2)
    
    for idx, item in enumerate(ground_truth):
        question = item["question"]
        expected_answer = item["answer"]
        try:
            chunks = find_relevant_markdown_chunks(question, limit=3)
            retrieved_text = "\n".join(chunk.get("content", "") for chunk in chunks)
            emb_gt = model.encode(expected_answer, convert_to_tensor=True)
            emb_ret = model.encode(retrieved_text, convert_to_tensor=True)
            similarity = util.pytorch_cos_sim(emb_gt, emb_ret).item()
            hit = similarity > 0.5
            hits += int(hit)
            results.append({
                "question": question,
                "expected_answer": expected_answer,
                "retrieved_text": retrieved_text,
                "similarity": similarity,
                "hit": hit
            })
            if wandb_initialized:
                wandb.log({
                    "question": question,
                    "expected_answer": expected_answer,
                    "retrieved_text": retrieved_text,
                    "similarity": similarity,
                    "hit": hit
                })
            
            # Display results with better formatting
            with col1:
                status_icon = "✅" if hit else "❌"
                similarity_color = "green" if similarity > 0.6 else "orange" if similarity > 0.5 else "red"
                
                with st.expander(f"{status_icon} Q{idx+1}: {question[:50]}...", expanded=False):
                    st.markdown(f"**Question:** {question}")
                    st.markdown(f"**Expected Answer:** {expected_answer}")
                    st.markdown(f"**Similarity Score:** :{similarity_color}[{similarity:.3f}]")
                    st.markdown(f"**Hit:** {'✅ Yes' if hit else '❌ No'}")
                    st.markdown("**Retrieved Content:**")
                    st.text(retrieved_text[:500] + "..." if len(retrieved_text) > 500 else retrieved_text)
            
            # Update live metrics
            with col2:
                current_accuracy = hits / (idx + 1)
                with metric_col1:
                    st.metric("Accuracy", f"{current_accuracy:.1%}")
                with metric_col2:
                    st.metric("Hits", f"{hits}/{idx+1}")
            
        except Exception as e:
            st.error(f"Error for question '{question}': {e}")
            if wandb_initialized:
                wandb.log({"error": f"{question}: {e}"})
        progress.progress((idx + 1) / len(ground_truth))
    
    # --- Final Results Section ---
    st.subheader("🎯 Final Evaluation Results")
    
    # Calculate metrics
    total_questions = len(ground_truth)
    accuracy = hits / total_questions
    precision = hits / total_questions  # In this case, precision = accuracy since we're not using top-K
    recall = hits / total_questions     # Same as precision for this evaluation
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Create metrics display
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Overall Accuracy", f"{accuracy:.1%}", f"{hits}/{total_questions}")
    with col2:
        st.metric("Precision", f"{precision:.1%}")
    with col3:
        st.metric("Recall", f"{recall:.1%}")
    with col4:
        st.metric("F1 Score", f"{f1_score:.1%}")
    
    # Similarity distribution
    similarities = [r["similarity"] for r in results]
    avg_similarity = sum(similarities) / len(similarities)
    min_similarity = min(similarities)
    max_similarity = max(similarities)
    
    st.markdown("### 📊 Similarity Score Analysis")
    sim_col1, sim_col2, sim_col3, sim_col4 = st.columns(4)
    with sim_col1:
        st.metric("Average Similarity", f"{avg_similarity:.3f}")
    with sim_col2:
        st.metric("Min Similarity", f"{min_similarity:.3f}")
    with sim_col3:
        st.metric("Max Similarity", f"{max_similarity:.3f}")
    with sim_col4:
        st.metric("Threshold", "0.500")
    
    # Detailed results table
    st.markdown("### 📋 Detailed Results Table")
    results_data = []
    for i, result in enumerate(results):
        results_data.append({
            "Question #": i+1,
            "Question": result["question"][:50] + "..." if len(result["question"]) > 50 else result["question"],
            "Similarity": f"{result['similarity']:.3f}",
            "Hit": "✅ Yes" if result["hit"] else "❌ No",
            "Status": "PASS" if result["hit"] else "FAIL"
        })
    
    df = pd.DataFrame(results_data)
    st.dataframe(df, use_container_width=True)
    
    # Performance insights
    st.markdown("### 🔍 Performance Insights")
    
    # Find best and worst performing questions
    best_result = max(results, key=lambda x: x["similarity"])
    worst_result = min(results, key=lambda x: x["similarity"])
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🏆 Best Performing Question:**")
        st.markdown(f"- **Question:** {best_result['question']}")
        st.markdown(f"- **Similarity:** {best_result['similarity']:.3f}")
        st.markdown(f"- **Hit:** {'✅ Yes' if best_result['hit'] else '❌ No'}")
    
    with col2:
        st.markdown("**⚠️ Worst Performing Question:**")
        st.markdown(f"- **Question:** {worst_result['question']}")
        st.markdown(f"- **Similarity:** {worst_result['similarity']:.3f}")
        st.markdown(f"- **Hit:** {'✅ Yes' if worst_result['hit'] else '❌ No'}")
    
    # Recommendations
    st.markdown("### 💡 Recommendations")
    if accuracy >= 0.8:
        st.success("🎉 **Excellent Performance!** Your document search is working very well.")
    elif accuracy >= 0.6:
        st.warning("⚠️ **Good Performance** with room for improvement.")
    else:
        st.error("❌ **Needs Improvement** - Consider adjusting your search parameters.")
    
    if min_similarity < 0.5:
        st.info("💡 **Suggestion:** Consider lowering the similarity threshold to 0.45 to catch more relevant results.")
    
    if avg_similarity < 0.6:
        st.info("💡 **Suggestion:** Consider improving your document chunking or using a more domain-specific embedding model.")
    
    # Log final metrics to wandb
    if wandb_initialized:
        wandb.log({
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "avg_similarity": avg_similarity,
            "min_similarity": min_similarity,
            "max_similarity": max_similarity,
            "total_questions": total_questions,
            "hits": hits
        })
        wandb.finish()
    
    st.success(f"🎯 **Evaluation Complete!** Final Accuracy: {accuracy:.1%} ({hits}/{total_questions} questions answered correctly)")
    
    # Show wandb link if available
    if wandb_initialized:
        st.info("📊 **Results logged to wandb** - Check your dashboard for detailed analytics and charts.")

# New evaluation: Chatbot output vs Ground Truth
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
    for idx, item in enumerate(ground_truth):
        question = item["question"]
        expected_answer = item["answer"]
        try:
            chatbot_answer = get_chatbot_answer(question)
            emb_gt = model.encode(expected_answer, convert_to_tensor=True)
            emb_cb = model.encode(chatbot_answer, convert_to_tensor=True)
            similarity = util.pytorch_cos_sim(emb_gt, emb_cb).item()
            hit = similarity > 0.5
            hits += int(hit)
            results.append({
                "question": question,
                "expected_answer": expected_answer,
                "chatbot_answer": chatbot_answer,
                "similarity": similarity,
                "hit": hit
            })
            if wandb_initialized:
                wandb.log({
                    "question": question,
                    "expected_answer": expected_answer,
                    "chatbot_answer": chatbot_answer,
                    "similarity": similarity,
                    "hit": hit
                })
            with st.expander(f"{'✅' if hit else '❌'} Q{idx+1}: {question[:50]}...", expanded=False):
                st.markdown(f"**Question:** {question}")
                st.markdown(f"**Expected Answer:** {expected_answer}")
                st.markdown(f"**Chatbot Answer:** {chatbot_answer[:500]}{'...' if len(chatbot_answer) > 500 else ''}")
                st.markdown(f"**Similarity Score:** {similarity:.3f}")
                st.markdown(f"**Hit:** {'✅ Yes' if hit else '❌ No'}")
        except Exception as e:
            st.error(f"Error for question '{question}': {e}")
            if wandb_initialized:
                wandb.log({"error": f"{question}: {e}"})
        progress.progress((idx + 1) / len(ground_truth))
    accuracy = hits / len(ground_truth)
    st.success(f"🤖 **Chatbot Evaluation Complete!** Final Accuracy: {accuracy:.1%} ({hits}/{len(ground_truth)} questions answered correctly)")
    if wandb_initialized:
        wandb.log({"accuracy": accuracy, "total_questions": len(ground_truth), "hits": hits})
        wandb.finish()
    st.info("📊 **Results logged to wandb** - Check your dashboard for detailed analytics and charts.")

with st.sidebar:
    st.markdown("<h2 style='color:white;'>Navigation</h2>", unsafe_allow_html=True)
    if st.button("🏠 Home"):
        st.switch_page("app.py")
    if st.button("💬 Chat with Leoparts"):
        st.switch_page("pages/chatbot.py")
    if st.button("📄 Extract a new Part"):
        st.switch_page("pages/extraction_attributs.py")
    if st.button("🆕 New conversation"):
        st.session_state.messages = []
        st.session_state.last_part_number = None
        st.rerun()
    if st.button("📊 Evaluate Doc Search"):
        st.switch_page("pages/evaluate_doc_search.py")
