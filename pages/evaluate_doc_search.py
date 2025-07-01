import os
import streamlit as st
import wandb
from pages.chatbot import find_relevant_markdown_chunks
from sentence_transformers import SentenceTransformer, util
from supabase import create_client

# Authenticate wandb using Streamlit secrets
os.environ["WANDB_API_KEY"] = st.secrets["WANDB_API_KEY"]
wandb.login(key=os.environ["WANDB_API_KEY"])

# Initialize wandb BEFORE any wandb.log
wandb.init(project="leoparts-doc-search-eval")

st.title("Document Search Evaluation with wandb")
st.write("This page evaluates your document search using the provided ground truth and logs results to wandb.")

# --- Supabase connection test ---
supabase_url = st.secrets["SUPABASE_URL"]
supabase_key = st.secrets["SUPABASE_SERVICE_KEY"]
supabase = create_client(supabase_url, supabase_key)

st.subheader("Supabase Connection Test")
try:
    data = supabase.table("markdown_chunks").select("*").limit(1).execute()
    st.success(f"Sample data from markdown_chunks: {data.data}")
    wandb.log({"supabase_table_test": str(data.data)})
except Exception as e:
    st.error(f"Error fetching from markdown_chunks: {e}")
    wandb.log({"supabase_table_test_error": str(e)})

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
    model = SentenceTransformer("all-MiniLM-L6-v2")
    hits = 0
    results = []
    progress = st.progress(0)
    for idx, item in enumerate(ground_truth):
        question = item["question"]
        expected_answer = item["answer"]
        try:
            chunks = find_relevant_markdown_chunks(question, limit=3)
            retrieved_text = "\n".join(chunk.get("content", "") for chunk in chunks)
            emb_gt = model.encode(expected_answer, convert_to_tensor=True)
            emb_ret = model.encode(retrieved_text, convert_to_tensor=True)
            similarity = util.pytorch_cos_sim(emb_gt, emb_ret).item()
            hit = similarity > 0.7
            hits += int(hit)
            results.append({
                "question": question,
                "expected_answer": expected_answer,
                "retrieved_text": retrieved_text,
                "similarity": similarity,
                "hit": hit
            })
            wandb.log({
                "question": question,
                "expected_answer": expected_answer,
                "retrieved_text": retrieved_text,
                "similarity": similarity,
                "hit": hit
            })
            st.write(f"**Q:** {question}")
            st.write(f"**Similarity:** {similarity:.2f} | **Hit:** {hit}")
            st.write("---")
        except Exception as e:
            st.error(f"Error for question '{question}': {e}")
            wandb.log({"error": f"{question}: {e}"})
        progress.progress((idx + 1) / len(ground_truth))
    accuracy = hits / len(ground_truth)
    wandb.log({"accuracy": accuracy})
    wandb.finish()
    st.success(f"Evaluation complete! Accuracy: {accuracy:.2f}")

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
