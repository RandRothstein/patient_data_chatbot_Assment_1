import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import os
import streamlit as st

# ---------------------------------------------------------
# MODEL CONFIG
# ---------------------------------------------------------
MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"


# ---------------------------------------------------------
# DEFAULT CSV PATHS
# ---------------------------------------------------------
DEFAULT_DF1_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "sample_data",
    "Health_Dataset_1_Sample.csv"
)

DEFAULT_DF2_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "sample_data",
    "Health_Dataset_2_Sample.csv"
)

# ---------------------------------------------------------
# LOAD DEFAULT CSVs
# ---------------------------------------------------------
df1_default = pd.read_csv(DEFAULT_DF1_PATH)
df2_default = pd.read_csv(DEFAULT_DF2_PATH)

df1_default["Patient_Number"] = df1_default["Patient_Number"].astype(str)
df2_default["Patient_Number"] = df2_default["Patient_Number"].astype(str)

# ACTIVE DFS (will be replaced if user uploads CSV)
ACTIVE_DF1 = df1_default.copy()
ACTIVE_DF2 = df2_default.copy()


# ---------------------------------------------------------
# ALLOW UI TO UPDATE CSVs
# ---------------------------------------------------------
def update_active_data(df1_uploaded=None, df2_uploaded=None):
    """Update ACTIVE_DF1 / ACTIVE_DF2 if the user uploads CSVs."""
    global ACTIVE_DF1, ACTIVE_DF2

    if df1_uploaded is not None:
        df1_uploaded["Patient_Number"] = df1_uploaded["Patient_Number"].astype(str)
        ACTIVE_DF1 = df1_uploaded

    if df2_uploaded is not None:
        df2_uploaded["Patient_Number"] = df2_uploaded["Patient_Number"].astype(str)
        ACTIVE_DF2 = df2_uploaded


# ---------------------------------------------------------
# MERGE PATIENT RECORD
# ---------------------------------------------------------
def get_merged_patient_record(patient_number: int):
    """Merges ACTIVE_DF1 and ACTIVE_DF2 and returns one patient's row as dict."""
    merged = ACTIVE_DF1.merge(ACTIVE_DF2, on="Patient_Number", how="left")
    result = merged[merged["Patient_Number"] == str(patient_number)]

    if result.empty:
        return None

    return result.iloc[0].to_dict()


# ---------------------------------------------------------
# LOAD MODEL (Cached — Loads ONLY Once)
# ---------------------------------------------------------
@st.cache_resource(show_spinner=True)
def load_llm_model():
    #print(f"Loading model: {MODEL_NAME}...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    qa = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        #device=0 if torch.cuda.is_available() else -1
    )

    print("Model loaded successfully.")
    return qa, tokenizer


# Load only once (cached)
QA_PIPELINE, tokenizer = load_llm_model()


# ---------------------------------------------------------
# LLM RESPONSE HANDLER
# ---------------------------------------------------------
def ask_llm_for_explanation(prompt):
    response = QA_PIPELINE(
        prompt,
        max_new_tokens=120,
        do_sample=False,
        temperature=0.0,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    output = response[0]["generated_text"]

    # Extract content after "Final Response:"
    if "Final Response:" in output:
        answer = output.split("Final Response:", 1)[1].strip()
        return answer

    return output.strip()


# ---------------------------------------------------------
# PROMPT BUILDER
# ---------------------------------------------------------
def build_prompt(patient_record, user_question):
    record_str = "\n".join([f"- {k.replace('_', ' ')}: {v}" for k, v in patient_record.items()])
    prompt = f"""
You are a medical data assistant. Answer the user's question directly using ONLY the patient data.

RULES:
- Ensure the final generated text is a grammatically complete sentence, even if the max_new_tokens limit is reached but ensure to complete within max_new_tokens of 100 characters.
- First, provide ONLY the direct answer to the user's question.
- If the data shows values that are unusually high, low, or risky, you may add an additional section:
  "Additional Insight (non-diagnostic): ..."
- Insights must NOT be diagnostic or prescriptive.
- Do NOT add chain-of-thought, follow-up questions, or long explanations.
- Do NOT continue the conversation beyond the answer and optional insight.
- Genetic Pedigree Coefficient (GPC) of an individual for a particular disease is a continuum between 0 and 1, where:
  GPC closer to 0 indicates very distant occurrence of that disease in her/his pedigree, and
  GPC closer to 1 indicates very immediate occurrence of that disease in her/his pedigree.

Patient Data:
{record_str}

User Question: {user_question}

FORMAT:
Answer: <the direct answer>
Additional Insight (non-diagnostic): <only if applicable>

Strictly return nothing else other than the mentioned format.

Final Response:"""
    return prompt.strip()
