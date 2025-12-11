import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import sys
import os

#MODEL_NAME = "tiiuae/falcon-7b-instruct"
MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"

# Load CSVs from data/ directory
try:
    df1 = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "sample_data", "Health_Dataset_1_Sample.csv"))
    df2 = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "sample_data", "Health_Dataset_2_Sample.csv"))
except FileNotFoundError as e:
    print(f"Error: {e}. Place CSVs in the project's data/ folder.")
    raise

# Ensure Patient_Number is string for stable merging/filtering
df1["Patient_Number"] = df1["Patient_Number"].astype(str)
df2["Patient_Number"] = df2["Patient_Number"].astype(str)

def get_merged_patient_record(patient_number: int):
    """Merges df1 and df2 and returns one patient's row as dict (or None)."""
    merged = df1.merge(df2, on="Patient_Number", how="left")
    result = merged[merged["Patient_Number"] == str(patient_number)]
    if result.empty:
        return None
    return result.iloc[0].to_dict()

# -------------------- MODEL LOADING --------------------
try:
    print(f"Loading model: {MODEL_NAME}...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    QA_PIPELINE = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer
    )
    print("Model loaded successfully.")

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

        # Extract only the section after "Final Response:"
        if "Final Response:" in output:
            answer = output.split("Final Response:", 1)[1].strip()
            return answer
        
        return output.strip()


except Exception as e:
    print(f"ERROR loading model: {e}")
    # re-raise so UI shows the problem
    raise

def build_prompt(patient_record, user_question):
    record_str = "\n".join([f"- {k.replace('_', ' ')}: {v}" for k, v in patient_record.items()])
    prompt = f"""
You are a medical data assistant. Answer the user's question directly using ONLY the patient data.

RULES:
- First, provide ONLY the direct answer to the user's question.
- If the data shows values that are unusually high, low, or risky, you may add an additional section:
  "Additional Insight (non-diagnostic): ..."
- Insights must NOT be diagnostic or prescriptive.
- Do NOT add chain-of-thought, follow-up questions, or long explanations.
- Do NOT continue the conversation beyond the answer and optional insight.
- Genetic Pedigree Coefficient (GPC) of an individual for a particular disease is a continuum between 0 and 1, where:
GPC closer to 0 indicates very distant occurrence of that disease in her/his pedigree, and
GPC closer to 1 indicates very immediate occurrence of that disease in her/his pedigree]


Patient Data:
{record_str}

User Question: {user_question}

FORMAT:
Answer: <the direct answer>
Additional Insight (non-diagnostic): <only if applicable>

Return nothing else.

Final Response:"""
    return prompt.strip()

