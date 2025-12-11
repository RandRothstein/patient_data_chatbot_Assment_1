import streamlit as st
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'app')))

from backend import get_merged_patient_record, build_prompt, ask_llm_for_explanation

st.set_page_config(page_title='Patient Health Analytics', layout='centered')

st.title('🧠 Patient Health Analytics (LLM-backed)')
st.write('Generate non-diagnostic, educational insights from patient data.')

with st.sidebar:
    st.header('Instructions')
    st.markdown('1. Enter a patient number.\n2. Ask a focused question.\n3. Click *Generate Insights*.')
    st.markdown('**Note:** The model may require a GPU and sufficient memory to load.')

patient_input = st.text_input('Patient Number', value='42')
question = st.text_area('Question about the patient', value="What insights can be drawn from the patient's lifestyle factors?", height=120)

generate = st.button('Generate Insights')

if generate:
    try:
        patient_number = int(patient_input)
    except ValueError:
        st.error('Please enter a valid integer for Patient Number.')
        st.stop()

    with st.spinner('Loading patient record...'):
        record = get_merged_patient_record(patient_number)

    if record is None:
        st.error(f'Patient {patient_number} not found in the dataset.')
    else:
        st.subheader('Patient Data (merged)')
        st.json(record)

        prompt = build_prompt(record, question)
        with st.spinner('Contacting the LLM to generate insights (may take time)...'):
            try:
                answer = ask_llm_for_explanation(prompt)
            except Exception as e:
                st.error(f'Error while generating response: {e}')
                st.stop()

        st.subheader('🔍 AI-Generated Insight')
        st.write(answer)
