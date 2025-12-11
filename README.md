# Patient Health Analytics (LLM-backed)

This repository contains a Streamlit UI and a backend that loads a HuggingFace causal LLM to generate **non-diagnostic** insights from merged patient data.
<img width="673" height="857" alt="image" src="https://github.com/user-attachments/assets/ecd839ce-2492-4027-9c19-65621de9a46a" />

## Quickstart

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git)
    cd YOUR_REPOSITORY_NAME
    ```
    (Replace `YOUR_USERNAME` and `YOUR_REPOSITORY_NAME` with your actual GitHub details)

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Streamlit application:**
    ```bash
    streamlit run ui/app.py
    ```
    This will open the application in your web browser, usually at `http://localhost:8501`.
