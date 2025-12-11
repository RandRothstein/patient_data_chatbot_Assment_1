# Patient Health Analytics (LLM-backed)

This repository contains a Streamlit UI and a backend that loads a HuggingFace causal LLM to generate **non-diagnostic** insights from merged patient data.
<img width="653" height="832" alt="image" src="https://github.com/user-attachments/assets/16d3f321-b1be-4827-92d2-b9dab8405ceb" />


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


5.  **Install PyTorch with CUDA support (IMPORTANT)
    ```bash
    pip install torch --index-url https://download.pytorch.org/whl/cu121
    ```
    Your RTX 3060 uses CUDA 12.1, so install the correct GPU version:cu121


6.  ** To check your GPU
    ```bash
    python -c "import torch; print(torch.cuda.is_available())"
    ```
    

