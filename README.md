# HEALTHBUDDY
The ML-Based Medical Chatbot is an AI-powered healthcare assistant designed to predict diseases and suggest appropriate medicines based on user symptoms. It uses Machine Learning (ML) and Natural Language Processing (NLP) techniques to interpret natural language queries and generate intelligent responses

📘 Project Overview

The ML-Based Medical Chatbot is an intelligent healthcare assistant designed to predict diseases and suggest medicines based on user-input symptoms. It leverages Machine Learning (ML) and Natural Language Processing (NLP) to interpret natural language queries and provide accurate, relevant medical insights.

# This chatbot utilizes three structured datasets:

Disease–Symptom Dataset:
Maps various diseases to their associated symptoms, allowing the chatbot to identify potential illnesses based on user-described health conditions.

Medicine Details Dataset:
Associates each disease with recommended medicines, dosage information, and general treatment suggestions to guide users with relevant medical insights.

Disease Precaution Dataset:
Provides preventive measures and health precautions for each disease, helping users understand the necessary steps to manage or avoid specific conditions.
By integrating these datasets, the system provides a more complete and reliable response — not only predicting possible illnesses but also suggesting corresponding medicines and preventive advice. It uses a heuristic model combined with fuzzy string matching algorithms to handle misspelled or approximate symptom inputs effectively.

The model inference and logic processing, while the frontend (HTML, CSS, and JavaScript) provides a clean and interactive user interface. This project demonstrates how AI can assist in healthcare by offering instant, preliminary medical guidance in a conversational format.

⚙️ Features

Analyzes symptoms and predicts possible diseases

Suggests medicines with dosage details

Provides precautionary measures for prevention

Handles spelling errors using fuzzy text matching

Real-time interactive chatbot interface

Tools & Libraries:

## Programming Language:
Python – Core programming language for development
Flask – Backend framework for building the web-based chatbot interface
Pandas & NumPy – For dataset processing and analysis
Scikit-learn – For implementing the heuristic score model and prediction logic
FuzzyWuzzy – For fuzzy matching of user symptoms to diseases
HTML, CSS, JavaScript – For front-end user interface design
Jupyter Notebook – For data exploration and model testing


<img width="674" height="403" alt="image" src="https://github.com/user-attachments/assets/38f1a351-6b76-4a82-9171-f5d5df8d1ad4" />

🧩 How to Run the Project Locally

-> Clone this repository:

git clone https://github.com/<your-username>/ML-Chatbot.git
cd ML-Chatbot

-> Create and activate a virtual environment:

python -m venv venv
venv\Scripts\activate        # For Windows
or
source venv/bin/activate     # For Mac/Linux

-> Install the dependencies:

pip install -r requirements.txt

-> Run the Flask app:

python app.py

-> Open in browser:

http://127.0.0.1:5000/

🧠 Machine Learning Model

Uses DistilBERT, a lightweight Transformer model from Hugging Face, for semantic understanding of user inputs.

Fine-tuned on medical symptom–disease datasets to identify the most likely disease contextually.

Combines Transformer embeddings with a custom heuristic classifier for prediction.

Uses fuzzy string matching to handle user typos and approximate symptom names.

Integrates three datasets — symptoms, medicines, and precautions — to provide a holistic response.

## Why DistilBERT?
DistilBERT offers near BERT-level performance while being smaller and faster, making it ideal for real-time chatbot inference.


## 📸 Screenshots

<img width="1873" height="1150" alt="image" src="https://github.com/user-attachments/assets/799b064a-ff97-41bf-87f5-1e899ffafb95" />

<img width="1880" height="2575" alt="image" src="https://github.com/user-attachments/assets/c4dc3e12-5b0f-498a-a08c-e32dd99bcf6d" />



🏆 Future Enhancements

Voice-based user interaction (speech-to-text and text-to-speech)

Multi-language support using multilingual BERT

Integration with live medical databases (like Health API)

Deployment for public use on platforms like Hugging Face Spaces or Render

⚠️ Disclaimer

This chatbot is developed for educational and research purposes only.
It does not replace professional medical advice.
Users should always consult certified doctors before taking any medical action.
