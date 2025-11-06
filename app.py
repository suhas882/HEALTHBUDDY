
import os
import json
import tempfile
import zipfile
import shutil
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import pandas as pd 
import numpy as np 
from fuzzywuzzy import process 


BASE = Path(__file__).resolve().parent
SINGLE_MODEL_FILE = BASE / "model.spacy" 
QA_MODEL_DIR = BASE / "qa_model" 
DRUG_FILE = BASE / "Medicine_Details.csv" 

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)
DRUG_OVERRIDES = {
    "typhoid": ["Ciprofloxacin", "Azithromycin", "Ceftriaxone"],
    "malaria": ["Chloroquine", "Hydrochloroquine", "Primaquine"],
    "fungal infection": ["Clotrimazole Cream", "Fluconazole Tablet", "Miconazole Powder"],
    "allergy": ["Cetirizine", "Loratadine", "Fexofenadine"],
}

def load_spacy_model_from_spacyfile(spacyfile_path: Path):
    import spacy
    if not spacyfile_path.exists():
        return None, None
    tempdir = Path(tempfile.mkdtemp(prefix="nlp_model_"))
    try:
        with zipfile.ZipFile(str(spacyfile_path), 'r') as zf:
            zf.extractall(tempdir)
        try:
            nlp = spacy.load(str(tempdir))
            return nlp, tempdir
        except Exception:
            for child in tempdir.iterdir():
                if child.is_dir():
                    try:
                        nlp = spacy.load(str(child))
                        return nlp, tempdir
                    except Exception:
                        continue
            shutil.rmtree(tempdir)
            return None, None
    except Exception as e:
        print("Error unpacking/loading model.spacy:", e)
        if tempdir.exists():
            shutil.rmtree(tempdir)
        return None, None


def load_diseases():
    try:
        df_symptoms = pd.read_csv(BASE / "DiseaseAndSymptoms.csv")
        df_precaution = pd.read_csv(BASE / "Disease precaution.csv")

        symptom_cols = [col for col in df_symptoms.columns if col.startswith("Symptom")]
        df_long = df_symptoms.melt(id_vars=["Disease"], value_vars=symptom_cols, value_name="Symptom")
        df_long = df_long.dropna(subset=["Symptom"]).drop(columns=["variable"])
        df_long['Symptom'] = df_long['Symptom'].astype(str).str.strip().str.lower().str.replace('_', ' ')
        disease_symptoms = df_long.groupby("Disease")["Symptom"].unique().apply(list).reset_index(name="symptoms")

        precaution_cols = [col for col in df_precaution.columns if col.startswith("Precaution")]
        df_prec_long = df_precaution.melt(id_vars=["Disease"], value_vars=precaution_cols, value_name="precaution")
        df_prec_long = df_prec_long.dropna(subset=["precaution"]).drop(columns=["variable"])
        df_prec_long['precaution'] = df_prec_long['precaution'].astype(str).str.strip()
        disease_precautions = df_prec_long.groupby("Disease")["precaution"].apply(lambda x: "; ".join(x.unique())).reset_index(name="treatment")

        df_merged = pd.merge(disease_symptoms, disease_precautions, on="Disease", how="outer")
        
        diseases_list = []
        for index, row in df_merged.iterrows():
            symptoms_to_use = row["symptoms"] if isinstance(row["symptoms"], list) else []
            treatment_to_use = row["treatment"] if isinstance(row["treatment"], str) else ""

            diseases_list.append({
                "name": row["Disease"].strip(),
                "symptoms": symptoms_to_use,
                "treatment": treatment_to_use,
                "type": "medical condition",
                "causes": ""
            })
            
        print(f"Successfully loaded {len(diseases_list)} unique diseases from CSV files.")
        return diseases_list

    except FileNotFoundError:
        print("Required Disease CSV files not found. Ensure they are present.")
        return []
    except Exception as e:
        print(f"Failed to load or process Disease CSV dataset: {e}")
        return []

def load_drug_data():
    """Loads and cleans drug data from Medicine_Details.csv."""
    try:
        df_drugs = pd.read_csv(DRUG_FILE)
        
        df_drugs = df_drugs[['Medicine Name', 'Uses', 'Side_effects']].copy()
        df_drugs.columns = ['name', 'uses', 'side_effects']
        
        df_drugs['name'] = df_drugs['name'].astype(str).str.strip()
        df_drugs['uses'] = df_drugs['uses'].astype(str).str.strip().str.replace('\n', ' ').str.lower()
        df_drugs['side_effects'] = df_drugs['side_effects'].astype(str).str.strip().str.replace('\n', ' ')
        
        df_drugs = df_drugs.dropna(subset=['name'])

        drugs_list = df_drugs.to_dict('records')
        print(f"Successfully loaded {len(drugs_list)} drugs from {DRUG_FILE.name}.")
        return drugs_list
        
    except FileNotFoundError:
        print(f"Drug file ({DRUG_FILE.name}) not found. Drug lookup will be unavailable.")
        return []
    except Exception as e:
        print(f"Failed to load or process drug dataset: {e}")
        return []


diseases = load_diseases()
drugs_data = load_drug_data() 


nlp = None
_temp_model_dir = None
try:
    nlp, _temp_model_dir = load_spacy_model_from_spacyfile(SINGLE_MODEL_FILE)
    if nlp:
        print("Loaded custom spaCy model from", SINGLE_MODEL_FILE)
    else:
        import spacy as _sp
        print("Falling back to 'en_core_web_sm' model.")
        nlp = _sp.load("en_core_web_sm")
except Exception as e:
    import spacy as _sp
    nlp = _sp.blank("en") if not hasattr(_sp, "load") else _sp.load("en_core_web_sm")

def load_qa_pipeline():
    try:
        from transformers import pipeline, DistilBertForQuestionAnswering, DistilBertTokenizerFast
        if QA_MODEL_DIR.exists() and any(QA_MODEL_DIR.iterdir()):
            print("Loading QA model from", QA_MODEL_DIR)
            tokenizer = DistilBertTokenizerFast.from_pretrained(str(QA_MODEL_DIR))
            model = DistilBertForQuestionAnswering.from_pretrained(str(QA_MODEL_DIR))
            return pipeline("question-answering", model=model, tokenizer=tokenizer)
        else:
            print("Loading default distilbert QA pipeline.")
            return pipeline("question-answering", model="distilbert-base-uncased-distilled-squad", tokenizer="distilbert-base-uncased-distilled-squad")
    except Exception as e:
        print("QA pipeline could not be loaded:", e)
        return None

qa_pipeline = load_qa_pipeline()

def extract_symptoms(text):
    try:
        doc = nlp(text)
        syms = [ent.text for ent in doc.ents if ent.label_.upper() == "SYMPTOM"]
        if syms:
            return syms
    except Exception as e:
        print("NER extraction error:", e)
    simple = [s.strip() for s in text.replace(" and ", ",").split(",") if s.strip()]
    return simple

def match_disease_fuzzy(symptoms, diseases_list, score_cutoff=60, threshold=1, max_results=5):
    
    def naive_match(symptoms_list):
        symptoms_lower = [s.lower() for s in symptoms_list]
        matched = []
        for d in diseases_list:
            ds = [s.lower() for s in d.get("symptoms", []) if isinstance(s, str)]
            cnt = sum(1 for s_in in symptoms_lower if s_in in " ".join(ds))
            if cnt >= threshold:
                matched.append((d["name"], d.get("symptoms",[]), cnt))
        if not matched:
            return "No match found. Try clearer symptoms."
        
        matched.sort(key=lambda x: x[2], reverse=True)
        lines = []
        for name, slist, cnt in matched[:max_results]:
            lines.append(f"**{name}** — matched: {cnt}. Key symptoms: {', '.join(slist)}")
        return "\n\n".join(lines)

    try:
        matched = []
        for d in diseases_list:
            ds = [s.lower() for s in d.get("symptoms", []) if isinstance(s, str)]
            cnt = 0
            for s in symptoms:
                best = process.extractOne(s.lower(), ds)
                if best and best[1] >= score_cutoff:
                    cnt += 1
            if cnt >= threshold:
                matched.append((d["name"], d.get("symptoms", []), cnt))
        
        if not matched:
            return "I couldn't match any disease. Try different symptom wording."
        
        matched.sort(key=lambda x: x[2], reverse=True)
        lines = []
        for name, slist, cnt in matched[:max_results]:
            lines.append(f"**{name}** — matched: {cnt}. Key symptoms: {', '.join(slist)}")
        return "\n\n".join(lines)
        
    except Exception as e:
        print(f"Fuzzywuzzy matching failed ({e}). Falling back to simple word count.")
        return naive_match(symptoms)

def answer_with_qa(question):
    if not qa_pipeline:
        return "QA system unavailable."
    context = " ".join(
        f"{d.get('name','')} is a {d.get('type','disease')}. Symptoms: {', '.join(d.get('symptoms',[]))}. Causes: {d.get('causes','')}. Treatment: {d.get('treatment','')}."
        for d in diseases
    )
    if not context.strip():
        return "No disease context loaded."
    try:
        res = qa_pipeline(question=question, context=context)
        return res.get("answer", "No answer found.")
    except Exception as e:
        print("QA error:", e)
        return "Error during QA."

@app.route("/")
def home():
    return render_template("indexchat.html")

@app.route("/chat", methods=["POST"])
def chat():
    payload = request.get_json(force=True)
    message = payload.get("message", "").strip()
    if not message:
        return jsonify({"reply":"Please send a message."}), 400
    
    msg_lower = message.lower()

    if drugs_data and any(k in msg_lower for k in ["tablet", "medicine", "drug", "medication", "dose"]):
        
        all_drug_names = [d.get('name') for d in drugs_data if d.get('name')]
        if all_drug_names:
            best_match = process.extractOne(msg_lower, all_drug_names, score_cutoff=75)
            if best_match:
                drug_name = best_match[0]
                drug_match = next((d for d in drugs_data if d.get("name") == drug_name), None)
                
                if drug_match:
                    reply = f"**{drug_match.get('name')}**\n\n**Uses:** {drug_match.get('uses')}\n\n**Common Side Effects:** {drug_match.get('side_effects')}"
                    return jsonify({"reply": reply})
        
        return jsonify({"reply":"I can provide information on medications. What specific medicine name are you asking about?"})


    name_match = next((d for d in diseases if d.get("name","").lower() in msg_lower), None)
    if name_match:
        d = name_match
        disease_name = d.get('name')
        disease_key = disease_name.lower().strip() # Key for dictionary lookup
        
        suggested_tablets = []

        if disease_key in DRUG_OVERRIDES:
            suggested_tablets = DRUG_OVERRIDES[disease_key]
        
        elif drugs_data:
            for drug in drugs_data:
                if disease_key in drug.get('uses', ''):
                    suggested_tablets.append(drug.get('name'))
        
        tablet_list = "; ".join(suggested_tablets[:3]) if suggested_tablets else "No specific drug suggestion available in the database."
        
        reply = f"**{disease_name}**\n"
        reply += f"- **Suggested Tablets/Drugs:** {tablet_list}\n"
        reply += f"- Symptoms: {', '.join(d.get('symptoms',[]))}\n"
        reply += f"- Treatment (Precautions): {d.get('treatment','')}"
        
        return jsonify({"reply": reply})

    if "symptom" in msg_lower or any(k in msg_lower for k in ["i have", "i am experiencing", "i feel", ","]):
        syms = extract_symptoms(message)
        if syms:
            matched = match_disease_fuzzy(syms, diseases)
            return jsonify({"reply": matched})
        else:
            return jsonify({"reply":"Could not extract symptoms. Try 'fever, cough' format."})

    ans = answer_with_qa(message)
    return jsonify({"reply": ans})

@app.route("/favicon.ico")
def favicon():
    return "", 204

@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory("static", filename)

if __name__ == "__main__":
    try:
        print("Starting Flask server on http://127.0.0.1:5000")
        app.run(host="0.0.0.0", port=5000, debug=True)
    finally:

        try:
            if '_temp_model_dir' in globals() and globals()['_temp_model_dir']:
                shutil.rmtree(globals()['_temp_model_dir'], ignore_errors=True)
        except Exception:
            pass
