from flask import Flask, request, jsonify
import joblib
import numpy as np
import requests
from sklearn.preprocessing import StandardScaler
import pandas as pd
import io
import os
app = Flask(__name__)

# ============ 1. ML DIABETES PREDICTION =================
model_url = "https://github.com/ahmed-morad15/diabetes-app/raw/main/Diabetes_Prediction_Model.pkl"
dataset_url = "https://github.com/ahmed-morad15/diabetes-app/raw/main/preprocessed_diabetes_data.csv"

try:
    response = requests.get(model_url)
    if response.status_code == 200:
        model = joblib.load(io.BytesIO(response.content))  
    else:
        print(f"Failed to download the model: {response.status_code}")
        model = None
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

try:
    response = requests.get(dataset_url)
    if response.status_code == 200:
        with open("preprocessed_diabetes_data.csv", "wb") as f:
            f.write(response.content)
        
        data = pd.read_csv("preprocessed_diabetes_data.csv")

        scaler = StandardScaler()
        numerical_features = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
        scaler.fit(data[numerical_features])  
    else:
        print(f"Failed to download the dataset: {response.status_code}")
        data = None
        scaler = None
except Exception as e:
    print(f"Error loading dataset or scaler: {e}")
    data = None
    scaler = None

# ==================== 2. Diabetes Prediction Function =====================
def predict_diabetes(input_data):
    if model is None or scaler is None:
        return None
    try:
        features = np.array([input_data[key] for key in [
            "gender", "age", "hypertension", "heart_disease",
            "smoking_history", "bmi", "HbA1c_level", "blood_glucose_level"
        ]], dtype=np.float32).reshape(1, -1)

        scaled_features = scaler.transform(features[:, 4:])  

        full_features = np.concatenate((
            features[:, :4],  
            scaled_features.flatten().reshape(1, -1)  
        ), axis=1)

        prediction = model.predict(full_features)
        return int(prediction[0])
    except Exception as e:
        return str(e)

# ============ 3. ENDPOINT: Predict Diabetes =============
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if not data:
        return jsonify({"error": "No input data provided"}), 400

    try:
        result = predict_diabetes(data)
        if result is None:
            return jsonify({"error": "Model not loaded or prediction failed"}), 500
        return jsonify({"diabetes_prediction": result})  
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# ============ 4. LLM CONFIG (OpenRouter API) ============
API_KEY = "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"
MODEL_NAME = "mistralai/mistral-7b-instruct"
LLM_API_URL = "https://openrouter.ai/api/v1/chat/completions"

def call_llm(prompt):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}]
    }
    
    try:
        res = requests.post(LLM_API_URL, headers=headers, json=payload)
        res.raise_for_status()  
        return res.json()['choices'][0]['message']['content']
    except requests.exceptions.RequestException as e:
        return f"Error with LLM API: {e}"

# ============ 5. ENDPOINT: Next Question ================
@app.route('/next-question', methods=['POST'])
def next_question():
    data = request.json
    last_q = data.get("last_question", "")
    last_a = data.get("last_answer", "")

    if not last_q and not last_a:
        prompt = """
    You are a diabetes assistant. Please generate a random **first question** to start a conversation with a diabetic patient.
    
    - The question should be short, clear, and health-related.
    - It should be relevant to lifestyle, symptoms, or diabetes self-care.
    - Avoid introducing yourself or saying hello. Just ask the question directly.
    - Vary the question each time to avoid repetition.
    - Focus on one of these areas: symptoms, nutrition, physical activity, medication, or blood sugar monitoring.
    """
    else:
        prompt = f"""
    You are an intelligent medical assistant specialized in diabetes care. Your goal is to **dynamically continue** a conversation with a diabetic patient by asking insightful, short, and medically-relevant questions.

    Context:
    - Previous question: "{last_q}"
    - Patient's answer: "{last_a}"

    Based on the context above:
    - Understand the patient's situation.
    - Think about the most logical follow-up topic or clarification.
    - Then, generate a **new question** that builds upon the previous one, or explores a connected aspect of diabetes care.
    - Focus on: blood glucose monitoring, medication, diet, physical activity, or medical follow-ups.
    - Do NOT repeat previous questions or ask about unrelated topics.

    Respond with the question only.
    """

    try:
        generated_question = call_llm(prompt)
        return jsonify({"next_question": generated_question.strip()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ============ 6. ENDPOINT: Generate Personalized Advice ==
@app.route('/generate-advice', methods=['POST'])
def generate_advice():
    history = request.json.get("qa_history", [])
    dialogue = ""
    for item in history:
        dialogue += f"Question: {item['question']}\nAnswer: {item['answer']}\n"

    prompt = f"""
    The patient has diabetes. Below is their interactive question-and-answer history:\n{dialogue}

     Based on this, generate **personalized medical advice** tailored to the patient’s condition.

    Guidelines:
    - Language: English
    - Format: Bullet points only (•)
    - Style: Brief, direct, and medically relevant
    - Tone: Friendly, supportive, and easy to understand

     Cover these key areas:
    • Nutrition  
    • Physical activity  
    • Medical follow-up  
    • Healthy daily habits

     Avoid:
    - Long explanations
    - Greetings or conclusions
    - Repeating the patient's answers

     Focus on clear, actionable advice that helps the patient manage diabetes effectively.
    """
    try:
        advice = call_llm(prompt)
        return jsonify({"advice": advice.strip()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
# ============ 7. ENDPOINT: ChatBot Message ===============
@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get("message", "")
    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    prompt = f"""
    You are a friendly and knowledgeable diabetes assistant.
    A patient has sent the following message:

    "{user_message}"

    Please respond with a helpful answer related to diabetes management or general advice.

    Guidelines:
    - Language: English
    - Keep your response **short and to the point** (2-3 sentences maximum)
    - Avoid long explanations or medical jargon
    - No greetings or sign-offs
    """

    try:
        response = call_llm(prompt)
        return jsonify({"response": response.strip()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ============ 8. Run the App ============================
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
