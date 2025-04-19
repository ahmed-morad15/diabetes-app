from flask import Flask, request, jsonify
import joblib
import numpy as np
import requests
from sklearn.preprocessing import StandardScaler
app = Flask(__name__)

# ============ 1. ML DIABETES PREDICTION =================
model_path = "C:/diabetes_prediction_project/env/app/Diabetes_Prediction_Model.pkl"  
try:
    model = joblib.load(model_path)  
except Exception as e:
    print(f"Error loading model: {e}")
    model = None  
def predict_diabetes(input_data):
    if model is None:
        return None  
    try:
        features = np.array([input_data[key] for key in [
            "gender", "age", "hypertension", "heart_disease",
            "smoking_history", "bmi", "HbA1c_level", "blood_glucose_level"
        ]], dtype=np.float32).reshape(1, -1)
        prediction = model.predict(features)
        return int(prediction[0])
    except Exception as e:
        return str(e)  
    
# ============ 2. LLM CONFIG (OpenRouter API) ============
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
        return jsonify({"diabetes_prediction": result})  # 1 = Diabetic, 0 = Healthy
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ============ 4. ENDPOINT: Next Question ================
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
        """
    else:
        prompt = f"""
        You are an intelligent medical assistant. Your task is to follow up with a diabetic patient by asking short, specific questions in English.

        The previous question was: "{last_q}"
        The patient's answer was: "{last_a}"

        Now, please ask a new short and relevant question related *only* to diabetes to accurately monitor the patient's condition.
        Avoid general questions, and focus on medical aspects such as: nutrition, physical activity, blood sugar monitoring, medication adherence, and lifestyle.
        """

    try:
        generated_question = call_llm(prompt)
        return jsonify({"next_question": generated_question.strip()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============ 5. ENDPOINT: Generate Personalized Advice ==
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
    
# ============ 6. ENDPOINT: ChatBot Message ===============

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

# ============ 7. Run the App ============================
if __name__ == '__main__':
    app.run(debug=True)
