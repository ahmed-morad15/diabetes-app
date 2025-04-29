import streamlit as st
import requests
import time

API_URL = "https://web-production-54845.up.railway.app/"

st.set_page_config(
    page_title="Diabetic Prediction System",
    page_icon="🩺",
    initial_sidebar_state="collapsed"  
)

if "qa_history" not in st.session_state:
    st.session_state.qa_history = []

if "current_question" not in st.session_state:
    st.session_state.current_question = ""

if "diabetes_prediction" not in st.session_state:
    st.session_state.diabetes_prediction = None

if "advice" not in st.session_state:
    st.session_state.advice = ""

if "current_page" not in st.session_state:
    st.session_state.current_page = "prediction_page"

if "show_loading" not in st.session_state:
    st.session_state.show_loading = False

# ------------------- Prediction Page -------------------
def prediction_page():
    st.markdown("<h1 style='text-align:center;'>🧬 Diabetic Prediction System </h1>", unsafe_allow_html=True)
    with st.form("prediction_form"):
        st.subheader("📋 Enter Your Health Data") 
        name = st.text_input("Enter Name")
        gender = st.selectbox("Gender", ("Male", "Female"))
        age = st.number_input("Age", min_value=0.0, max_value=120.0, value=30.0, step=1.0)
        hypertension = st.selectbox("Hypertension", ("No", "Yes"))
        heart_disease = st.selectbox("Heart Disease", ("No", "Yes"))
        smoking_history = st.selectbox("Smoking History", ("never", "current", "formerly", "No Info", "ever", "not current"))
        bmi = st.number_input("BMI", min_value=10.0, max_value=100.0, value=25.0, step=0.1)
        HbA1c_level = st.number_input("HbA1c Level", min_value=3.5, max_value=9.0, value=5.5, step=0.1)
        blood_glucose_level = st.number_input("Blood Glucose Level", min_value=80, max_value=300, value=140, step=1)

        col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
        with col3:
            submitted = st.form_submit_button("🎯 Predict")
    if not submitted:
        st.markdown("""
        <div style='display: flex; justify-content: center; margin-top: 20px;'>
            <a href="https://ahmedshmees.github.io/Diabetes-prediction-app" target="_blank">
                <button style='
                    padding: 10px 20px; 
                    font-size: 18px; 
                    background-color: transparent; 
                    color: white; 
                    border: 2px solid gray; 
                    border-radius: 8px; 
                    cursor: pointer;
                '>
                    🏠 Go To Home Page
                </button>
            </a>
        </div>
    """, 
    unsafe_allow_html=True)

    if submitted:
        gender_numeric = 1 if gender == "Male" else 0
        hypertension_numeric = 1 if hypertension == "Yes" else 0
        heart_disease_numeric = 1 if heart_disease == "Yes" else 0
        smoking_history_numeric = {
            "never": 0,
            "current": 1,
            "formerly": 2,
            "No Info": 3,
            "ever": 4,
            "not current": 5
        }[smoking_history]

        payload = {
            "gender": gender_numeric,
            "age": age,
            "hypertension": hypertension_numeric,
            "heart_disease": heart_disease_numeric,
            "smoking_history": smoking_history_numeric,
            "bmi": bmi,
            "HbA1c_level": HbA1c_level,
            "blood_glucose_level": blood_glucose_level
        }

        try:
            response = requests.post(f"{API_URL}/predict", json=payload)
            result_json = response.json()

            if "diabetes_prediction" in result_json:
                prediction = result_json["diabetes_prediction"]
                result = "Diabetic" if prediction == 1 else "Non-Diabetic"
                name_prefix = "Mr." if gender == "Male" else "Ms."

                st.markdown(f"<h3 style='text-align:center;'>Medical Report for {name_prefix} {name}</h3>", unsafe_allow_html=True)
                st.markdown(
                    f"""
                    **Patient Name:** {name_prefix} {name}  
                    **Gender:** {gender}  
                    **Age:** {age}  
                    **Hypertension:** {hypertension}  
                    **Heart Disease:** {heart_disease}  
                    **Smoking History:** {smoking_history}  
                    **BMI:** {bmi}  
                    **HbA1c Level:** {HbA1c_level}  
                    **Blood Glucose Level:** {blood_glucose_level}  
                    """, unsafe_allow_html=True
                )

                st.markdown(f"<h3 style='text-align:center; color: {'red' if result == 'Diabetic' else 'green'};'>Prediction: {result}</h3>", unsafe_allow_html=True)
                message = "🩺 Take Care of your Health, Have a NICE Day" if result == "Diabetic" else "🎉 Congrats! You Seem Healthy, Have a NICE Day"
                st.markdown(f"<h4 style='text-align:center;'>{message}</h4>", unsafe_allow_html=True)

                if result == "Diabetic":
                    st.markdown(
                        """
                        <div style='text-align: center; background-color: #d4edda; padding: 15px; border-radius: 8px; border: 1px solid #c3e6cb; color: #155724;'>
                            <strong> You are diagnosed as Diabetic. Redirecting to the questions page...</strong>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    st.session_state.show_loading = True
                    st.session_state.current_page = "questions_page"
                    st.rerun()
                    
                else:
                    st.markdown(
                        """
                        <div style='text-align: center; background-color: #d4edda; padding: 15px; border-radius: 8px; border: 1px solid #c3e6cb; color: #155724;'>
                            <strong> No further action is required. You are healthy.</strong>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            else:
                st.error(" Failed to get prediction from server. Try again.")
        except Exception as e:
            st.error(f" Failed to connect to backend: {e }")

        st.markdown("""
        <div style='display: flex; justify-content: center; margin-top: 20px;'>
            <a href="https://ahmedshmees.github.io/Diabetes-prediction-app" target="_blank">
                <button style='
                    padding: 10px 20px; 
                    font-size: 18px; 
                    background-color: transparent; 
                    color: white; 
                    border: 2px solid gray; 
                    border-radius: 8px; 
                    cursor: pointer;
                '>
                    🏠 Go To Home Page
                </button>
            </a>
        </div>
    """, 
    unsafe_allow_html=True)

# ------------------- Questions Page -------------------
def questions_page():
    st.markdown("<h1 style='text-align:center;'> 🔍Follow-up Questions </h1>", unsafe_allow_html=True)
    
    if st.session_state.show_loading:
        st.markdown(
        """
        <div style="display: flex; justify-content: center; align-items: center; height: 200px; flex-direction: column;">
            <div style="border: 4px solid #f3f3f3; border-top: 4px solid #8B0000; border-radius: 50%; width: 40px; height: 40px; animation: spin 1s linear infinite;"></div>
            <div style="margin-top: 10px; font-size: 18px; color: #4B4B4B;">
                ⏳ Preparing questions... Please wait 5 seconds...
            </div>
        </div>
        <style>
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        </style>
        """,
        unsafe_allow_html=True
        )
        time.sleep(2)
        st.session_state.show_loading = False
        st.rerun()

    if st.session_state.current_question == "":
        try:
            res = requests.post(f"{API_URL}/next-question", json={})
            if res.status_code == 200:
                st.session_state.current_question = res.json().get("next_question")
            else:
                st.error("Failed to load the first question.")
        except Exception as e:
            st.error(f"Error connecting to backend: {e}")
            st.stop()
    
    st.markdown("<h4 style='text-align:center;'>🤖 Let’s Chat About Your Health!</h4>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align:center;'>💬 Here's a Question Tailored Just for You</h3>", unsafe_allow_html=True)

    st.markdown("""
    <div style="
        border: 1px solid #333333;
        padding: 15px;
        border-radius: 10px;
        margin-top: 10px;
        font-size: 17px;
        color: white;">
          <b>Question:</b> <br>
        {question}
    </div>
    """.format(question=st.session_state.current_question), unsafe_allow_html=True)

    user_answer = st.text_input("✍️ Your Answer:", key="answer_input", placeholder="Enter your Answer here...")

    col0,col10,col1, = st.columns([5,4,2])
    with col10:
        if st.button("Next", key="next_btn"):
            if user_answer.strip() == "":
                st.markdown('<span style="color:#4B4B4B; font-weight:bold;">❗answer</span>', unsafe_allow_html=True)
            else:
                st.session_state.qa_history.append({
                    "question": st.session_state.current_question,
                    "answer": user_answer
                })
                res = requests.post(f"{API_URL}/next-question", json={
                    "last_question": st.session_state.current_question,
                    "last_answer": user_answer
                })

                if res.status_code == 200:
                    st.session_state.current_question = res.json().get("next_question")
                    st.rerun()
                else:
                    st.error("Failed to generate next question.")

    if st.session_state.qa_history:
        with st.expander("📚 Question & Answer History"):
            for i, qa in enumerate(st.session_state.qa_history):
                st.markdown(f"**Q{i+1}:** {qa['question']}")
                st.markdown(f"**A{i+1}:** {qa['answer']}")

    col4, col5, col6 = st.columns([1.2,1.5,1.1])
    with col5:
        if st.button("📋 Analyze Answers and Get Advice", key="analyze_btn"):
            if len(st.session_state.qa_history) < 3:
                st.markdown('<span style="color:#4B4B4B; font-weight:bold;">❗❗ Please answer at least 3 questions.</span>', unsafe_allow_html=True)
            else:
                res = requests.post(f"{API_URL}/generate-advice", json={
                    "qa_history": st.session_state.qa_history
                })

                if res.status_code == 200:
                    st.session_state.advice = res.json().get("advice")
                    st.rerun()
                else:
                    st.error("An error occurred while analyzing the answers.")

    if st.session_state.advice:
        st.markdown("<h3 style='text-align:center;'>🩺 Personalized Advice Based on Your Case:</h3>", unsafe_allow_html=True)
        st.success(st.session_state.advice)

    col7, col8, col9 = st.columns([1,8, 1])
    with col8:
        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.markdown("""
                <div style='display: flex; justify-content: center; margin-top: 20px;'>
                    <form>
                        <button type="submit" style='
                            width: 100%;
                            padding: 10px 20px; 
                            font-size: 18px; 
                            background-color: transparent; 
                            color: white; 
                            border: 2px solid gray; 
                            border-radius: 8px; 
                            cursor: pointer;
                        ' formaction="#">
                            ↩️ Back to Prediction Page
                        </button>
                    </form>
                </div>
            """, unsafe_allow_html=True)

        with col_right:
            st.markdown("""
                <div style='display: flex; justify-content: center; margin-top: 20px;'>
                    <a href="https://ahmedshmees.github.io/Diabetes-prediction-app" target="_blank">
                        <button type="submit" style='
                            width: 100%;
                            padding: 10px 20px; 
                            font-size: 18px; 
                            background-color: transparent; 
                            color: white; 
                            border: 2px solid gray; 
                            border-radius: 8px; 
                            cursor: pointer;
                        '>
                            🏠 Go To Home Page
                        </button>
                    </a>
                </div>
            """, unsafe_allow_html=True)

# ----------------- Chatbot State Setup -----------------
if "chat_open" not in st.session_state:
    st.session_state.chat_open = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.markdown("""
    <style>
    .icon-button {
        position: fixed;
        bottom: 25px;
        right: 25px;
        background-color: #4CAF50;
        border: none;
        color: white;
        text-align: center;
        font-size: 30px;
        border-radius: 50%;
        width: 60px;
        height: 60px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.3);
        cursor: pointer;
        z-index: 9999;
    }
    </style>
""", unsafe_allow_html=True)

col1, col2 = st.columns([0.9, 0.1])
with col2:
    if st.button("💬", key="chat_toggle", help="Open chat", use_container_width=True):
        st.session_state.chat_open = not st.session_state.chat_open

# ----------------- Chatbot Page -----------------
def chatbot_page():
    if st.session_state.chat_open:
        st.markdown(
    """
    <h1 style='text-align: center; font-size: 48px;'>🤖 Chat with HealthBot</h1>
    <p style='text-align: center; font-size: 20px;'>Your smart assistant for personalized diabetes care</p>
    """,
    unsafe_allow_html=True
)
        st.markdown("""
    <style>
.chat-box {
    
    border: 2px solid #ccc;
    max-height: 10000px;
    overflow-y: auto;
    font-size: 18px;
    background-color: transparent; 
    margin: 0 auto;
    max-width: 650px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.1);
}
.chat-message {
    margin-bottom: 15px;
    padding: 10px;
    border-radius: 10px;
    background-color: rgba(255, 255, 255, 0.0); 
}
.user-msg {
    color: #1a73e8;
    font-weight: bold;
    border-left: 4px solid #1a73e8;
    padding-left: 10px;
    background-color: rgba(26, 115, 232, 0.05); 
}
.bot-msg {
    color: #2e7d32;
    font-weight: bold;
    border-left: 4px solid #2e7d32;
    padding-left: 10px;
    background-color: rgba(46, 125, 50, 0.05); /* خلفية خفيفة شفافة */
}
</style>
""", unsafe_allow_html=True)

    chat_box = st.container()
    with chat_box:
        st.markdown('<div class="chat-box">', unsafe_allow_html=True)

        for entry in st.session_state.chat_history:
            st.markdown(f'<div class="chat-message"><span class="user-msg">🧑 You:</span> {entry["user"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="chat-message"><span class="bot-msg">🤖 Bot:</span> {entry["bot"]}</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)
        
        user_input = st.text_input("💬 Type your message", key="chat_input")

        col1, col2, col3,col4, col5 = st.columns([1, 1,1,1, 1]) 
        with col3:
            if st.button("📨 Send", key="send_msg_btn"):
                if user_input.strip():
                    try:
                        res = requests.post(f"{API_URL}/chat", json={"message": user_input})
                        if res.status_code == 200:
                            bot_reply = res.json().get("response", "Sorry, I didn't understand.")
                        else:
                            bot_reply = "Server error."
                    except:
                        bot_reply = "Connection error."

                    st.session_state.chat_history.append({
                        "user": user_input,
                        "bot": bot_reply
                    })
                    st.rerun()
                    
# ------------------- Main App Logic -------------------
if st.session_state.chat_open:
    chatbot_page()
elif st.session_state.current_page == "prediction_page":
    prediction_page()
elif st.session_state.current_page == "questions_page":
    questions_page()
