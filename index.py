import streamlit as st
import tensorflow as tf
import numpy as np
import google.generativeai as genai
from chatbot import predictDisease, chatWithGemini

# GOOGLE_API_KEY = "AIzaSyByxml_-GWvg9m-b62aCTxw4QGv2pMyYUU"
# genai.configure(api_key=GOOGLE_API_KEY)
# chat_model = genai.GenerativeModel("models/gemini-1.5-flash")


#Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to batch
    predictions = model.predict(input_arr)
    result_index = np.argmax(predictions)  # Get the index of the predicted disease
    class_names = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                    'Tomato___healthy']  # Your class names list
    
    # Get the disease name from the class_names list using the result index
    predicted_disease_name = class_names[result_index]

    return predicted_disease_name


#Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","About","Disease Recognition","ChatBot"])

#Main Page
if(app_mode=="Home"):
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "home_page.jpeg"
    st.image(image_path,use_column_width=True)
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍
    
    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.
    4. **ChatBot:** Interact with the chatbot by asking questions related to the disease for better understanding.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    """)

#About Project
elif(app_mode=="About"):
    st.header("About")
    st.markdown("""
                #### About Dataset
                This dataset is recreated using offline augmentation from the original dataset.The original dataset can be found on this github repo.
                This dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 38 different classes.The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure.
                A new directory containing 33 test images is created later for prediction purpose.
                #### Content
                1. train (70295 images)
                2. test (33 images)
                3. validation (17572 images)
                ### Class Labels
                1.  Apple___Apple_scab
                2.  Apple___Black_rot
                3.  Apple___Cedar_apple_rust
                4.  Apple___healthy
                5.  Blueberry___healthy
                6.  Cherry_(including_sour)___Powdery_mildew 
                7.  Cherry_(including_sour)___healthy
                8.  Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot 
                9.  Corn_(maize)___Common_rust_
                10. Corn_(maize)___Northern_Leaf_Blight
                11. Corn_(maize)___healthy', 
                12. Grape___Black_rot
                13. Grape___Esca_(Black_Measles)
                14. Grape___Leaf_blight_(Isariopsis_Leaf_Spot) 
                15. Grape___healthy
                16. Orange___Haunglongbing_(Citrus_greening)
                17. Peach___Bacterial_spot
                18. Peach___healthy
                19. Pepper,_bell___Bacterial_spot
                20. Pepper,_bell___healthy 
                21. Potato___Early_blight
                22. Potato___Late_blight
                23. Potato___healthy 
                24. Raspberry___healthy
                25. Soybean___healthy
                26. Squash___Powdery_mildew
                27. Strawberry___Leaf_scorch
                28. Strawberry___healthy
                29. Tomato___Bacterial_spot
                30. Tomato___Early_blight
                31. Tomato___Late_blight
                32. Tomato___Leaf_Mold 
                33. Tomato___Septoria_leaf_spot
                34. Tomato___Spider_mites Two-spotted_spider_mite 
                35. Tomato___Target_Spot
                36. Tomato___Tomato_Yellow_Leaf_Curl_Virus
                37. Tomato___Tomato_mosaic_virus
                38. Tomato___healthy
                """)

#Prediction Page
elif(app_mode=="Disease Recognition"):
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")
    if(st.button("Show Image")):
        st.image(test_image,width=4,use_column_width=True)
    #Predict button
    if(st.button("Predict")):
        st.snow()
        st.write("Our Prediction")
        disease = model_prediction(test_image)
        st.success("Model is Predicting it's a {}".format(disease))
        st.session_state.predicted_disease = disease

elif app_mode == "ChatBot":
    st.header("🤖 Plant Disease Diagnosis ChatBot")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "predicted_disease" not in st.session_state or not st.session_state.predicted_disease:
        st.warning("⚠️ Please go to the **Disease Recognition** page and upload an image first.")
        st.stop()

    st.success(f"💡 Chatting about: **{st.session_state.predicted_disease}**")

    # Chat input
    user_input = st.chat_input("Ask something about the disease...")

    if user_input:
        # Add user's message
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        # Build prompt from chat history
        prompt_history = [
            f"User: {m['content']}" if m["role"] == "user" else f"Assistant: {m['content']}"
            for m in st.session_state.chat_history
        ]

        if len([msg for msg in st.session_state.chat_history if msg["role"] == "user"]) == 1:
            intro = f"You are a plant pathologist AI. The diagnosed plant disease is '{st.session_state.predicted_disease}'."
            prompt_history.insert(0, intro)

        with st.spinner("🤖 Thinking..."):
            response = chatWithGemini(user_input, prompt_history)

        st.session_state.chat_history.append({"role": "assistant", "content": response})

    for msg in st.session_state.chat_history:
        with st.chat_message("user" if msg["role"] == "user" else "assistant", avatar="👤" if msg["role"] == "user" else "🤖"):
            st.markdown(msg["content"])
