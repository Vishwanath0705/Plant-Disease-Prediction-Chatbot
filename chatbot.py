import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import google.generativeai as genai


GOOGLE_API_KEY = "AIzaSyByxml_-GWvg9m-b62aCTxw4QGv2pMyYUU"
genai.configure(api_key=GOOGLE_API_KEY)
chat_model = genai.GenerativeModel("models/gemini-1.5-flash")

cnn = tf.keras.models.load_model("trained_plant_disease_model.keras")

validation_set = tf.keras.utils.image_dataset_from_directory(
    'New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/valid',
    labels="inferred",
    label_mode="categorical",
    image_size=(128, 128),
    batch_size=32
)
class_names = validation_set.class_names

def predictDisease(image_path):
    image = tf.keras.preprocessing.image.load_img(image_path,target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])

    predictions = cnn.predict(input_arr)
    result_index = np.argmax(predictions)
    predicted_class = class_names[result_index]

    return predicted_class

def chatWithGemini(user_input, history):
    prompt = "\n".join(history + [f"User: {user_input}", "Assistant:"])
    response = chat_model.generate_content(prompt)
    return response.text.strip()


def main():
    print("Plant Disease Diagnosis Chatbot!!")
    image_path = input("Enter image path of plant leaf: ").strip()

    try:
        predictedDisease = predictDisease(image_path)
        print(f"Predicted Disease : {predictedDisease}")
    except exeption as e:
        print(f"Error : {e}")
        return

    chat_history = [
        f"You are a plant pathologist AI. The model has predicted the plant disease as '{predictedDisease}'.",
        f"Provide an overview, symptoms, and possible treatments for {predictedDisease}."
    ]

    ai_intro = chat_model.generate_content("\n".join(chat_history)).text.strip()
    print("AI : ",ai_intro)
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Session ended.")
            break
        response = chatWithGemini(user_input, chat_history)
        chat_history.append(f"User: {user_input}")
        chat_history.append(f"Assistant: {response}")
        print("AI:", response)

if __name__ == "__main__":
    main()