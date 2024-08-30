import streamlit as st
import tensorflow as tf 
import numpy as np
import json
import requests
import joblib
import base64
import pickle
import pandas as pd

# tensorflow model prediction

def model_prediction(test_image):
    model = tf.keras.models.load_model('final_trained_model.keras')
    image = tf.keras.preprocessing.image.load_img(test_image, target_size = (128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    return result_index

def model_suggestion(result_index):
    with open('outputs.json','r') as file:
        data = json.load(file)
        
    class_name = [
        'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
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
        'Tomato___healthy'
    ]
    predicted_class = class_name[result_index] 
    
    if predicted_class in data:
        class_info = data[predicted_class]
        return (
            f"Disease Name: {class_info['disease']}\n"
            f"Cause: {class_info['cause']}\n"
            f"Prevention: {class_info['prevention']}\n"
            f"Precautions: {class_info['precautions']}"
        )
    else:
        return f"Class '{predicted_class}' not found in the JSON file."
    
    

loaded_model = joblib.load('random_forest_model.pkl')
    
def predict_crop(N, P, K, temperature, humidity, ph, rainfall):
    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    prediction = loaded_model.predict(features)
    return prediction[0]

# ricrop setup
with open('crop_recommendation_model_extended.pkl', 'rb') as file:
    crop_recommendation_model_extended = pickle.load(file)

with open('yield_prediction_model_extended.pkl', 'rb') as file:
    yield_prediction_model_extended = pickle.load(file)

df_extended = pd.read_csv('extended_crop_data_30crops.csv')

def recommend_crop(soil_type, climate, region, soil_pH, avg_temp, rainfall, market_demand):
    prediction = crop_recommendation_model_extended.predict([[soil_type, climate, region, soil_pH, avg_temp, rainfall, market_demand]])
    return df_extended['recommended_crop'].cat.categories[prediction[0]]

def predict_yield(soil_type, climate, fertilizer_used, irrigation, soil_pH, avg_temp, rainfall, market_demand):
    prediction = yield_prediction_model_extended.predict([[soil_type, climate, fertilizer_used, irrigation, soil_pH, avg_temp, rainfall, market_demand]])
    return prediction[0]

# streamlit customization with markdown and css

import streamlit as st
import base64

# Caching the image loading function
@st.cache_data
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Load the image as base64
img = get_img_as_base64("finale.jpg")
img1 = get_img_as_base64("leaf.jpg")

# Create the CSS for the background image
page_bg_img = f"""
<style>
.stApp {{
    background-image: url("data:image/png;base64,{img}");
    background-size: 1440px;
    background-repeat: no-repeat;
    background-attachment: fixed;
    background-position: 250% center;
}}
[data-testid="stHeader"]{{
    background-color: rgba(0,0,0,0);
}}
[data-testid="stSidebarContent"]{{
    background-color: #246804;
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
    background-position: center;
}}
</style>
"""

# Inject the CSS into the Streamlit app
st.markdown(page_bg_img, unsafe_allow_html=True)


# sidebar

st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Our Services",["Home","About","Disease Recognition","Crop Recommendation","Weather","Real Time Market","Crop Yield"])

# Home Page
if app_mode == "Home" :
    st.title("FarmTech")
    image_path = "home.webp"
    st.image(image_path, width=300)
    st.markdown(""" 
    Welcome to FarmTech: Empowering Modern Agriculture

🌱 Your Digital Partner in Crop Management and Disease Prevention 🌱

Why FarmTech?
In today’s rapidly changing climate and market conditions, ensuring crop health and maximizing yields have become more challenging than ever. Farmers face constant threats from diseases, unpredictable weather, and fluctuating market prices. FarmTech is here to revolutionize how you manage your farm, combining the power of advanced AI with user-friendly interfaces to offer real-time solutions tailored to your needs.

Our Mission
At FarmTech, our goal is to empower farmers with cutting-edge technology that helps them make informed decisions, protect their crops, and boost their productivity. Whether it's identifying plant diseases, recommending the best crops for your soil, or staying updated with real-time weather and market prices, we’ve got you covered.

Services We Provide
🌾 Disease Recognition: Upload an image of your crop, and our AI model will accurately detect any diseases and provide actionable insights to prevent further damage.

🌱 Crop Recommendation: Get tailored crop suggestions based on your soil properties, climate, and local conditions to maximize your yields.

☁️ Weather Updates: Receive real-time weather updates to help you plan your farming activities efficiently and reduce risks due to unexpected weather changes.

📊 Real-Time Market Prices: Stay ahead of the market with live updates on crop prices from across the country, enabling you to make informed selling decisions.

Why You Need FarmTech

Reduce Crop Losses: Early detection of diseases can save up to 50% of your yield from potential losses.

Increase Profitability: Optimizing crop selection and staying updated on market trends can enhance your profit margins significantly.

Save Time and Resources: Automated insights and recommendations save you the hassle of manual research and guesswork.

Make Data-Driven Decisions: Leverage the power of AI and data analytics to drive every aspect of your farming operations with confidence.


Get Started Now!
Click on the Disease Recognition page to upload an image and let our system do the rest. Explore other features via the sidebar to unlock the full potential of your farming with FarmTech."""
    )
    
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

                """)

#Prediction Page
elif(app_mode=="Disease Recognition"):
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")
    if test_image and st.button("Show Image"):
        st.image(test_image,width=4,use_column_width=True)
        
    #Predict button
    if test_image and st.button("Predict"):
        
        st.write("Our Prediction")
        result_index = model_prediction(test_image)
        #Reading Labels
        class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
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
                      'Tomato___healthy']
        
        st.success(f"Model is Predicting it's a {class_name[result_index]}")
        description = model_suggestion(result_index)
        st.markdown(f"**Description on the Disease detected:**\n```\n{description}\n```")
        
        
        
#Recommendation Page
elif(app_mode=="Crop Recommendation"):
    st.header("Your Crop Recommendation")
    
    st.write("Enter the following parameters to get the recommended crop:")


    N = st.number_input('Nitrogen (N)', min_value=0, max_value=200, value=90)
    P = st.number_input('Phosphorus (P)', min_value=0, max_value=200, value=42)
    K = st.number_input('Potassium (K)', min_value=0, max_value=200, value=43)
    temperature = st.number_input('Temperature (°C)', min_value=0.0, max_value=50.0, value=25.0)
    humidity = st.number_input('Humidity (%)', min_value=0.0, max_value=100.0, value=80.0)
    ph = st.number_input('Soil pH', min_value=0.0, max_value=14.0, value=6.5)
    rainfall = st.number_input('Rainfall (mm)', min_value=0.0, max_value=500.0, value=200.0)


    if st.button('Recommend Crop'):
       crop = predict_crop(N, P, K, temperature, humidity, ph, rainfall)
       st.success(f'The recommended crop for the given conditions is: {crop}')



# Weather updates
elif(app_mode=="Weather"):
    st.header("Current Weather")
    def get_weather(city):
       api_key = '77592f46efff7cda527f30c73731cc84'
       url = f'http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}'
       response = requests.get(url)
       return response.json()
   
    city = st.text_input('Enter city name', '________')
    if st.button('Get Weather'):
        weather_data = get_weather(city)
        if weather_data.get('main'):
           st.write(f"Temperature: {round(weather_data['main']['temp']-273.15,2)}`C")
           st.write(f"Weather: {weather_data['weather'][0]['description']}")
        else:
           st.write('City not found')
        
        
elif(app_mode=="Real Time Market"):
    
    
    api_key = '579b464db66ec23bdd000001c98b328b805d43fc56845f942c90a114'

    # Example dataset ID (replace with your actual dataset ID)
    dataset_id = '9c4eeb10-b2a4-406c-9025-aeaa3011cc48'

    # Construct the API URL
    api_url = f'https://data.gov.in/resource/{dataset_id}?api-key={api_key}'

    #   Function to fetch data from the API
    def fetch_market_prices():
        response = requests.get(api_url)
        st.write(f"Status Code: {response.status_code}")
        st.write(f"Response Text: {response.text}")
        if response.status_code == 200:
            try:
               return response.json()
            except ValueError:
               st.error("Failed to parse JSON. Response content may not be in JSON format.")
               return None
        else:
            st.error(f"API request failed with status code {response.status_code}.")
            return None

    # Streamlit app
    st.title('Real-Time Crop Market Prices in India')

    # Fetch and display data
    market_data = fetch_market_prices()

    if market_data:
        st.write("Market Prices:")
        for item in market_data['records']:
            st.write(f"Crop: {item['crop_name']}, Price: {item['price']}, Market: {item['market_name']}")
    else:
        st.error("Failed to fetch market prices. Please check the API URL and key.")

    # api_key = '579b464db66ec23bdd000001c98b328b805d43fc56845f942c90a114'
    # dataset_id = '9c4eeb10-b2a4-406c-9025-aeaa3011cc48'
    # base_url = f"https://data.gov.in/resource/{dataset_id}"
       
    # params = {
    #     "api-key": api_key,
    #     "format": "json",  # You can specify the format as JSON
    #     "offset": 0,       # Starting point of records
    #     "limit": 10        # Number of records to fetch
    # }

    # response = requests.get(base_url, params=params)
    # if response.status_code == 200:
    #     # Parsing the JSON response
    #     data = response.json()
    #     print("Data fetched successfully:")
    #     print(data)
    # else:
    #     print(f"Failed to fetch data. HTTP Status code: {response.status_code}")
    
elif(app_mode == 'Crop Yield'):
    st.title('Crop Recommendation and Yield Prediction System')

    st.header('Input Parameters')
    soil_type = st.number_input("Enter the soil type (1, 2, 3):", min_value=1, max_value=3, step=1)
    climate = st.number_input("Enter the climate type (1, 2, 3):", min_value=1, max_value=3, step=1)
    region = st.number_input("Enter the region (1, 2):", min_value=1, max_value=2, step=1)
    fertilizer_used = st.number_input("Enter if fertilizer is used (1 for yes, 0 for no):", min_value=0, max_value=1, step=1)
    irrigation = st.number_input("Enter if irrigation is used (1 for yes, 0 for no):", min_value=0, max_value=1, step=1)
    soil_pH = st.number_input("Enter the pH of the soil:", min_value=0.0, max_value=14.0, step=0.1)
    avg_temp = st.number_input("Enter average temp (°C):", min_value=-50, max_value=50, step=1)
    rainfall = st.number_input("Enter rainfall (in mm):", min_value=0, max_value=10000, step=1)
    market_demand = 2  # You can also add input here if needed
    
    
    if st.button('Recommend Crop'):
        recommended_crop = recommend_crop(soil_type, climate, region, soil_pH, avg_temp, rainfall, market_demand)
        st.success(f"Recommended Crop based on your inputs: {recommended_crop}")

    if st.button('Predict Yield'):
        predicted_yield = predict_yield(soil_type, climate, fertilizer_used, irrigation, soil_pH, avg_temp, rainfall, market_demand)
        st.success(f"Predicted Yield: {predicted_yield:.2f} tons/hectare")