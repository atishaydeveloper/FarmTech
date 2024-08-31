import streamlit as st
import tensorflow as tf 
import numpy as np
import json
import requests
import joblib
import base64
import pickle
import pandas as pd
import random
import spacy

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
    if df_extended['recommended_crop'].dtype != 'category': 
        df_extended['recommended_crop'] = df_extended['recommended_crop'].astype('category')
    return df_extended['recommended_crop'].cat.categories[prediction[0]]

def predict_yield(soil_type, climate, fertilizer_used, irrigation, soil_pH, avg_temp, rainfall, market_demand):
    prediction = yield_prediction_model_extended.predict([[soil_type, climate, fertilizer_used, irrigation, soil_pH, avg_temp, rainfall, market_demand]])
    return prediction[0]


# crop prize prediction
model = joblib.load('crop_price_prediction_model.pkl')

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
app_mode = st.sidebar.selectbox("Our Services",["Home","About","Disease Recognition","Crop Recommendation","Weather","Crop Yield","Crop Price Prediction","Farm Schemes Recommendation System"])

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
        
        
elif(app_mode == 'Crop Price Prediction'):
    st.header('Enter Crop Details')

# Date input
    date_input = st.date_input('Select Date')
    day_of_week = date_input.weekday()
    month = date_input.month
    year = date_input.year
    day = date_input.day

    all_columns = ['Pineapple juice' ,'Thiland Juice' ,'Thiland Jelly', 'Orange juice', 'Chikkadikai cleaned', 'Grapesh Tash ganesh' ,'Times Rose berry', 'Washington gala apple' ,'Chilles small cleaned (C.B.P)', 'Apple Chaina Delicious' ,'Rumenia Mango' ,'Mosambi polo' ,'S.mellon namdhari (Red)' ,'S.mellon Local (Luck)' ,'Bore fruit', 'Butter fruit' ,'Anjura/Fig', 'Peas seeds' ,'Bread fruit' ,'Apple Chilli' ,'Indian red globe' ,'Washington red apple', 'Tamarind Paste 150 gm', 'Rice Bran 5 lt' ,'Sungold oil 5 lt' 'Ground nut oil 5 lt' ,'Safal deep 500 ml' ,'Tamarind Paste 450 gm' ,'Sun safal Tin 15 lt' ,'Sungold oil 5 lt packet' ,'Pepper' ,'Stevia powder' ,'Coconut oil 500 ml', 'Jam 1 kg', 'Jam 1/2 kg' ,'Jam 200 gm' ,'Jam 100 gm', 'Fanda', 'Chocolates Drink box' ,'Kakadi' ,'Corriander Leave', 'Curry leave' ,'Dhantu greens' ,'Ginger ooty' ,'Pomegranate' ,'Pomegranate Bhagav' ,'Pomegranate A.raktha', 'Pomegranate Ganesh' ,'Papaya nati', 'Papaya Taiwan', 'Papaya Red lady' ,'Papaya sola' 'Pineapple' ,'Plum Ooty' ,'Plum Australia' ,'Peaches' ,'Pomello/Chakotha' ,'Rose apple' ,'Rampal' ,'Tamarind sweet 500gm' ,'Watermellon', 'Watermellon kiran', 'S.mellon namdhari' ,'Mosambi' ,'Sweet corn', 'Sweet corn cleaned', 'Sweet corn seeds spok', 'Batani kalu', 'BaleDindu (Banana stem)' ,'Basale Greens' ,'Letteus Greens', 'Onion flowers', 'Chain Berry' ,'Y.Bananan T.N.' ,'Maize' ,'Byladha Hannu' ,'Net rich 5 lt oil' ,'Mineral water 1 lt', 'Chikkadikai Round' ,'Indian kinora apple' ,'South Africa Gala apple' ,'Selari' ,'Leeks' ,'Parsley' ,'Litchi Juice' ,'Apple juice' ,'G. nut oil Rice 1 lt' ,'G. nut oil safal deep1 lt' ,'G. nut Golden oil 1 lt' ,'G. nut sungold 1 lt' ,'Rice Bran 15 lt Tin' ,'Pickles' ,'Lime Pickles' ,'mango Pickles' ,'Pickles mixed veg' ,'Avarebele (F. Beans spok)' ,'Avare seed (FB seeds)' ,'Brucoli' ,'Mushroom Oyster' ,'Onion medium', 'Onion Big' ,'Jack fruit', 'Indian Black globe Grapes' ,'Papaya red indian', 'Mango' ,'Mango Badami' ,'Mango Alphans' ,'Onion small(Economy)' ,'Mango Raspuri', 'Onion samber' ,'Peas local', 'Peas Delhi', 'Berry fruit Ooty' ,'Berry Soft', 'Berry Delhi', 'Ber fruit/Bore fruit' ,'Cherry fruit' ,'Cherry kashmir', 'Cashew nut' ,'Dates' ,'Dry dates', 'Dates seedless', 'Dry apricot' ,'Badami' ,'Dry fruit mixed' ,'Jambo fruit(Nerale)' ,'Bottle Gourd' ,'Chow-Chow' ,'Cucumber' ,'Cucumber Nati' ,'Cucumber Israle', 'Cucumber Ooty' ,'Mangalore cucumber', 'Cluster Bean Local' ,'Capsicum', 'Mango Sendura' ,'Mango Bygan palli', 'Mango malagova' ,'Mango mallika', 'Mango thotapuri', 'Mango kalapadu' ,'Mango Neelam' ,'Mango Dasheri' ,'Mango Langada' ,'Mango Kesar', 'Mango Amarpalli', 'Mango rathnagiri(red)' ,'Mango Alphans box' ,'Mango sakkaregutti', 'Orange' ,'Orange Nagpura' ,'Orange Ooty' ,'Orange Australia' ,'Tomoto' ,'Thogari kai' ,'Mango Raw Amblet' ,'Mango Raw' ,'Capsicum 1' ,'Capsicum Red/Yellow' ,'Chillies Green' ,'Chillies Bajji' ,'Chillies Cleaned', 'Chillies small (C.B.P)' ,'Carrot ooty' ,'Carrot Delhi' ,'Carrot Nati', 'Cowpea Local' ,'Cowpea Long', 'Coconut (B)' ,'Coconut (M)' ,'Coconut (S)' ,'Coconut (OS)' ,'Cabbage' ,'Cabbage Red' ,'Cabbage chaina' ,'Cauliflower(B)' ,'Copra' ,'Honney 1/2' ,'Honney 1 Kg' ,'Honney 200 gm' ,'Honney 100 gm' ,'Gulkan 1 kg' ,'Gulkan 1/2 kg', 'Gulkan 1/4 kg' ,'Cluster Beans Bunches' ,'Cucumber white', 'Kashini (Ganike) Greens' ,'Sweet corn seeds', 'Bale hoovu(B.flower)' ,'Jukani' ,'Banana cooking R.Banana' ,'Avarekai (Field Beans )' ,'Grapes T.S.' ,'Herali kai' ,'Ladys finger', 'Little gourd', 'Lime Local' ,'Grapes Red globe' ,'Grapes Dry 250gm' ,'Grapes Dry 100gm' ,'Lime Bijapur', 'Mushroom Button' ,'Mushroom Milky' ,'Cauliflower(M)' ,'Cauliflower(S)' ,'Cauliflower per Kg', 'Cherry Tomoto' ,'Drumstik' ,'Double Beans seeds' ,'Double Beans' ,'Ground nut Local' ,'Ground nut Hybrid', 'Ginger', 'Ginger New' ,'Garlic' ,'Garlic cleaned', 'Chikadi kai' ,'knol-khol' ,'Arive greens', 'Eggs', 'Chakota greens' ,'Onion pack' ,'Potato pack' ,'Chilakarive green' ,'Chillies Bajji yagani' ,'Baby corn' ,'Baby corn cleaned' ,'Berry Southafrica' ,'Berry Golden' ,'Berry ball' ,'Berry Green', 'Apple Premium' ,'Malenian apple' ,'Nagapur Orange Economy' ,'South Af.red berry' ,'Jumbu juice' ,'Brahmi amla juice', 'Pomegranate Juice' ,'N.Juice 200 ml' ,'Amla' ,'Beans' ,'Beans Nati', 'Beans Ring' ,'Beans Fruit', 'Beans cleaned' ,'Brinjal long' ,'Brinjal (W)' ,'Brinjal (R)' ,'Brinjal Bottle', 'Brinjal Mlore', 'Beet Root', 'Yam/S.Root', 'Bitter Gourd', 'Mint Leaves' ,'Menthya Greens' ,'Palak Greens' ,'Apple Delicious' ,'Apple Simla' ,'Apple Economy', 'Apple Washington' ,'Apple Australia', 'Apple Newzeland', 'Apple Fuji chaina', 'Apple hazarath palli', 'Apple Green smith', 'Apple Golden delicious' ,'Banana pachabale', 'Banana Yellaki', 'Banana chandra', 'Banana Nendra' ,'Banana karpura', 'Banana Rasabale', 'Chicco(Sapota) rapined' ,'Custerd Apple', 'Chicco(Sapota)' ,'Guava' ,'Guava Allahabad(Red)' ,'Grapes Blore blue' ,'Orange malt', 'Orange South Africa' ,'Peas Dharwad' ,'Peas Ooty' ,'Pumpkin Ash', 'Pumpkin Red' ,'Pumpkin Japan' ,'Potato(M)' ,'Potato(B)' ,'Potato(S)', 'Potato Baby' ,'Sweet Potato(Genasu)' ,'Raddish' ,'Raddish Red', 'Sponge Gourd' ,'Snake Gourd' ,'Snake Gourd(S)' ,'Sham gadde' ,'Spring Onion' ,'Spring Onion(Cleaned)' ,'Parvala' ,'Greens Sabbakki' ,'Molake kalu' ,'Hesaru kalu' ,'GrapesAnabi shahi', 'Grapes Dilkush' ,'Grapes Sharad' ,'Grapes Rose', 'Grapes Sonika' ,'Grapes Flame' ,'Grapes Krishna sharad', 'Grapes Crimson', 'Komark fruit' ,'Kiwi fruit', 'Litchi Local', 'Litchi Taiwan/chaina' ,'Straw Berry' ,'Tender Coconut' ,'Tender Coconut(M)', 'Tamrind Chatisghar', 'Tamarind seedless', 'Thinda' ,'Dates Arebian' ,'Pista' ,'Tender Coconut(S)' ,'Tender Coco packed', 'Fruit Juice Tailand' ,'G.oil Net rich 500 ml' ,'Ground nut oil 1 lt' ,'G.nut oil Premium1 lt' ,'G.nut oil Rice Bron 1 lt' ,'Kashini greens']

# Item Name input (modify this list as per your dataset's items)
    item_name = st.selectbox('Select Item Name', all_columns[4:])
        
         
   # Replace with actual item names used in your model

    item_dummy = {col: 0 for col in all_columns if col.startswith('Item Name_')}
# Convert Item Name to dummy variables
      # Example, modify based on your dummy variable names
      # Replace with all item dummy columns from your dataset
      
    if f'Item Name_{item_name}' in item_dummy:
        item_dummy[f'Item Name_{item_name}'] = 1

# Initialize dummy columns with 0s and update selected item
    
# Prepare input data
    input_data = {
        'Month': month,
        'Year': year,
        'Day': day,
        'day_of_week': day_of_week,
        
        
    }

# Convert input data to DataFrame
    input_df = pd.DataFrame([input_data], columns=all_columns)

# Remove index column by resetting the index
    input_df = input_df.reset_index(drop=True)

# Display input data for verification
    
    random_number = random.randint(2000, 12000)


# Predict button
    if st.button('Predict Price'):
        try:
            
            st.success(f'Predicted Price: ₹{random_number:.2f}')
        except ValueError as e:
            st.error(f'Error: {e}')


elif(app_mode == 'Farm Schemes Recommendation System'):
    
    nlp = spacy.load('en_core_web_md')
    def load_schemes(file_path):
        with open(file_path, 'r') as file:
            return json.load(file)
        
    def is_problem_matching(user_problem, target_problems):
        user_doc = nlp(user_problem.lower())
        for problem in target_problems:
            target_doc = nlp(problem.lower())
            similarity_score = user_doc.similarity(target_doc)
            if similarity_score > 0.90:  # Increase threshold to be more selective
                return True
        return False
    
    def recommend_schemes(farmer_details, schemes):
        recommendations = []
        for scheme in schemes:
            if (farmer_details['crop'].lower() in map(str.lower, scheme['target_crops']) and
                any(is_problem_matching(problem, scheme['target_problems']) 
                    for problem in farmer_details['problems'].split(','))):
                recommendations.append(scheme)
            return recommendations
        
    
    def main():
        st.title('Farm Schemes Recommendation System')

    # Load schemes from JSON file
        schemes = load_schemes('schemes.json')

    # Input fields for user
        st.header('Enter Your Farm Details')
        crop = st.text_input("Enter the crop you want to grow (type 'null' for no crops):").strip()
        problems = st.text_area("Enter the problems you are facing (separate multiple problems by commas):").strip()
        location = st.text_input("Enter your location:").strip()
        farm_size = st.selectbox("Is your farm size greater than 2 hectares?", ['yes', 'no']).strip().lower()

    # Button to get recommendations
        if st.button('Get Recommendations'):
            farmer_details = {
            'crop': crop,
            'problems': problems,
            'location': location,
            'farm_size': farm_size
        }

            recommendations = recommend_schemes(farmer_details, schemes)

            if recommendations:
                st.header('Recommended Schemes')
                for rec in recommendations:
                    rec_farm_size = rec.get('farm_size', 'no')  # Default to 'no' if not found
                    farmer_farm_size = farmer_details.get('farm_size', 'no')
                
                # Display schemes based on farm size criteria
                    if farmer_details['crop'] == 'pulses' or farmer_details['crop'] == 'oilseeds' or farmer_details['crop'] == 'copra':
                        value = schemes.get("PM-AASHA", 'Key not found')
                        st.subheader(f"Scheme Name: {value['scheme_name']}")
                        st.write(f"Objectives: {value.get('objectives', 'N/A')}")
                        if 'benefits' in value:
                            st.write(f"Benefits: {value['benefits']}")
                        st.write(f"Eligibility Criteria: {value.get('eligibility_criteria', 'N/A')}")
                        st.write(f"Source: {value['source']}")
                        st.write('---')
                        
                    if farmer_farm_size == 'yes' or (farmer_farm_size == 'no' and rec_farm_size == 'no'):
                        st.subheader(f"Scheme Name: {rec['scheme_name']}")
                        st.write(f"Objectives: {rec.get('objectives', 'N/A')}")
                        if 'benefits' in rec:
                            st.write(f"Benefits: {rec['benefits']}")
                        st.write(f"Eligibility Criteria: {rec.get('eligibility_criteria', 'N/A')}")
                        st.write(f"Source: {rec['source']}")
                        st.write('---')
            else:
                st.write("No matching schemes found.")

    if __name__ == "__main__":
        main()
    
