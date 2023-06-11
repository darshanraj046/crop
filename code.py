import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import warnings


def load_model(modelfile):
    return pickle.load(open(modelfile, 'rb'))


def main():
    html_temp = """
    <div>
    <h1 style="color:MEDIUMSEAGREEN;text-align:center;"> Crop Recommendation  🌱 </h1><br>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    col1, col2 = st.columns([2, 2])

    with col1:
        with st.expander(" ℹ️ Information", expanded=True):
            st.write("""
            Crop recommendation is one of the most important aspects of precision agriculture. Crop recommendations are based on a number of factors. Precision agriculture seeks to define these criteria on a site-by-site basis in order to address crop selection issues. While the "site-specific" methodology has improved performance, there is still a need to monitor the systems' outcomes.Precision agriculture systems aren't all created equal. 
            However, in agriculture, it is critical that the recommendations made are correct and precise, as errors can result in significant material and capital loss.

            """)
        '''
        ## How does it work ❓ 
        Complete all the parameters and the machine learning model will predict the most suitable crops to grow in a particular farm based on various parameters
        '''

    with col2:
        st.subheader(" Find out the most suitable crop to grow in your farm 👨‍🌾")
        N = st.number_input("Nitrogen", 0, 250)
        P = st.number_input("Phosporus", 0, 250)
        K = st.number_input("Potassium", 0, 250)
        temp = st.number_input("Temperature", 0.0, 100.0)
        humidity = st.number_input("Humidity in %", 0.0, 250.0)
        rainfall = st.number_input("Rainfall in mm", 0.0, 300.0)
        season = st.selectbox("Season", ["autumn", "kharif", "rabi", "summer", "whole year", "winter"])
        area = st.number_input("Area", 0)
        production = st.number_input("Production", 0)

        feature_list = [N, P, K, temp, humidity, rainfall]
        single_pred = np.array(feature_list).reshape(1, -1)
        loaded_model = load_model('lgb_model.pkl')
        prediction = loaded_model.predict(single_pred)
        #print(prediction.item())

        crop_encoded = np.array([1 if c == (prediction.item()) else 0 for c in ['rice', 'maize', 'chickpea', 'kidneybeans', 'pigeonpeas', 'mothbeans', 'mungbean', 'blackgram', 'lentil', 'pomegranate', 'banana', 'mango', 'grapes', 'watermelon', 'muskmelon', 'apple', 'orange', 'papaya', 'coconut', 'cotton', 'jute', 'coffee']])
        #print(crop_encoded)

        X_train = ['rainfall', 'area', 'production', 'crop_apple', 'crop_banana',
       'crop_blackgram', 'crop_chickpea', 'crop_coconut', 'crop_coffee',
       'crop_cotton', 'crop_grapes', 'crop_jute', 'crop_kidneybeans',
       'crop_lentil', 'crop_maize', 'crop_mango', 'crop_mothbeans',
       'crop_mungbean', 'crop_muskmelon', 'crop_orange', 'crop_papaya',
       'crop_pigeonpeas', 'crop_pomegranate', 'crop_rice', 'crop_watermelon',
       'season_autumn', 'season_kharif', 'season_rabi', 'season_summer',
       'season_whole year', 'season_winter']

        new_data = pd.DataFrame({'rainfall': [rainfall],
                                 'crop': [prediction.item()],
                                 'season': [season],
                                 'area': [area],
                                 'production': [production]})

        new_data_encoded = pd.get_dummies(new_data)
        new_data_encoded = new_data_encoded.reindex(columns=X_train, fill_value=0)

        a = prediction.item().title()

        if st.button('Predict'):
            loaded_model1 = load_model('yield_prediction_model.sav')
            prediction1 = loaded_model1.predict(new_data_encoded)
            col1.write('''
		    ## Results 🔍 
		    ''')
            col1.success(a + " are recommended for your farm with yield of {:.3f}.".format(prediction1[0]))


hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
