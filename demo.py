import streamlit as st
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np
import time as t
import joblib






def load_cow_data():
    return pd.read_csv('cow_beef.csv')

def save_model(model):
    joblib.dump(model, 'model.joblib')

def load_model():
    return joblib.load('model.joblib')

def page_one():
    global X,y
    st.title("Cow beef")
    st.caption("This is a cow price prediction machine learning system.")
    col1, col2 = st.columns(2)
    option = col2.selectbox(
        'How would you like to be contacted?',
        ('Begin!', ' Load your Dataset ', 'Training', "click this before predict"))

    if option == 'Let start!!!':
        t0 = int(t.time())
        with st.spinner('generating. . .'):
            t1 = int(t.time())
            t.sleep(1 + t1 - t0)
            data = pd.read_csv('cow_beef.csv')
            data1 = pd.DataFrame(data)
            X = data1.drop(columns='price', axis=1)
            y = data1['price']
            data.to_excel('cow_beef.xlsx')
        col2.success("Generating Complete. . .")

    if option == ' Load your Dataset ':  # Load model fucntions
        t0 = int(t.time())
        with st.spinner('Loading. . .'):
            t1 = int(t.time())
            t.sleep(1 + t1 - t0)
            df = pd.read_csv('cow_beef.csv', index_col=0)
            pd.DataFrame(df)
            st.write(df.head())
        col2.success("Loading complete . . .")

    if option == 'Training!':  # training model
        with st.spinner('Training. . .'):
            t.sleep(5)
        col2.success("Training Complete . . .")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        model = LogisticRegression()
        model.fit(X_train, y_train)
        df = pd.read_csv('cow_beef.csv', index_col=0)
        pd.DataFrame(df)
        st.write(df.head())
        save_model(model)
    if option == 'go!!!':
        pass

    breed = 0
    a_breed =col1.radio(
        "What\'s your breed?",
        ('Ready for Breeding (non-Wagyu)', 'Ready for Breeding (Wagyu)','Pregnant Cow (non-Wagyu)','Pregnant Cow (Wagyu)','steer (non-Wagyu)','steer  (Wagyu)'))
    if a_breed == 'Ready for Breeding (Wagyu)': breed = 1
    if a_breed == 'Pregnant Cow (non-Wagyu)': breed = 2
    if a_breed == 'Pregnant Cow (Wagyu)': breed = 3
    if a_breed == 'steer (non-Wagyu)': breed = 4
    if a_breed == 'steer  (Wagyu)': breed = 5

    age = col1.slider("Input your age", 0, 8)

    weight = col1.slider("Input your weight", 1000, 1600)
    sex = 0
    genre_sex = col1.radio(
        "What\'s your sex?",
        ('Female', 'Male'))
    if genre_sex == 'Male': sex = 1

    Pred = col1.button('Press here to Prediction')
    if Pred:
        model = load_model()
        Pred_data = (breed, sex, weight, age)
        Pred_data_array = np.asarray(Pred_data)
        Pred_data_array_re = Pred_data_array.reshape(1, -1)
        predict_Price = model.predict(Pred_data_array_re)
        result = predict_Price[0]
        st.success(f"The predicted price is: {result}")




