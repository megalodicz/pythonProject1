import streamlit as st
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib
import numpy as np
import time as t

def load_cow_data():
    return pd.read_csv('cow_beef.csv')

def save_model(model):
    joblib.dump(model, 'model.joblib')

def load_model():
    return joblib.load('model.joblib')

def page_one():
    global t0,X,Y
    st.title("Cow beef")
    st.caption("This is a cow price prediction machine learning system.")
    col1, col2 = st.columns(2)
    option = col1.selectbox(
        'How would you like to be contacted?',
        ('Let start!!!', '👀 Load your Dataset 👀', '👾Training!👾', "go!!!"))
    if option == 'Let start!!!':  # Create model fuction
        t0 = int(t.time())
        with st.spinner('generating. . .'):
            t1 = int(t.time())
            t.sleep(1 + t1 - t0)
            data = pd.read_csv('cow_beef.csv')
            data = pd.DataFrame(data)
            X = data.drop(columns='price', axis=1)
            Y = data['price']
            data.to_csv('cow_beef.csv')
        col1.success("Generating Complete. . .")
    if option == '👀 Load your Dataset 👀':  # Load model fucntions
        t0 = int(t.time())
        with st.spinner('Loading. . .'):
            t1 = int(t.time())
            t.sleep(1 + t1 - t0)
            df = pd.read_csv('cow_beef.csv', index_col=0)
            pd.DataFrame(df)
            st.write(df.head())
        col1.success("Loading complete . . .")
    if option == '👾Training!👾':  # training model
        with st.spinner('Training. . .'):
            t.sleep(5)
        col1.success("Training Complete . . .")
        data = pd.read_csv('cow_beef.csv')
        X = data.drop(columns='price', axis=1)
        Y = data['price']
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
        model = LogisticRegression()
        model.fit(x_train, y_train)
        save_model(model)
    if option == 'go!!!':
        pass
    breed = 0
    a_breed =col2.radio(
        "What\'s your breed?",
        ('Ready for Breeding (non-Wagyu)', 'Ready for Breeding (Wagyu)','Pregnant Cow (non-Wagyu)','Pregnant Cow (Wagyu)','steer (non-Wagyu)','steer  (Wagyu)'))
    if a_breed == 'Ready for Breeding (Wagyu)': breed = 1
    if a_breed == 'Pregnant Cow (non-Wagyu)': breed = 2
    if a_breed == 'Pregnant Cow (Wagyu)': breed = 3
    if a_breed == 'steer (non-Wagyu)': breed = 4
    if a_breed == 'steer  (Wagyu)': breed = 5

    age = col2.slider("Input your age", 0, 8)
    weight = col2.slider("Input your weight", 1000, 1600)
    sex = 0
    genre_sex = col2.radio(
        "What\'s your sex?",
        ('Female', 'Male'))
    if genre_sex == 'Male': sex = 1
    weight = col2.slider("weight", 0, 2000)

    Pred = col2.button('Press here to Prediction')
    if Pred:
        model = load_model()
        Pred_data = (breed, sex, weight, age)
        Pred_data_array = np.asarray(Pred_data)
        Pred_data_array_re = Pred_data_array.reshape(1, -1)
        predict_Price = model.predict(Pred_data_array_re)
        result = predict_Price[0]


    st.title("Cow beef")
    st.caption("This is a cow price prediction machine learning system.")
    col1, col2 = st.columns(2)
    col1.write("")
    col2.write("")
    col1.write("")
    col2.write("")
    col1.write("")
    col2.write("")
    col1.write("")
    col2.write("")
    col1.write("")
    col2.write("")
    col1.write("")
    col2.write("")
    col1.write("")
    col2.write("")
    col1.write("")
    col2.write("")
    col1.write("")
    col2.write("")
    col1.write("")
    col2.write("")
    col1.write("")
    col2.write("")
    col1.write("")
    col2.write("")
    col1.write("")
    col2.write("")

    if breed == 1:
        breed_txt = "Ready for Breeding (Wagyu)"
    elif breed == 2:
        breed_txt = "Pregnant Cow (non-Wagyu)"
    elif breed == 3:
        breed_txt = "Pregnant Cow (Wagyu)"
    elif breed == 4:
        breed_txt = "steer (non-Wagyu)"
    elif breed == 5:
        breed_txt = "steer  (Wagyu)"

    if Pred:
        col1.write("Breed:")
        col2.write(breed_txt)
        col1.write("Age:")
        col2.write(str(age))
        col1.write("Weight:")
        col2.write(str(weight))
        col1.write("Sex:")
        col2.write(genre_sex)
        col1.write("Price Prediction:")
        col2.write(str(result))
