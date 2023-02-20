import streamlit as st
import demo as d1
import demo2 as d2
import demo3 as d3
import time as t

with st.sidebar:
    genre = st.selectbox(
        "Choose option",
        ('ภูมิใจนำเสนอ', 'Cow price Prediction','Cow_wagyu_data','beef_cow_data'))

if genre == 'ภูมิใจนำเสนอ':

    with st.spinner():
        t.sleep(5)

    st.title("ภูมิใจนำเสนอ")
    st.header(
        "ยินดีต้อนรับสู่ Cow price prediction Project ที่ไว้คำนวนราคาวัวที่คุณต้องการ ")
    st.text("นายดวงตะวัน สิ่งส่า 65114540215 ")
    st.markdown("")

    st.markdown(
        f"""
           <style>
           .stApp {{
               background-image: url("https://rawryou.com/wp-content/uploads/2021/05/12.png");
               background-attachment: fixed;
               background-size: cover;
               /* opacity: 0.3; */
           }}
           </style>
           """,
        unsafe_allow_html=True
    )

if genre == 'Cow price Prediction':
    d1.page_one()

if genre == 'Cow_wagyu_data':
    d2.page_two()

if genre == 'beef_cow_data':
    d3.page_three()
