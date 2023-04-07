import streamlit as st



def show_privacy():


    streamlit_style = """
        <style>
        @import url('https://fonts.googleapis.com/css?family=Poppins');

        html, body, [class*="css"]  {
        font-family: 'Poppins', sans-serif;
        }
        </style>
        """
    st.markdown(streamlit_style, unsafe_allow_html=True)

    st.title("PRIVACY STATEMENT")
    st.write("")
    st.write("PRIVACY CONSENT STATEMENT")
    st.write("Holy Angel University respects the student’s privacy and are committed to protecting information. By using the enrollment prediction app, applicants are consenting to the collection, use, and disclosure of personal information as described in this privacy policy.")
    st.write("")
