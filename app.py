import streamlit as st
import pickle
import pandas as pd
import time
import streamlit.components.v1 as components

voting_pipeline = pickle.load(open('voting_pipeline.pkl', 'rb'))

st.title("SMS Spam Classifier")

input_sms = st.text_area("Enter your message")

if st.button("Predict"):
    if input_sms.strip():
        input_df = pd.DataFrame({'text': [input_sms.strip()]})
        prediction = voting_pipeline.predict(input_df)[0]
        if prediction == 1:
            st.header("Spam")
        else:
            st.header("Not Spam")

        # Wait and clear
        time.sleep(5)
        components.html(
            """<script>
            const textarea = parent.document.querySelector('textarea');
            if (textarea) textarea.value = "";
            </script>""",
            height=0,
        )
    else:
        st.warning("Please enter a message!")