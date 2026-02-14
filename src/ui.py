"""
Streamlit UI for sentiment analysis.
Provides a simple web interface to interact with the sentiment analysis API.
"""
import os
import streamlit as st
import requests

# Configuration
API_URL = os.getenv('API_URL', 'http://api:8000')

st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="😊",
    layout="centered"
)

st.title("Sentiment Analysis Tool")
st.markdown("---")

st.markdown("""
Enter any text below to analyze its sentiment using our fine-tuned BERT model.
The model will classify the sentiment as either **positive** or **negative** 
and provide a confidence score.
""")

st.markdown("---")

# Text input
user_text = st.text_area(
    "Enter text for sentiment analysis:",
    placeholder="Type or paste your text here...",
    height=150
)

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    analyze_button = st.button("Analyze Sentiment", use_container_width=True)

if analyze_button:
    if not user_text or not user_text.strip():
        st.error("Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing..."):
            try:
                response = requests.post(
                    f"{API_URL}/predict",
                    json={"text": user_text},
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    sentiment = result['sentiment'].upper()
                    confidence = result['confidence']
                    
                    st.markdown("---")
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Sentiment", sentiment)
                    
                    with col2:
                        st.metric("Confidence", f"{confidence:.2%}")
                    
                    st.markdown("---")
                    
                    # Visual feedback
                    if sentiment == "POSITIVE":
                        st.success(f"✅ This text has a positive sentiment!")
                    else:
                        st.warning(f"⚠️ This text has a negative sentiment.")
                    
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")
            
            except requests.exceptions.ConnectionError:
                st.error("❌ Could not connect to the API. Make sure the API service is running.")
            except requests.exceptions.Timeout:
                st.error("❌ Request timed out. The API took too long to respond.")
            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")

st.markdown("---")
st.markdown(
    """
    **How it works:**
    - This application uses a fine-tuned DistilBERT model trained on the IMDB dataset
    - The model analyzes the sentiment of your input text
    - Results include both the predicted sentiment class and a confidence score
    """
)
