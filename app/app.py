import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords

# Download stopwords if not already downloaded
try:
    stopwords.words('english')
except:
    nltk.download('stopwords', quiet=True)

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Restaurant Sentiment Analyzer",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS STYLING
# ============================================================================

st.markdown("""
    <style>
    /* Main background */
    .main {
        background-color: #f5f7fa;
    }
    
    /* Sidebar styling - FIXED FOR VISIBILITY */
    [data-testid="stSidebar"] {
        background-color: #2c3e50 !important;
        padding: 20px;
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: white !important;
    }
    
    /* Text area styling */
    .stTextArea textarea {
        font-size: 16px;
        border-radius: 10px;
        border: 2px solid #e0e0e0;
    }
    
    /* Result boxes */
    .positive-result {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
    }
    .negative-result {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 30px;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(245, 87, 108, 0.4);
    }
    
    /* Metric cards */
    .metric-card {
        background-color: rgba(255, 255, 255, 0.1);
        padding: 20px;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# LOAD MODELS
# ============================================================================

@st.cache_resource
def load_models():
    """Load the trained model and vectorizer"""
    with open('models/best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

model, vectorizer = load_models()

# ============================================================================
# TEXT PREPROCESSING FUNCTIONS
# ============================================================================

def clean_text(text):
    """Clean and normalize text"""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = ' '.join(text.split())
    return text

def remove_stopwords(text):
    """Remove stopwords while keeping negations"""
    stop_words = set(stopwords.words('english'))
    negation_words = {'not', 'no', 'nor', 'never', 'neither', 'nobody', 
                     'nothing', 'nowhere', "don't", "didn't", "doesn't", 
                     "won't", "wouldn't", "shouldn't", "couldn't", "can't"}
    stop_words = stop_words - negation_words
    
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)

def preprocess_review(text):
    """Complete preprocessing pipeline"""
    cleaned = clean_text(text)
    processed = remove_stopwords(cleaned)
    return processed

# ============================================================================
# PREDICTION FUNCTION
# ============================================================================

def predict_sentiment(review_text):
    """Predict sentiment of a review"""
    # Preprocess
    processed_text = preprocess_review(review_text)
    
    # Vectorize
    vectorized = vectorizer.transform([processed_text])
    
    # Predict
    prediction = model.predict(vectorized)[0]
    confidence = model.decision_function(vectorized)[0]
    
    # Convert confidence to probability-like score
    probability = 1 / (1 + abs(confidence))
    if prediction == 1:
        confidence_score = 0.5 + (probability * 0.5)
    else:
        confidence_score = 0.5 + (probability * 0.5)
    
    return prediction, confidence_score

# ============================================================================
# HEADER
# ============================================================================

st.markdown("<h1 style='text-align: center; color: #2c3e50;'>🍽️ Restaurant Review Sentiment Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px; color: #7f8c8d;'>Powered by Machine Learning | SVM Model with 84.5% Accuracy</p>", unsafe_allow_html=True)
st.markdown("---")

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.header("📊 Model Information")
    
    st.markdown("""
    <div class='metric-card'>
    <h3>🏆 Best Model</h3>
    <p><strong>Algorithm:</strong> Support Vector Machine (SVM)</p>
    <p><strong>Accuracy:</strong> 84.5%</p>
    <p><strong>Precision:</strong> 85%</p>
    <p><strong>Recall:</strong> 84-85%</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='metric-card'>
    <h3>📈 All Models Compared</h3>
    <ul>
        <li>SVM: 84.5% 🥇</li>
        <li>Logistic Regression: 82.5% 🥈</li>
        <li>Naive Bayes: 80.0% 🥉</li>
        <li>Random Forest: 79.0%</li>
        <li>Decision Tree: 73.0%</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    <div class='metric-card'>
    <h3>ℹ️ About</h3>
    <p>This app uses Natural Language Processing to analyze restaurant reviews and predict customer sentiment.</p>
    <p><strong>Dataset:</strong> 996 reviews</p>
    <p><strong>Features:</strong> 1000 TF-IDF vectors</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("**👨‍💻 Built by:** Akakinad")
    st.markdown("[![GitHub](https://img.shields.io/badge/GitHub-View_Code-black?logo=github)](https://github.com/Akakinad/restaurant-sentiment-analysis)")

# ============================================================================
# MAIN APP
# ============================================================================

# Create two columns
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("✍️ Enter Your Review")
    review_text = st.text_area(
        "Type or paste a restaurant review below:",
        height=150,
        placeholder="Example: The food was amazing and the service was excellent!",
        help="Enter any restaurant review to analyze its sentiment"
    )
    
    analyze_button = st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True)

with col2:
    st.subheader("🎯 Quick Test Examples")
    st.markdown("Click to try these sample reviews:")
    
    if st.button("✅ Great service and delicious food!", use_container_width=True):
        review_text = "Great service and delicious food!"
        analyze_button = True
    
    if st.button("❌ Waited too long, cold food", use_container_width=True):
        review_text = "Waited too long, cold food"
        analyze_button = True
    
    if st.button("✅ Amazing experience, will come back!", use_container_width=True):
        review_text = "Amazing experience, will come back!"
        analyze_button = True
    
    if st.button("❌ Terrible service and overpriced", use_container_width=True):
        review_text = "Terrible service and overpriced"
        analyze_button = True

# ============================================================================
# PREDICTION & RESULTS
# ============================================================================

if analyze_button and review_text.strip():
    with st.spinner("🤖 Analyzing sentiment..."):
        # Get prediction
        sentiment, confidence = predict_sentiment(review_text)
        
        st.markdown("---")
        st.subheader("📊 Analysis Results")
        
        # Display result
        if sentiment == 1:
            st.markdown(f"""
            <div class='positive-result'>
                <h1 style='margin:0; font-size: 60px;'>😊</h1>
                <h2 style='margin:10px 0;'>POSITIVE SENTIMENT</h2>
                <p style='font-size: 20px; margin:10px 0;'>This review expresses satisfaction!</p>
                <h3 style='margin:20px 0;'>Confidence: {confidence*100:.1f}%</h3>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='negative-result'>
                <h1 style='margin:0; font-size: 60px;'>😞</h1>
                <h2 style='margin:10px 0;'>NEGATIVE SENTIMENT</h2>
                <p style='font-size: 20px; margin:10px 0;'>This review expresses dissatisfaction.</p>
                <h3 style='margin:20px 0;'>Confidence: {confidence*100:.1f}%</h3>
            </div>
            """, unsafe_allow_html=True)
        
        # Show confidence meter
        st.markdown("### Confidence Score")
        st.progress(confidence)
        
        # Show processed text
        with st.expander("🔍 See Processed Text"):
            processed = preprocess_review(review_text)
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Original:**")
                st.info(review_text)
            with col_b:
                st.markdown("**After Processing:**")
                st.success(processed)

elif analyze_button:
    st.warning("⚠️ Please enter a review to analyze!")

# ============================================================================
# FOOTER WITH VISITOR COUNTER
# ============================================================================

st.markdown("---")

# Visitor counter using hits.sh (more reliable)
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown("""
    <div style='text-align: center;'>
        <img src='https://hits.sh/github.com/Akakinad/restaurant-sentiment-analysis.svg?label=Visitors&color=4c1&labelColor=2c3e50' alt='Visitors'/>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div style='text-align: center; color: #7f8c8d; padding: 20px;'>
    <p>Built with Streamlit 🎈 | Powered by Machine Learning 🤖</p>
    <p>This is a portfolio project demonstrating end-to-end ML workflow</p>
</div>
""", unsafe_allow_html=True)