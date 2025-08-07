import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from llama_index.llms.groq import Groq
from sentence_transformers import SentenceTransformer
import numpy as np
from transformers import pipeline
import re
from sklearn.metrics.pairwise import cosine_similarity
import whisper
import tempfile
from audio_recorder_streamlit import audio_recorder
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import json
from datetime import datetime
import plotly.graph_objects as go
import easyocr


load_dotenv()
GROQ_API_KEY = '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'
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable not found. Please set it in your .env file.")

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def load_data(file_path):
    return pd.read_excel(file_path, sheet_name=None)

def transcribe_voice(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(uploaded_file.read())
        whisper_model = whisper.load_model("large-v2")
        result = whisper_model.transcribe(f.name)
        return result["text"]

def generate_caption(image_file):
    reader = easyocr.Reader(['en'])  
    image = Image.open(image_file)
    result = reader.readtext(np.array(image), detail=0)  
    return " ".join(result)

SYSTEM_ROLE = """You are an advanced AI-powered Hyper-Personalization Engine specializing in financial services. 
Your core capabilities include:

1. Financial Recommendations: Analyzing customer profiles to provide actionable, personalized financial advice and strategies.
2. Product/Service Suggestions: Matching customers with tailored financial products based on their behavior, needs, and sentiment analysis.
3. Promotional Offers: Generating personalized retention offers and promotional recommendations based on customer engagement patterns.
4. Financial Health Assessment: Calculating and explaining customer financial health scores with detailed insights and improvement suggestions.
5. Personalized Product Matches: Using advanced matching algorithms to suggest the most relevant financial products with detailed explanations.

You maintain a professional yet friendly tone, prioritize data privacy, and ensure all recommendations are actionable, realistic and specific to each customer's unique profile."""
  
def initialize_groq_llama():
    return Groq(model="llama-3.3-70b-versatile", api_key=GROQ_API_KEY)

# Create the final prompt
def create_prompt_with_system_role(user_prompt):
    return f"{SYSTEM_ROLE}\n\nUser: {user_prompt}\nAssistant:"

# Generate concise financial recommendations
def generate_recommendations(user_data, llama_model):
    user_prompt = (
        "Based on the following real-time customer profile and behavior, "
        "provide 3 short, actionable financial recommendations. Be concise, use bullet points, and adapt. do not provide any introduction before the response. "
        "to recent shifts in behavior (e.g., from budget-conscious to luxury spending):\n\n"
        f"{user_data}\n\n"
        "Format:\n"
        "- Recommendation 1\n"
        "- Recommendation 2\n"
        "- Recommendation 3"
    )
    full_prompt = create_prompt_with_system_role(user_prompt)
    response = llama_model.complete(full_prompt)
    return response.text.strip()

# Generate product suggestions based on sentiment
def generate_product_suggestions(user_data, sentiment, llama_model):
    sentiment_note = (
        "The customer has shown positive sentiment recently, so feel free to recommend premium or advanced products."
        if sentiment == "Positive"
        else "The customer has shown negative sentiment recently. Prioritize helpful, supportive, or cost-saving products."
    )
    user_prompt = (
        f"{sentiment_note}\n\n"
        "Below is a customer's financial profile, including their behavior, recent transactions, and engagement history.\n"
        "Write 3 clear, friendly, and helpful product or service suggestions as if you're speaking directly to the customer.\n"
        "Be brief, use bullet points, and make the tone customer-centric and easy to understand. Do not include any introduction or extra text\n\n"
        f"Customer Profile:\n{user_data}\n\n"
        "Format:\n"
        "- Product/Service 1: (1-sentence benefit to the customer)\n"
        "- Product/Service 2: (1-sentence benefit to the customer)\n"
        "- Product/Service 3: (1-sentence benefit to the customer)"
    )
    full_prompt = create_prompt_with_system_role(user_prompt)
    response = llama_model.complete(full_prompt)
    return response.text.strip()

# Analyze sentiment using sentence embeddings
sentiment_pipeline = pipeline("sentiment-analysis")
sent_transformer = SentenceTransformer("all-MiniLM-L6-v2")

def analyze_sentiment(text):
    if not text.strip():
        return "Neutral"
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    results = [sentiment_pipeline(line)[0] for line in lines]
    sentiment_map = {"POSITIVE": 1, "NEGATIVE": -1, "NEUTRAL": 0}
    total_score = sum(sentiment_map.get(res["label"], 0) * res["score"] for res in results)
    if total_score > 0.2:
        return "Positive"
    elif total_score < -0.2:
        return "Negative"
    else:
        return "Neutral"

def build_customer_profile(customer_id, sheets, image_caption="", voice_text=""):
    ind = sheets["customer profile (individual)"]
    txn = sheets["transaction history"]
    sent = sheets["social media sentiments"]

    user = ind[ind["Customer_Id"] == customer_id].squeeze()
    user_txn = txn[txn["Customer_Id"] == customer_id]
    user_sent = sent[sent["Customer_Id"] == customer_id]

    parts = [
        f"Age: {user['Age']}, Gender: {user['Gender']}, Location: {user['Location']}",
        f"Occupation: {user['Occupation']}, Education: {user['Education']}",
        f"Income: {user['Income per year']}",
        f"Interests: {user['Interests']}",
        f"Preferences: {user['Preferences']}"
    ]
    if not user_txn.empty:
        categories = user_txn["Category"].value_counts().index.tolist()
        avg_amt = user_txn["Amount (In Dollars)"].mean()
        parts.append(f"Top categories: {', '.join(categories[:3])}")
        parts.append(f"Avg transaction amount: ${avg_amt:.2f}")

    if not user_sent.empty:
        avg_sent = user_sent["Sentiment_Score"].mean()
        intents = ", ".join(user_sent["Intent"].unique())
        parts.append(f"Avg sentiment: {avg_sent:.2f}")
        parts.append(f"Common intents: {intents}")

    if image_caption:
        parts.append(f"Image Insight: {image_caption}")
    if voice_text:
        parts.append(f"Voice Note: {voice_text}")

    return " | ".join(parts)

def match_products(customer_text, product_list):
    customer_vec = sent_transformer.encode([customer_text])
    product_vecs = sent_transformer.encode([p["description"] for p in product_list])
    scores = cosine_similarity(customer_vec, product_vecs).flatten()
    top_idxs = scores.argsort()[::-1][:3]
    return [product_list[i] for i in top_idxs]

financial_products = [
    {"name": "Wells Fargo Active Cash® Card", "description": "Unlimited 2% cashback on purchases."},
    {"name": "Way2Save® Savings", "description": "Automatic savings with each purchase."},
    {"name": "WellsTrade® Investment Account", "description": "Commission-free online stock trading."},
    {"name": "Wells Fargo Personal Loan", "description": "Flexible loan options with no collateral."},
    {"name": "Reflect® Card", "description": "Intro APR offer on balance transfers and purchases."}
]
llama_model = initialize_groq_llama()

# Recommend financial products based on matched products
def recommend_financial_products(customer_text, matched_products):
    product_block = "\n".join([
        f"{i+1}. {p['name']} - {p['description']}" for i, p in enumerate(matched_products)
    ])
    return (
        f"Customer Profile:\n{customer_text}\n\n"
        f"Based on the profile above, write 3 friendly product suggestions directly to the customer. "
        f"Explain briefly why each product is relevant. Use bullet points.\n\n"
        f"Product Options:\n{product_block}"
    )

# Predict insights using the text profile 
def predict_customer_behavior_text(enriched_text, llama_model):
    user_prompt = (
        "Here is the comprehensive customer profile including multimodal inputs:\n\n"
        f"{enriched_text}\n\n"
        "Based on this profile, generate strategic insights for relationship management including:\n"
        "1. A summary of customer behavior\n"
        "2. Customer preferences (e.g., budget-conscious, luxury spender, risk-averse, tech-savvy)\n"
        "3. Churn risk level (Low, Medium, High) and a brief reason\n"
        "4. Purchasing potential (Low, Medium, High)\n"
        "5. Retention offers (list exactly 3 offers)\n"
        "6. 2–3 recommended financial products or services\n\n"
        "Respond with a valid JSON object enclosed in triple backticks. "
        "The JSON object must have the following keys exactly: "
        "\"preferences\" (an array of strings), "
        "\"purchasing_potential\" (a string), and "
        "\"retention_offers\" (an array of strings).\n"
        "Example output:\n"
        "```\n"
        "{\n"
        '  "preferences": ["tech-savvy", "budget-conscious"],\n'
        '  "purchasing_potential": "Medium",\n'
        '  "retention_offers": ["Offer 1", "Offer 2", "Offer 3"]\n'
        "}\n"
        "```"
    )
    full_prompt = create_prompt_with_system_role(user_prompt)
    response = llama_model.complete(full_prompt)
    try:
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response.text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = response.text.strip()
        insights = json.loads(json_str)
    except Exception as e:
        insights = {"error": str(e), "llm_response": response.text.strip()}
    return insights

def calculate_financial_health_score(customer_data):
    
    txn_data = customer_data.get("transaction history", [])
    if not txn_data:
        return None

    txn_df = pd.DataFrame(txn_data)
    num_txn = len(txn_df)
    unique_categories = txn_df["Category"].nunique() if "Category" in txn_df else 0

    sentiment_data = customer_data.get("social media sentiments", [])
    if sentiment_data:
        sent_df = pd.DataFrame(sentiment_data)
        avg_sentiment = sent_df["Sentiment_Score"].mean() if "Sentiment_Score" in sent_df else 0.5
    else:
        avg_sentiment = 0.5

    score = (num_txn * 4) + (unique_categories * 6) + (avg_sentiment * 50)
    score = min(100, score)
    return score, num_txn, unique_categories, avg_sentiment

def extract_additional_insights(score, num_txn, unique_categories, avg_sentiment, llama_model):
    user_prompt = (
        f"For a customer, the financial health score is {score:.1f}/100. "
        f"This score is calculated based on {num_txn} transactions, "
        f"{unique_categories} unique spending categories, and an average sentiment score of {avg_sentiment:.2f}.\n\n"
        "Please provide analysis that includes:\n"
        "1. What these numbers suggest about the customer's financial behavior in 2 lines,\n"
        "2. Potential financial risks in a crisp and concise manner,\n\n"
        "Format your response in bullet points. Do not include any introduction line to response"
    )
    full_prompt = create_prompt_with_system_role(user_prompt)
    response = llama_model.complete(full_prompt)
    return response.text.strip()

def fine_tune_llm_with_rlhf(feedback_data, current_model):
    

    #st.write("Simulated RLHF Training:")
    #for rec, rating in feedback_data:
        #st.write(f"Recommendation: {rec}\nRating: {rating}")
    #st.success("RLHF training complete. The model has been updated based on feedback (simulation).")
    return current_model


def main():
    st.title("AI-Powered Hyper-Personalization Engine")

    # Upload Customer Data File OR load historical data if none is uploaded
    uploaded_file = st.file_uploader("Upload Customer Excel File", type=["xlsx"])
    if uploaded_file is None:
        historical_file = "C:/Users/chand/Downloads/hackathon_data.xlsx"  # adjust path as needed
        sheets = load_data(historical_file)
    else:
        sheets = load_data(uploaded_file)
    
    st.markdown("### Multimodal Inputs")
    col1, col2 = st.columns(2)
    with col1:
        customer_id_for_image = st.text_input("Customer ID for Image")
        image_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"], key="image_upload")
    with col2:
        customer_id_for_voice = st.text_input("Customer ID for Voice")
        voice_file = st.file_uploader("Upload a Voice Message", type=["mp3", "wav"], key="voice_upload")
    
    image_caption = ""
    voice_text = ""

    if image_file:
        if customer_id_for_image.strip():
            
            image_caption = generate_caption(image_file)
            with st.expander("🖼️ Extracted Text from Image"):
                st.markdown(f"**Caption:** {image_caption}")
        else:
            st.warning("Please enter Customer ID for Image to process it.")
    
   
    if voice_file:
        if customer_id_for_voice.strip():
            voice_text = transcribe_voice(voice_file)
            st.audio(voice_file)
            with st.expander("🎤 Extracted Text from voice"):
                st.markdown(f"**Voice Transcription:** {voice_text}")
        else:
            st.warning("Please enter Customer ID for Voice to process it.")
    
    if sheets is not None:
        customer_ids = set()
        for df in sheets.values():
            if "Customer_Id" in df.columns:
                customer_ids.update(df["Customer_Id"].unique())
        customer_id = st.selectbox("Select a Customer Id", sorted(customer_ids))
    
        def merge_customer_data(sheets, customer_id):
            combined_data = {}
            for sheet_name, df in sheets.items():
                if "Customer_Id" not in df.columns:
                    continue
                customer_rows = df[df["Customer_Id"] == customer_id]
                if not customer_rows.empty:
                    combined_data[sheet_name] = customer_rows.to_dict(orient="records")
            return combined_data
    
        customer_data = merge_customer_data(sheets, customer_id)
        st.subheader(f"🧾 Combined Profile for {customer_id}")
        for section, entries in customer_data.items():
            st.markdown(f"**{section.title()}**")
            st.dataframe(pd.DataFrame(entries))
    
        customer_text = build_customer_profile(customer_id, sheets, image_caption, voice_text)
    
        llama_model = initialize_groq_llama()
    
        # --- RLHF: Generate Recommendations & Collect Feedback ---
        if "rec_list" not in st.session_state:
            st.session_state["rec_list"] = []
        if "show_feedback_form" not in st.session_state:
            st.session_state["show_feedback_form"] = False
        if "updated_recs" not in st.session_state:
            st.session_state["updated_recs"] = []

        # Button to generate initial recommendations
        if st.button("📊 Generate Recommendations"):
            recommendations = generate_recommendations(customer_text, llama_model)
            remove_phrase = "Based on the customer profile, here are three short, actionable financial recommendations"
            recommendations = recommendations.replace(remove_phrase, "")
    
            bullets = [b.strip() for b in recommendations.split("- ") if b.strip()]
            if bullets and (bullets[0].lower().startswith("based on the") or bullets[0] == ""):
                bullets = bullets[1:]
    
            st.session_state["rec_list"] = []
            for idx, item in enumerate(bullets, start=1):
                parts = item.split(":", 1)
                title = parts[0].strip() if len(parts) > 1 else f"Recommendation {idx}"
                description = parts[1].strip() if len(parts) > 1 else item
                st.session_state["rec_list"].append((title, description))
            st.session_state["show_feedback_form"] = True
    
        # Display recommendations and feedback form if available
        if st.session_state.get("show_feedback_form") and st.session_state["rec_list"]:
            st.markdown("#### Initial Recommendations:")
            for idx, (title, description) in enumerate(st.session_state["rec_list"], start=1):
                with st.expander(f"{idx}. {title}"):
                    st.write(description)
    
            st.markdown("#### Please rate the above recommendations (1 = poor, 5 = excellent)")
            with st.form("feedback_form"):
                feedback = []
                for idx, (title, description) in enumerate(st.session_state["rec_list"], start=1):
                    rating = st.slider(
                        f"Rating for Recommendation {idx} - {title}",
                        min_value=1,
                        max_value=5,
                        key=f"rec_rating_{idx}"
                    )
                    feedback.append((f"{title}: {description}", rating))
                submitted = st.form_submit_button("Submit Feedback")
                if submitted:
                    #st.write("Feedback submitted. Ratings:")
                    #st.write(feedback)
                    # Simulated RLHF training: update the model based on feedback
                    llama_model = fine_tune_llm_with_rlhf(feedback, llama_model)
                    st.session_state["show_feedback_form"] = False
                    # Generate updated recommendations with the updated model
                    updated_recommendations = generate_recommendations(customer_text, llama_model)
                    remove_phrase = "Based on the customer profile, here are three short, actionable financial recommendations"
                    updated_recommendations = updated_recommendations.replace(remove_phrase, "")

                    st.session_state["updated_recs"] = [r.strip() for r in updated_recommendations.split("- ") if r.strip()]
    
        # Display updated recommendations after RLHF update
        if st.session_state.get("updated_recs"):
            st.markdown("### Updated Recommendations after RLHF:")
            updated_list = []
            for idx, item in enumerate(st.session_state["updated_recs"], start=1):
                parts = item.split(":", 1)
                title = parts[0].strip() if len(parts) > 1 else f"Updated Recommendation {idx}"
                description = parts[1].strip() if len(parts) > 1 else item
                updated_list.append((title, description))
            for idx, (title, description) in enumerate(updated_list, start=1):
                with st.expander(f"{idx}. {title}"):
                    st.write(description)
    
        # --- Generate Product/Service Suggestions ---
        if st.button("🎯 Product/Service Suggestions"):
            social_text = ""
            if "social media sentiments" in customer_data:
                df = pd.DataFrame(customer_data["social media sentiments"])
                social_text = " ".join(df.select_dtypes(include="object").astype(str).values.flatten())
            sentiment = analyze_sentiment(social_text)
            st.markdown(
                f"####  <span style='color: {'green' if sentiment == 'Positive' else 'red'};'>{sentiment}</span>",
                unsafe_allow_html=True,
            )
            suggestions = generate_product_suggestions(customer_text, sentiment, llama_model)
            bullets = [b.strip() for b in suggestions.split("- ") if b.strip()]
            for idx, item in enumerate(bullets, start=1):
                parts = item.split(":", 1)
                title = parts[0].strip() if len(parts) > 1 else f"Product {idx}"
                description = parts[1].strip() if len(parts) > 1 else item
                with st.expander(f"{idx}. {title}"):
                    st.write(description)
    
        # --- Predict Customer Insights using enriched text (Option 2) ---
        if st.button("🎁 Promotion Offers for you!"):
            insights = predict_customer_behavior_text(customer_text, llama_model)
            retention_offers = insights.get("retention_offers", [])
            if retention_offers:
                for idx, offer in enumerate(retention_offers, 1):
                    with st.expander(f"Offer {idx}"):
                        st.write(offer)
            else:
                st.write("No insights available.")
    
        # --- Show Financial Health Score & Additional Insights ---
        if st.button("🧾Show Financial Health Score", key="financial_health_button"):
            health_score, num_txn, unique_categories, avg_sentiment = calculate_financial_health_score(customer_data)
            additional_insights = extract_additional_insights(health_score, num_txn, unique_categories, avg_sentiment, llama_model)
    
            if health_score is not None:
                col1, col2 = st.columns([1, 1])
                with col1:
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=health_score,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        gauge={
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "#1f77b4"},
                            'bgcolor': "white",
                            'borderwidth': 2,
                            'bordercolor': "gray",
                            'steps': [
                                {'range': [0, 50], 'color': '#ff9999'},
                                {'range': [50, 100], 'color': '#99ff99'}
                            ],
                        },
                    ))
    
                    fig.update_layout(
                        height=150,
                        font={'size': 10},
                        margin=dict(l=20, r=20, t=30, b=20)
                    )
    
                    st.plotly_chart(fig, use_container_width=True)
    
                    if health_score < 50:
                        st.warning("Your financial health score indicates room for improvement. Consider reviewing your spending habits and exploring our tailored recommendations.")
                    else:
                        st.success("Great! Your financial health score is strong. Keep up the good work!")
    
                st.markdown("### Additional Insights:")
                st.write(additional_insights)
            else:
                st.write("Insufficient data to calculate the financial health score.")
    
        # --- Personalized Product Matches (Flip Cards) ---
        if st.button("💡 Personalized Product Suggestions"):
            top_products = match_products(customer_text, financial_products)
            prompt = recommend_financial_products(customer_text, top_products)
            response = llama_model.complete(prompt)
            card_data = []
            current_card = {}
            for line in response.text.strip().split("\n"):
                line = line.strip().lstrip("*").strip()
                if not line:
                    continue
                if ":" in line:
                    if current_card:
                        card_data.append(current_card)
                    name, desc = line.split(":", 1)
                    if len(name.strip().split()) < 2 or "suggestion" in name.lower():
                        current_card = {}
                        continue
                    current_card = {"name": name.strip(), "bullets": [desc.strip()]}
                elif line.startswith("-") and current_card:
                    current_card["bullets"].append(line.lstrip("-").strip())
            if current_card:
                card_data.append(current_card)
    
            st.markdown(
                """
                <style>
                .flip-card { background-color: transparent; width: 100%; height: 150px; perspective: 1000px; margin-bottom: 20px; }
                .flip-card-inner { position: relative; width: 100%; height: 100%; text-align: center; transition: transform 0.8s; transform-style: preserve-3d; }
                .flip-card:hover .flip-card-inner { transform: rotateY(180deg); cursor: pointer; }
                .flip-card-front, .flip-card-back {
                    position: absolute; width: 100%; height: 100%; padding: 15px; backface-visibility: hidden;
                    border: 1px solid #eee; border-radius: 10px; background-color: #f9f9f9;
                }
                .flip-card-front {
                    display: flex; align-items: center; justify-content: center;
                    font-weight: bold; font-size: 1.1rem; color: #8B0000;
                }
                .flip-card-back {
                    transform: rotateY(180deg); text-align: left; font-size: 0.95rem; color: #444;
                }
                </style>
                """,
                unsafe_allow_html=True,
            )
            for product in card_data:
                bullets_html = "".join(f"<li>{b}</li>" for b in product["bullets"])
                st.markdown(
                    f"""
                    <div class="flip-card">
                        <div class="flip-card-inner">
                            <div class="flip-card-front">{product['name']}</div>
                            <div class="flip-card-back"><ul>{bullets_html}</ul></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


if __name__ == "__main__":
    main()
