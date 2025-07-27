import pandas as pd
import joblib
from fuzzywuzzy import process
from groq import Groq
import os
from dotenv import load_dotenv
import streamlit as st
import warnings
import numpy as np

# === Setup ===
load_dotenv()
warnings.filterwarnings("ignore")

# Debug API key loading
api_key = st.secrets.get("GROQ_API_KEY", "")

# DEBUG: Add this temporarily to see what's happening
if api_key:
    st.sidebar.success(f"✅ API Key loaded: {api_key[:10]}...{api_key[-4:]}")
else:
    st.sidebar.error("❌ No API key found in secrets")
    st.sidebar.info("Check your Streamlit secrets configuration")

# Load dataset
df = pd.read_csv("pakwheels_used_cars.csv", encoding="utf-8")

# === Intelligent Price Calculation Functions ===
def calculate_base_price_by_segments(make, model_name, engine_cc, age):
    """Calculate base price using market segments and depreciation"""
    
    # Market segments with base prices (in PKR)
    luxury_brands = ['mercedes', 'bmw', 'audi', 'lexus', 'infiniti']
    premium_brands = ['toyota', 'honda', 'nissan', 'hyundai', 'kia', 'mazda']
    economy_brands = ['suzuki', 'daihatsu', 'changan', 'proton', 'united']
    
    # Base prices by engine size and brand category
    base_prices = {
        'luxury': {
            'small': 3000000,   # <1500cc
            'medium': 5000000,  # 1500-2500cc
            'large': 8000000    # >2500cc
        },
        'premium': {
            'small': 1500000,   # <1500cc
            'medium': 2500000,  # 1500-2500cc
            'large': 4000000    # >2500cc
        },
        'economy': {
            'small': 800000,    # <1500cc
            'medium': 1200000,  # 1500-2500cc
            'large': 1800000    # >2500cc
        }
    }
    
    # Determine brand category
    make_lower = make.lower()
    if make_lower in luxury_brands:
        category = 'luxury'
    elif make_lower in premium_brands:
        category = 'premium'
    else:
        category = 'economy'
    
    # Determine engine size category
    if engine_cc < 1500:
        size_cat = 'small'
    elif engine_cc <= 2500:
        size_cat = 'medium'
    else:
        size_cat = 'large'
    
    base_price = base_prices[category][size_cat]
    
    # Apply model-specific adjustments
    popular_models = {
        'corolla': 1.2, 'civic': 1.15, 'city': 1.1, 'accord': 1.3,
        'alto': 0.9, 'cultus': 0.95, 'swift': 1.05, 'wagon r': 0.92,
        'vitz': 1.0, 'aqua': 1.1, 'prius': 1.25, 'camry': 1.4
    }
    
    model_multiplier = popular_models.get(model_name.lower(), 1.0)
    base_price *= model_multiplier
    
    # Apply age-based depreciation (more realistic curve)
    if age <= 1:
        depreciation = 0.85  # 15% depreciation in first year
    elif age <= 3:
        depreciation = 0.75 - (age - 1) * 0.05  # 5% per year for years 2-3
    elif age <= 7:
        depreciation = 0.65 - (age - 3) * 0.04  # 4% per year for years 4-7
    elif age <= 15:
        depreciation = 0.49 - (age - 7) * 0.02  # 2% per year for years 8-15
    else:
        depreciation = max(0.15, 0.33 - (age - 15) * 0.01)  # Minimum 15% of base
    
    return int(base_price * depreciation)

def calculate_intelligent_range(raw_input, predicted_price, df):
    """Calculate intelligent price range based on multiple factors"""
    
    # Get similar cars from dataset
    similar_cars = df[
        (df['make'].str.lower() == raw_input['make'].lower()) &
        (df['model'].str.lower() == raw_input['model'].lower())
    ]
    
    # Check if required columns exist in the dataset
    has_age = 'age' in df.columns
    has_mileage = 'mileage' in df.columns
    has_price = 'price' in df.columns
    
    # If we have similar cars and price column, use their price distribution
    if len(similar_cars) >= 3 and has_price:
        prices = similar_cars['price'].dropna()
        if len(prices) > 0:
            # Use percentiles for range
            q10, q25, q75, q90 = prices.quantile([0.1, 0.25, 0.75, 0.9])
            
            # Try to filter by age and mileage if columns exist
            similar_age_mileage = similar_cars.copy()
            
            if has_age:
                similar_age_mileage = similar_age_mileage[
                    similar_age_mileage['age'].between(raw_input['age'] - 2, raw_input['age'] + 2)
                ]
            
            if has_mileage:
                similar_age_mileage = similar_age_mileage[
                    similar_age_mileage['mileage'].between(raw_input['mileage'] * 0.7, raw_input['mileage'] * 1.3)
                ]
            
            if len(similar_age_mileage) >= 3:
                adj_prices = similar_age_mileage['price'].dropna()
                if len(adj_prices) > 0:
                    lower = int(adj_prices.quantile(0.25))
                    upper = int(adj_prices.quantile(0.75))
                else:
                    # Fall back to original percentiles
                    lower = int(q25)
                    upper = int(q75)
            else:
                # Adjust percentiles based on age and mileage
                age_factor = max(0.7, 1 - (raw_input['age'] * 0.03))
                mileage_factor = max(0.8, 1 - (raw_input['mileage'] / 1000000))
                
                lower = int(q25 * age_factor * mileage_factor)
                upper = int(q75 * age_factor * mileage_factor)
    else:
        # Use intelligent base calculation
        base_price = calculate_base_price_by_segments(
            raw_input['make'], 
            raw_input['model'], 
            raw_input['engine_cc'], 
            raw_input['age']
        )
        
        # Create range based on various factors
        range_factors = []
        
        # Mileage factor
        avg_annual_mileage = 15000
        expected_mileage = raw_input['age'] * avg_annual_mileage
        if raw_input['mileage'] > expected_mileage * 1.5:
            range_factors.append(0.85)  # High mileage
        elif raw_input['mileage'] < expected_mileage * 0.7:
            range_factors.append(1.15)  # Low mileage
        else:
            range_factors.append(1.0)   # Average mileage
        
        # Transmission factor
        if raw_input['is_automatic']:
            range_factors.append(1.1)   # Automatic premium
        else:
            range_factors.append(1.0)
        
        # Import factor
        if raw_input['is_imported']:
            range_factors.append(1.2)   # Import premium
        else:
            range_factors.append(1.0)
        
        # Fuel type factor
        fuel_multipliers = {
            'petrol': 1.0,
            'diesel': 1.05,
            'hybrid': 1.15,
            'electric': 1.25,
            'cng': 0.95
        }
        fuel_factor = fuel_multipliers.get(raw_input['fuel_type'].lower(), 1.0)
        range_factors.append(fuel_factor)
        
        # Calculate final adjustment
        total_factor = np.prod(range_factors)
        adjusted_base = int(base_price * total_factor)
        
        # Create range (±15% from adjusted base)
        lower = int(adjusted_base * 0.85)
        upper = int(adjusted_base * 1.15)
        
        # Ensure the ML prediction is within a reasonable range of our calculation
        ml_vs_calculated_ratio = predicted_price / adjusted_base
        if 0.5 <= ml_vs_calculated_ratio <= 2.0:
            # ML prediction is reasonable, blend it with our calculation
            blended_price = int((predicted_price * 0.4) + (adjusted_base * 0.6))
            lower = int(blended_price * 0.85)
            upper = int(blended_price * 1.15)
        else:
            # ML prediction seems off, rely more on our calculation
            lower = int(adjusted_base * 0.85)
            upper = int(adjusted_base * 1.15)
    
    # Ensure minimum range and reasonable bounds
    min_range = 100000  # Minimum 100k range
    if upper - lower < min_range:
        mid_point = (upper + lower) // 2
        lower = mid_point - min_range // 2
        upper = mid_point + min_range // 2
    
    # Ensure positive values
    lower = max(50000, lower)
    upper = max(lower + min_range, upper)
    
    return lower, upper

def get_market_condition_factor(make, model_name, age):
    """Get market condition multiplier based on current demand"""
    
    # High demand models (retain value better)
    high_demand = ['corolla', 'civic', 'city', 'vitz', 'aqua', 'prius']
    # Low demand models (depreciate faster)
    low_demand = ['cultus', 'mehran', 'khyber', 'baleno']
    
    model_lower = model_name.lower()
    
    if model_lower in high_demand:
        return 1.1 if age <= 5 else 1.05
    elif model_lower in low_demand:
        return 0.9 if age <= 5 else 0.85
    else:
        return 1.0

# === Fuzzy correction ===
def correct_input(user_input, choices, threshold=70):
    match, score = process.extractOne(user_input, choices)
    return match if score >= threshold else user_input

# === AI Description function ===
def generate_ai_description(raw_input, lower, upper, expected_mileage, api_key):
    """Generate AI description using Groq API"""
    try:
        # Check if API key exists and is not empty
        if not api_key or api_key.strip() == "":
            return None, "API key is missing or empty"
            
        # Initialize Groq client
        client = Groq(api_key=api_key.strip())
        
        prompt = (
            f"You are an expert Pakistani car market analyst. Provide a detailed 5-6 line analysis of why a {2024 - raw_input['age']} "
            f"{raw_input['make']} {raw_input['model']} ({raw_input['body']}) with {raw_input['mileage']:,} km mileage, "
            f"{raw_input['engine_cc']}cc {raw_input['fuel_type']} engine, and "
            f"{'automatic' if raw_input['is_automatic'] else 'manual'} transmission is intelligently valued between Rs. {lower:,} and Rs. {upper:,}. "
            f"The expected mileage for this age would be {expected_mileage:,} km. "
            f"Consider market positioning, demand trends, fuel efficiency, maintenance costs, and resale potential. "
            f"Mention specific factors affecting this vehicle's pricing in Pakistan's current market."
        )

        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": "You are a professional automotive market analyst specializing in Pakistan's used car market with deep knowledge of pricing trends, brand positioning, and consumer preferences."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )

        return response.choices[0].message.content, None
        
    except Exception as e:
        return None, str(e)

# === Streamlit page config & style ===
st.set_page_config(page_title="Car Price Estimator", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        /* Root variables for consistent theming */
        :root {
            --primary-navy: #001f4d;
            --secondary-navy: #003366;
            --accent-blue: #0066cc;
            --light-blue: #00aaff;
            --dark-bg: #0a0a0a;
            --darker-bg: #050505;
            --text-primary: #ffffff;
            --text-secondary: #e0e0e0;
            --border-color: #333333;
            --gradient-bg: linear-gradient(135deg, #001f4d 0%, #003366 50%, #0066cc 100%);
        }
        
        /* Main app styling */
        .main {
            background: linear-gradient(180deg, #0a0a0a 0%, #0f0f0f 100%);
            color: var(--text-primary);
            font-family: 'Inter', sans-serif;
        }
        
        /* Hide Streamlit default elements */
        .stApp > header {
            background-color: transparent;
        }
        
        .stApp > div:first-child {
            background-color: transparent;
        }
        
        /* Enhanced navy ribbon with gradient and shadow - Full width */
        .navy-ribbon {
            background: var(--gradient-bg);
            color: var(--text-primary);
            padding: 20px 40px;
            font-size: 28px;
            font-weight: 700;
            border-radius: 0;
            margin: -20px -100vw 30px -100vw;
            padding-left: 100vw;
            padding-right: 100vw;
            text-align: center;
            box-shadow: 0 8px 32px rgba(0, 31, 77, 0.3);
            border: 2px solid rgba(255, 255, 255, 0.1);
            border-left: none;
            border-right: none;
            backdrop-filter: blur(10px);
            position: relative;
            overflow: hidden;
        }
        
        .navy-ribbon::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
            animation: shimmer 3s infinite;
        }
        
        @keyframes shimmer {
            0% { left: -100%; }
            100% { left: 100%; }
        }
        
        /* Enhanced button styling */
        .stButton > button {
            background: var(--gradient-bg) !important;
            color: var(--text-primary) !important;
            border: none !important;
            border-radius: 12px !important;
            padding: 15px 40px !important;
            font-size: 18px !important;
            font-weight: 600 !important;
            box-shadow: 0 6px 20px rgba(0, 31, 77, 0.3) !important;
            transition: all 0.3s ease !important;
            text-transform: uppercase !important;
            letter-spacing: 1px !important;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 8px 25px rgba(0, 102, 204, 0.4) !important;
            background: linear-gradient(135deg, #0066cc 0%, #001f4d 50%, #003366 100%) !important;
        }
        
        .stButton > button:active {
            transform: translateY(0px) !important;
        }
        
        /* Result styling */
        .price-result {
            background: linear-gradient(135deg, var(--dark-bg) 0%, var(--darker-bg) 100%);
            border: 2px solid var(--accent-blue);
            border-radius: 15px;
            padding: 25px;
            margin: 20px 0;
            box-shadow: 0 8px 32px rgba(0, 170, 255, 0.1);
            position: relative;
            overflow: hidden;
        }
        
        .price-result::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: var(--gradient-bg);
        }
        
        /* Description section styling */
        .description-section {
            background: linear-gradient(135deg, var(--darker-bg) 0%, var(--dark-bg) 100%);
            border-radius: 12px;
            padding: 20px;
            margin: 20px 0;
            border-left: 4px solid var(--light-blue);
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        }
        
        /* Breakdown section */
        .breakdown-section {
            background: linear-gradient(135deg, var(--darker-bg) 0%, var(--dark-bg) 100%);
            border-radius: 8px;
            padding: 15px;
            margin: 15px 0;
            border-left: 3px solid #ff6b35;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
        }
        
        .breakdown-content {
            font-size: 13px;
            line-height: 1.5;
        }
    </style>
""", unsafe_allow_html=True)

# === Sidebar with project description ===
with st.sidebar:
    st.markdown(
        """
        <p style="font-size: 18px; font-weight: bold; color: #ffffff; margin-bottom: 10px;">
        Project Description
        </p>
        """,
        unsafe_allow_html=True
    )
    st.markdown(
        """
        <p class="justified-text">
        This intelligent car price estimator uses advanced algorithms that consider market segments, brand positioning, depreciation curves, mileage patterns, and real-time market conditions.
        </p>
        """,
        unsafe_allow_html=True
    )

# === Main content ===
st.markdown(
    '<div class="navy-ribbon"> Intelligent Car Price Estimator </div>', 
    unsafe_allow_html=True
)

# Filter out numeric values from model column
def filter_string_models(models):
    """Filter out numeric values and keep only string model names"""
    string_models = []
    for model in models:
        model_str = str(model).strip()
        if not model_str.isdigit() and model_str.lower() not in ['nan', 'none', '']:
            try:
                float(model_str)
                continue
            except ValueError:
                string_models.append(model_str)
    return sorted(list(set(string_models)))

# Input controls
st.markdown("### 🔧 Vehicle Details")

# First row
row1_cols = st.columns([2, 2, 1.5, 1.8, 1.5])

make_options = ["Select Make"] + sorted(df['make'].unique())
make = row1_cols[0].selectbox("Make", make_options, index=0)

if make != "Select Make":
    filtered_models = filter_string_models(df['model'].dropna().unique())
    model_options = ["Select Model"] + filtered_models
else:
    model_options = ["Select Model"]
model_input = row1_cols[1].selectbox("Model", model_options, index=0)

age = row1_cols[2].slider("Age (years)", 0, 25, 5)
mileage = row1_cols[3].number_input("Mileage (km)", min_value=0, value=50000, step=500)

engine_options = ["Select Engine CC"] + sorted([str(cc) for cc in df['engine_cc'].dropna().unique()])
engine_cc = row1_cols[4].selectbox("Engine CC", engine_options, index=0)

st.markdown("---")

# Second row
row2_cols = st.columns([2, 2, 2, 2])

fuel_options = ["Select Fuel Type"] + sorted(df['fuel_type'].dropna().unique())
fuel_type = row2_cols[0].selectbox("Fuel Type", fuel_options, index=0)

body_options = ["Select Body Type"] + sorted(df['body'].dropna().unique())
body = row2_cols[1].selectbox("Body Type", body_options, index=0)

transmission = row2_cols[2].selectbox("Transmission", ["Select Transmission", "Manual", "Automatic"], index=0)
assembled = row2_cols[3].selectbox("Assembled", ["Select Assembly", "Local", "Imported"], index=0)

popular_makes = ["toyota", "suzuki", "honda"]
is_popular_make = 1 if make.lower() in popular_makes else 0

# Submit button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    all_selected = (
        make != "Select Make" and 
        model_input != "Select Model" and 
        engine_cc != "Select Engine CC" and 
        fuel_type != "Select Fuel Type" and 
        body != "Select Body Type" and 
        transmission != "Select Transmission" and 
        assembled != "Select Assembly"
    )
    
    if all_selected:
        submit_btn = st.button("🔍 Get Intelligent Price Analysis", key="submit_btn")
    else:
        st.info("⚠️ Please select all vehicle details to get price analysis")
        submit_btn = False

if submit_btn:
    with st.spinner('Running advanced market analysis and AI algorithms...'):
        
        try:
            engine_cc_value = int(float(engine_cc)) if engine_cc != "Select Engine CC" else 1000
        except ValueError:
            engine_cc_value = 1000
        
        raw_input = {
            'make': correct_input(make, df['make'].unique()),
            'model': correct_input(model_input, df['model'].unique()),
            'age': age,
            'mileage': mileage,
            'mileage_per_year': mileage / age if age != 0 else mileage,
            'engine_cc': engine_cc_value,
            'engine_cc_per_age': engine_cc_value / age if age != 0 else engine_cc_value,
            'is_popular_make': is_popular_make,
            'is_automatic': 1 if transmission == "Automatic" else 0,
            'is_imported': 1 if assembled == "Imported" else 0,
            'fuel_type': correct_input(fuel_type, df['fuel_type'].unique()),
            'body': correct_input(body, df['body'].unique())
        }
        
        # Calculate prices
        ml_predicted_price = calculate_base_price_by_segments(
            raw_input['make'], 
            raw_input['model'], 
            raw_input['engine_cc'], 
            raw_input['age']
        )

        lower, upper = calculate_intelligent_range(raw_input, ml_predicted_price, df)
        
        market_factor = get_market_condition_factor(raw_input['make'], raw_input['model'], raw_input['age'])
        lower = int(lower * market_factor)
        upper = int(upper * market_factor)

        def format_price(n):
            return f"{n:,}"

        def format_k(n):
            return f"{n//1000}k"

        # Display price result
        st.markdown(
            f"""
            <div class="price-result">
                <h2 style='color: #00aaff; margin-bottom: 15px; text-align: center;'>
                     Intelligent Market Valuation 💵
                </h2>
                <h3 style='color: #ffffff; text-align: center; font-size: 24px; margin: 0;'>
                    PKR. {format_price(lower)} - PKR. {format_price(upper)}
                </h3>
                <p style='color: #e0e0e0; text-align: center; margin: 10px 0 0 0; font-size: 16px;'>
                    ({format_k(lower)} - {format_k(upper)})
                </p>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        # Price breakdown
        base_price = calculate_base_price_by_segments(raw_input['make'], raw_input['model'], raw_input['engine_cc'], raw_input['age'])
        avg_annual_mileage = 15000
        expected_mileage = raw_input['age'] * avg_annual_mileage
        
        available_columns = list(df.columns)
        has_price_col = 'price' in available_columns
        
        st.markdown(
            f"""
            <div class="breakdown-section">
                <h3 style='color: #ff6b35; margin-bottom: 12px; font-size: 18px;'>
                    📊 Price Analysis
                </h3>
                <div class='breakdown-content' style='color: #e0e0e0;'>
                    <strong>Base Price:</strong> Rs. {format_price(base_price)} | 
                    <strong>Intelligent Prediction:</strong> Rs. {format_price(int(ml_predicted_price))}<br>
                    <strong>Expected Mileage:</strong> {format_price(expected_mileage)} km | 
                    <strong>Actual:</strong> {format_price(raw_input['mileage'])} km 
                    {'⚠️' if raw_input['mileage'] > expected_mileage * 1.2 else '✅' if raw_input['mileage'] <= expected_mileage * 1.1 else '⭐'}<br>
                    <strong>Segment:</strong> {'Luxury' if raw_input['make'].lower() in ['mercedes', 'bmw', 'audi'] else 'Premium' if raw_input['make'].lower() in ['toyota', 'honda', 'nissan'] else 'Economy'} | 
                    <strong>Depreciation:</strong> {100 - int((lower + upper) / 2 / base_price * 100)}% | 
                    <strong>Data Source:</strong> {'Market data' if has_price_col else 'Calculated estimates'}
                </div>
            </div>
            """, 
            unsafe_allow_html=True
        )

        # AI Description generation
        description, error = generate_ai_description(raw_input, lower, upper, expected_mileage, api_key)
        
        if description:
            # Display AI-generated description
            st.markdown(
                f"""
                <div class="description-section">
                    <h3 style='color: #00aaff; margin-bottom: 15px;'>
                      📑 AI Market Valuation Report
                    </h3>
                    <p style='color: #e0e0e0; line-height: 1.7; font-size: 15px;'>
                        {description}
                    </p>
                </div>
                """, 
                unsafe_allow_html=True
            )
        else:
            # Show error details and fallback
            st.warning(f"🤖 AI Description unavailable: {error}")
            
            # Enhanced fallback description
            segment = 'Luxury' if raw_input['make'].lower() in ['mercedes', 'bmw', 'audi', 'lexus', 'bentley'] else 'Premium' if raw_input['make'].lower() in ['toyota', 'honda', 'nissan'] else 'Economy'
            
            efficiency_desc = 'excellent fuel efficiency' if raw_input['engine_cc'] < 1500 else 'balanced performance and efficiency' if raw_input['engine_cc'] <= 2000 else 'strong performance characteristics'
            
            mileage_status = 'low mileage' if raw_input['mileage'] < expected_mileage * 0.8 else 'average mileage' if raw_input['mileage'] <= expected_mileage * 1.2 else 'higher mileage'
            
            transmission_benefit = 'convenience and smooth driving experience' if raw_input['is_automatic'] else 'better fuel economy and lower maintenance costs'
            
            fallback_description = f"""
            This {2024 - raw_input['age']} {raw_input['make']} {raw_input['model']} represents a solid choice in the {segment.lower()} segment of Pakistan's used car market. 
            The {raw_input['engine_cc']}cc {raw_input['fuel_type']} engine offers {efficiency_desc}, making it suitable for both city and highway driving. 
            With {mileage_status} at {raw_input['mileage']:,} km, this vehicle shows {'excellent' if mileage_status == 'low mileage' else 'reasonable'} usage patterns for its age. 
            The {'automatic' if raw_input['is_automatic'] else 'manual'} transmission provides {transmission_benefit}. 
            Current market valuation reflects the model's reputation for {'luxury and prestige' if segment == 'Luxury' else 'reliability and resale value' if segment == 'Premium' else 'affordability and practicality'}, 
            positioning it competitively within the Rs. {lower//1000}k-{upper//1000}k price range based on similar vehicles in the market.
            """
        
            st.markdown(
                f"""
                <div class="description-section">
                    <h3 style='color: #00aaff; margin-bottom: 15px;'>
                      📑 Market Valuation Report
                    </h3>
                    <p style='color: #e0e0e0; line-height: 1.7; font-size: 15px;'>
                        {fallback_description}
                    </p>
                </div>
                """, 
                unsafe_allow_html=True
            )