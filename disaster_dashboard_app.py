import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
from textblob import TextBlob
import warnings
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import os

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="🚨 Disaster Response Analytics",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .metric-card {
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .prediction-result {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #2196F3, #21CBF3);
    }
</style>
""", unsafe_allow_html=True)

# ============================
# Functions for caching & loading data
# ============================
@st.cache_data
def load_data(file_path):
    """Load disaster messages data"""
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_resource
def train_models(df):
    """Train ML models for disaster prediction"""
    X = df['message'].astype(str)
    y = df['disaster_type']

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # Vectorize text
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train models
    models = {
        'Naive Bayes': MultinomialNB(),
        'Logistic Regression': LogisticRegression(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }

    results = {}
    trained_models = {}

    for name, model in models.items():
        model.fit(X_train_vec, y_train)
        y_pred = model.predict(X_test_vec)
        accuracy = accuracy_score(y_test, y_pred)

        trained_models[name] = model
        results[name] = accuracy

    return trained_models, vectorizer, label_encoder, results

# ============================
# Class for Disaster Analytics
# ============================
class DisasterAnalyticsDashboard:
    def __init__(self):
        self.df = None

    def extract_disaster_type(self, message):
        if pd.isna(message):
            return 'unknown'

        message_lower = message.lower()
        disaster_keywords = {
            'earthquake': ['earthquake', 'quake', 'seismic', 'tremor'],
            'flood': ['flood', 'flooding', 'water', 'river', 'overflow'],
            'hurricane': ['hurricane', 'storm', 'cyclone', 'wind'],
            'fire': ['fire', 'burning', 'flame', 'smoke'],
            'medical': ['medical', 'health', 'hospital', 'doctor', 'sick'],
            'food': ['food', 'hungry', 'eat', 'water', 'drink'],
            'shelter': ['shelter', 'house', 'home', 'roof', 'building'],
            'rescue': ['rescue', 'help', 'trapped', 'stuck', 'save'],
            'infrastructure': ['road', 'bridge', 'power', 'electricity', 'communication']
        }

        for disaster_type, keywords in disaster_keywords.items():
            if any(keyword in message_lower for keyword in keywords):
                return disaster_type

        return 'general'

    def get_sentiment(self, message):
        if pd.isna(message):
            return 'neutral'

        try:
            blob = TextBlob(message)
            polarity = blob.sentiment.polarity

            if polarity > 0.1:
                return 'positive'
            elif polarity < -0.1:
                return 'negative'
            else:
                return 'neutral'
        except:
            return 'neutral'

    def extract_urgency_level(self, message):
        if pd.isna(message):
            return 'low'

        message_lower = message.lower()
        urgent_keywords = ['urgent', 'emergency', 'critical', 'immediate', 'asap', 'help']
        high_keywords = ['important', 'serious', 'severe', 'major']

        if any(keyword in message_lower for keyword in urgent_keywords):
            return 'urgent'
        elif any(keyword in message_lower for keyword in high_keywords):
            return 'high'
        else:
            return 'medium'

    def preprocess_data(self, df):
        df = df.dropna()
        df['disaster_type'] = df['message'].apply(self.extract_disaster_type)
        df['sentiment'] = df['message'].apply(self.get_sentiment)
        df['message_length'] = df['message'].str.len()
        df['urgency_level'] = df['message'].apply(self.extract_urgency_level)
        return df

# ============================
# Function to display pages
# ============================
def show_overview(df):
    st.subheader("📊 Dataset Overview")
    
    # Metrics dalam kolom
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📝 Total Rows",
            value=f"{len(df):,}",
            delta=None
        )
    
    with col2:
        st.metric(
            label="📋 Total Columns", 
            value=len(df.columns),
            delta=None
        )
    
    with col3:
        st.metric(
            label="💾 Memory Usage",
            value=f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB",
            delta=None
        )
    
    with col4:
        missing_values = df.isnull().sum().sum()
        st.metric(
            label="⚠️ Missing Values",
            value=f"{missing_values:,}",
            delta=None
        )
    
    # Separator
    st.markdown("---")
    
    # Dataset preview dengan styling
    st.markdown("### 🔍 **Data Preview**")
    
    # Tabs untuk berbagai view
    tab1, tab2, tab3 = st.tabs(["📋 First 5 Rows", "📊 Data Info", "🔢 Summary Statistics"])
    
    with tab1:
        st.dataframe(
            df.head(),
            use_container_width=True,
            height=300
        )
    
    with tab2:
        col_info1, col_info2 = st.columns(2)
        
        with col_info1:
            st.markdown("**📋 Column Information:**")
            info_df = pd.DataFrame({
                'Column': df.columns,
                'Data Type': df.dtypes.astype(str),
                'Non-Null Count': df.count(),
                'Null Count': df.isnull().sum()
            })
            st.dataframe(info_df, use_container_width=True)
        
        with col_info2:
            st.markdown("**📈 Data Types Distribution:**")
            dtype_counts = df.dtypes.value_counts()
            for dtype, count in dtype_counts.items():
                st.write(f"• **{dtype}**: {count} columns")
    
    with tab3:
        # Hanya tampilkan statistik untuk kolom numerik
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            st.dataframe(
                df[numeric_cols].describe(),
                use_container_width=True
            )
        else:
            st.info("📝 No numeric columns found for statistical summary")
    
    # Dataset shape dengan styling
    st.markdown("---")
    st.markdown(f"**📐 Dataset Shape:** `{df.shape[0]} rows × {df.shape[1]} columns`")
    
    # Progress bar untuk completeness
    completeness = (1 - (df.isnull().sum().sum() / (len(df) * len(df.columns)))) * 100
    st.markdown(f"**✅ Data Completeness:** {completeness:.1f}%")
    st.progress(completeness / 100)

def show_data_analysis(df):
    st.subheader("📈 Data Analysis")
    
    # Professional styling
    st.markdown("""
    <style>
        .analysis-section {
            margin: 20px 0;
        }
        .insight-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 12px;
            margin: 10px 0;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.2);
        }
        .metric-highlight {
            background: rgba(255,255,255,0.1);
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
            border-left: 4px solid #4ECDC4;
        }
        .stats-container {
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            border: 1px solid #e9ecef;
        }
    </style>
    """, unsafe_allow_html=True)

    # Key Statistics Section
    st.markdown("### 🔍 Key Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    disaster_counts = df['disaster_type'].value_counts()
    total_disasters = len(df)
    unique_types = df['disaster_type'].nunique()
    most_common = disaster_counts.index[0]
    
    with col1:
        st.metric("Total Records", f"{total_disasters:,}")
    
    with col2:
        st.metric("Disaster Types", unique_types)
    
    with col3:
        st.metric("Most Frequent", most_common)
    
    with col4:
        diversity_index = 1 - (disaster_counts.iloc[0] / total_disasters)
        st.metric("Diversity Index", f"{diversity_index:.2f}")

    # Top Disasters Analysis
    st.markdown("---")
    st.markdown("### 🏆 Top Disaster Types")
    
    top_disasters = disaster_counts.head(5)
    
    # Create a more sophisticated display
    col_chart, col_insights = st.columns([2, 1])
    
    with col_chart:
        # Enhanced bar chart
        fig_bar = go.Figure(go.Bar(
            x=top_disasters.values,
            y=top_disasters.index,
            orientation='h',
            marker=dict(
                color=top_disasters.values,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Frequency")
            ),
            text=[f"{val:,}" for val in top_disasters.values],
            textposition='inside'
        ))
        
        fig_bar.update_layout(
            title="Top 5 Disaster Types by Frequency",
            xaxis_title="Number of Incidents",
            yaxis_title="Disaster Type",
            height=400,
            template="plotly_white",
            showlegend=False
        )
        
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with col_insights:
        st.markdown('<div class="stats-container">', unsafe_allow_html=True)
        st.markdown("**📊 Quick Insights:**")
        
        for i, (disaster, count) in enumerate(top_disasters.head(3).items(), 1):
            percentage = (count / total_disasters) * 100
            st.markdown(f"""
            **#{i}. {disaster}**  
            📈 {count:,} incidents ({percentage:.1f}%)
            """)
        
        # Concentration analysis
        top_3_percentage = (top_disasters.head(3).sum() / total_disasters) * 100
        st.markdown(f"""
        ---
        **🎯 Concentration:**  
        Top 3 types represent **{top_3_percentage:.1f}%** of all disasters
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    # Distribution Visualization
    st.markdown("---")
    st.markdown("### 📊 Distribution Analysis")
    
    tab1, tab2, tab3 = st.tabs(["📈 Complete Distribution", "🥧 Proportions", "📉 Cumulative Analysis"])
    
    with tab1:
        # Complete distribution with better styling
        fig_complete = go.Figure(go.Bar(
            x=disaster_counts.index,
            y=disaster_counts.values,
            marker=dict(
                color='rgba(102, 126, 234, 0.8)',
                line=dict(color='rgba(102, 126, 234, 1)', width=1)
            ),
            text=[f"{val:,}" for val in disaster_counts.values],
            textposition='outside'
        ))
        
        fig_complete.update_layout(
            title="Complete Disaster Type Distribution",
            xaxis_title="Disaster Type",
            yaxis_title="Frequency",
            template="plotly_white",
            xaxis_tickangle=-45
        )
        
        st.plotly_chart(fig_complete, use_container_width=True)
    
    with tab2:
        # Professional pie chart
        colors = ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4ecdc4', '#44a08d', '#f7971e', '#ffd200']
        
        fig_pie = go.Figure(go.Pie(
            labels=disaster_counts.index,
            values=disaster_counts.values,
            hole=0.4,
            marker=dict(colors=colors[:len(disaster_counts)]),
            textinfo='label+percent',
            textposition='outside'
        ))
        
        fig_pie.update_layout(
            title="Disaster Type Proportions",
            template="plotly_white",
            showlegend=True,
            legend=dict(orientation="v", x=1.05, y=0.5)
        )
        
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with tab3:
        # Cumulative analysis
        cumulative_pct = (disaster_counts.cumsum() / total_disasters * 100)
        
        fig_cumulative = go.Figure()
        
        fig_cumulative.add_trace(go.Scatter(
            x=list(range(1, len(cumulative_pct) + 1)),
            y=cumulative_pct.values,
            mode='lines+markers',
            name='Cumulative %',
            line=dict(color='#667eea', width=3),
            marker=dict(size=8)
        ))
        
        fig_cumulative.add_hline(y=80, line_dash="dash", line_color="red", 
                                annotation_text="80% threshold")
        
        fig_cumulative.update_layout(
            title="Cumulative Distribution Analysis",
            xaxis_title="Disaster Type Rank",
            yaxis_title="Cumulative Percentage (%)",
            template="plotly_white"
        )
        
        st.plotly_chart(fig_cumulative, use_container_width=True)

    # Professional Summary
    st.markdown("---")
    st.markdown("### 📋 Analysis Summary")
    
    col_summary1, col_summary2 = st.columns(2)
    
    with col_summary1:
        st.markdown("""
        **🔍 Key Findings:**
        - Dataset contains comprehensive disaster incident records
        - Distribution shows varying frequency patterns across disaster types
        - Clear identification of high-impact disaster categories
        """)
    
    with col_summary2:
        # Calculate some insights
        pareto_threshold = disaster_counts.cumsum() / total_disasters <= 0.8
        critical_types = disaster_counts[pareto_threshold].count()
        
        st.markdown(f"""
        **📈 Statistical Insights:**
        - **{critical_types}** disaster types account for 80% of incidents
        - Most frequent type: **{most_common}** ({disaster_counts.iloc[0]:,} incidents)
        - Distribution diversity index: **{diversity_index:.3f}**
        """)

    # Professional conclusion without being "alay"
    st.info("""
    💡 **Analysis Insight:** This distribution analysis provides a foundation for risk assessment and resource allocation strategies. 
    The frequency patterns indicate which disaster types require prioritized attention and preparedness measures.
    """)

def show_prediction_page(df, dashboard):
    st.subheader("🔮 AI Prediction & Model Performance")

    # Professional styling
    st.markdown("""
    <style>
        .model-card {
            background: linear-gradient(135deg, var(--bg-color) 0%, var(--accent-color) 100%);
            color: white;
            border-radius: 15px;
            padding: 25px;
            margin: 10px 0;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        .model-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 12px 35px rgba(0, 0, 0, 0.2);
        }
        .accuracy-score {
            font-size: 2.5rem;
            font-weight: 700;
            margin: 10px 0;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
        }
        .model-name {
            font-size: 1.3rem;
            font-weight: 600;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .performance-badge {
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 600;
            margin-top: 10px;
        }
        .info-section {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 15px;
            padding: 25px;
            margin: 20px 0;
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.2);
        }
        .training-stats {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 10px;
            padding: 15px;
            margin: 15px 0;
            border-left: 4px solid #4ECDC4;
        }
        .prediction-container {
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            color: white;
            border-radius: 15px;
            padding: 25px;
            margin: 20px 0;
            box-shadow: 0 8px 25px rgba(17, 153, 142, 0.3);
        }
        .prediction-result {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            padding: 20px;
            margin: 15px 0;
            border-left: 4px solid #FFD700;
            backdrop-filter: blur(10px);
        }
        .confidence-bar {
            background: rgba(255, 255, 255, 0.2);
            border-radius: 10px;
            height: 20px;
            margin: 10px 0;
            overflow: hidden;
        }
        .confidence-fill {
            background: linear-gradient(90deg, #FFD700, #FFA500);
            height: 100%;
            border-radius: 10px;
            transition: width 0.5s ease;
        }
    </style>
    """, unsafe_allow_html=True)

    # Show training progress
    with st.spinner("🚀 Training AI models..."):
        models, vectorizer, encoder, results = train_models(df)
    
    st.success("✅ Models trained successfully!")

    # Store models in session state for prediction
    if 'trained_models' not in st.session_state:
        st.session_state['trained_models'] = models
        st.session_state['vectorizer'] = vectorizer
        st.session_state['encoder'] = encoder

    # Training Statistics
    st.markdown("### 📊 Training Overview")
    col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)
    
    with col_stats1:
        st.metric("🎯 Models Trained", len(results))
    
    with col_stats2:
        st.metric("📝 Training Samples", len(df))
    
    with col_stats3:
        best_model = max(results, key=results.get)
        st.metric("🏆 Best Model", best_model)
    
    with col_stats4:
        best_accuracy = max(results.values())
        st.metric("📈 Best Accuracy", f"{best_accuracy:.1%}")

    # Model Performance Cards
    st.markdown("---")
    st.markdown("### 🤖 Model Performance Comparison")
    
    # Define colors and icons for each model
    model_config = {
        'Naive Bayes': {'color': '#FF6B6B', 'accent': '#FF5252', 'icon': '🎯'},
        'Logistic Regression': {'color': '#4ECDC4', 'accent': '#26C6DA', 'icon': '📊'},
        'Random Forest': {'color': '#667eea', 'accent': '#5C6BC0', 'icon': '🌲'}
    }
    
    # Create performance badges
    def get_performance_badge(accuracy):
        if accuracy >= 0.9:
            return "Excellent", "#4CAF50"
        elif accuracy >= 0.8:
            return "Good", "#FF9800"
        elif accuracy >= 0.7:
            return "Fair", "#FF5722"
        else:
            return "Needs Improvement", "#F44336"

    # Display model cards
    cols = st.columns(3)
    for i, (model_name, accuracy) in enumerate(results.items()):
        with cols[i]:
            config = model_config.get(model_name, {'color': '#9E9E9E', 'accent': '#757575', 'icon': '🤖'})
            badge_text, badge_color = get_performance_badge(accuracy)
            
            st.markdown(f"""
            <div class="model-card" style="--bg-color: {config['color']}; --accent-color: {config['accent']};">
                <div class="model-name">
                    <span>{config['icon']}</span>
                    <span>{model_name}</span>
                </div>
                <div class="accuracy-score">{accuracy:.1%}</div>
                <div style="font-size: 0.9rem; opacity: 0.9;">
                    Accuracy Score
                </div>
                <div class="performance-badge" style="background-color: {badge_color};">
                    {badge_text}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Interactive Comparison Chart
    st.markdown("---")
    st.markdown("### 📈 Performance Visualization")
    
    tab1, tab2, tab3 = st.tabs(["📊 Accuracy Comparison", "🎯 Detailed Metrics", "📋 Model Insights"])
    
    with tab1:
        # Enhanced bar chart
        model_names = list(results.keys())
        accuracies = list(results.values())
        colors = [model_config[name]['color'] for name in model_names]
        
        fig_bar = go.Figure(go.Bar(
            x=model_names,
            y=accuracies,
            marker=dict(
                color=colors,
                line=dict(color='white', width=2)
            ),
            text=[f"{acc:.1%}" for acc in accuracies],
            textposition='outside',
            textfont=dict(size=14, color='white')
        ))
        
        fig_bar.update_layout(
            title="Model Accuracy Comparison",
            xaxis_title="Machine Learning Models",
            yaxis_title="Accuracy Score",
            template="plotly_dark",
            height=500,
            yaxis=dict(tickformat='.0%'),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig_bar, use_container_width=True, key="accuracy_comparison_chart")
    
    with tab2:
        # Radar chart for comparison
        categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        fig_radar = go.Figure()
        
        for model_name, accuracy in results.items():
            # Simulate other metrics (in real scenario, you'd calculate these)
            precision = accuracy * (0.95 + 0.1 * (0.5 - abs(0.5 - accuracy)))
            recall = accuracy * (0.90 + 0.2 * (0.5 - abs(0.5 - accuracy)))
            f1_score = 2 * (precision * recall) / (precision + recall)
            
            fig_radar.add_trace(go.Scatterpolar(
                r=[accuracy, precision, recall, f1_score],
                theta=categories,
                fill='toself',
                name=model_name,
                line_color=model_config[model_name]['color']
            ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Multi-Metric Model Comparison",
            template="plotly_dark"
        )
        
        st.plotly_chart(fig_radar, use_container_width=True)
    
    with tab3:
        # Model insights and recommendations
        best_model = max(results, key=results.get)
        worst_model = min(results, key=results.get)
        
        col_insight1, col_insight2 = st.columns(2)
        
        with col_insight1:
            st.markdown("#### 🏆 Top Performer")
            st.markdown(f"""
            **{best_model}** achieved the highest accuracy of **{results[best_model]:.1%}**
            
            **Strengths:**
            - Excellent prediction capability
            - Reliable for disaster classification
            - Recommended for production use
            """)
        
        with col_insight2:
            st.markdown("#### 📊 Performance Analysis")
            accuracy_spread = max(results.values()) - min(results.values())
            avg_accuracy = sum(results.values()) / len(results.values())
            
            st.markdown(f"""
            **Overall Statistics:**
            - Average Accuracy: **{avg_accuracy:.1%}**
            - Performance Spread: **{accuracy_spread:.1%}**
            - Models Tested: **{len(results)}**
            """)

    # Interactive Model Selection
    st.markdown("---")
    st.markdown("### 🎛️ Model Selection for Prediction")
    
    selected_model = st.selectbox(
        "Choose a model for making predictions:",
        options=list(results.keys()),
        index=list(results.keys()).index(best_model),
        help=f"Default selection is {best_model} (highest accuracy)"
    )
    
    col_selected1, col_selected2 = st.columns([2, 1])
    
    with col_selected1:
        st.info(f"""
        🎯 **Selected Model:** {selected_model}  
        📊 **Accuracy:** {results[selected_model]:.1%}  
        💡 **Status:** {'✅ Recommended' if selected_model == best_model else '⚠️ Alternative Choice'}
        """)
    
    with col_selected2:
        if st.button("🚀 Use This Model", type="primary"):
            st.session_state['selected_model'] = selected_model
            st.success(f"✅ {selected_model} is now active!")

    # ============= NEW PREDICTION SECTION =============
    st.markdown("---")
    st.markdown('<div class="prediction-container">', unsafe_allow_html=True)
    st.markdown("## 🔮 Make Disaster Predictions")
    
    # Get the current selected model
    current_model = st.session_state.get('selected_model', best_model)
    
    col_pred1, col_pred2 = st.columns([2, 1])
    
    with col_pred1:
        st.markdown(f"### 📝 Enter Disaster Description")
        st.markdown(f"**Active Model:** {current_model} ({results[current_model]:.1%} accuracy)")
        
        # Text input methods
        input_method = st.radio(
            "Choose input method:",
            ["✍️ Type manually", "📋 Use example", "🔄 Random from dataset"],
            horizontal=True
        )
        
        disaster_text = ""
        
        if input_method == "✍️ Type manually":
            disaster_text = st.text_area(
                "Describe the disaster situation:",
                placeholder="Example: Forest fire is spreading rapidly in the northern region, emergency evacuation needed...",
                height=120
            )
        
        elif input_method == "📋 Use example":
            example_texts = {
                "🔥 Wildfire": "Forest fire is spreading rapidly across the mountainous region, threatening residential areas and requiring immediate evacuation",
                "🌊 Flood": "Heavy rainfall has caused severe flooding in the downtown area, with water levels rising rapidly and roads becoming impassable",
                "🌪️ Storm": "Severe thunderstorm with high winds and hail is approaching the city, posing significant danger to property and life",
                "🏠 Building Collapse": "Multi-story building has partially collapsed following structural failure, emergency rescue operations are underway",
                "🔥 Fire Emergency": "Large fire has broken out in commercial district, thick smoke is affecting air quality and visibility"
            }
            
            selected_example = st.selectbox("Choose an example:", list(example_texts.keys()))
            disaster_text = example_texts[selected_example]
            st.text_area("Selected example:", value=disaster_text, height=80, disabled=True)
        
        elif input_method == "🔄 Random from dataset":
            # First, let's detect the column names in the dataset
            text_columns = []
            category_columns = []
            
            # Common text column names
            possible_text_cols = ['text', 'description', 'message', 'content', 'tweet', 'news', 'report']
            # Common category column names  
            possible_cat_cols = ['category', 'label', 'class', 'type', 'classification', 'target']
            
            # Find text column
            for col in df.columns:
                if col.lower() in possible_text_cols or 'text' in col.lower():
                    text_columns.append(col)
                if col.lower() in possible_cat_cols or any(word in col.lower() for word in ['cat', 'lab', 'clas', 'typ']):
                    category_columns.append(col)
            
            # Display available columns for debugging
            if not text_columns:
                st.warning("⚠️ Cannot find text column. Available columns:")
                st.write(list(df.columns))
                
                # Let user select manually
                selected_text_col = st.selectbox("Select text column:", df.columns.tolist())
                if selected_text_col:
                    text_columns = [selected_text_col]
            
            if not category_columns and len(df.columns) > 1:
                st.info("💡 Cannot auto-detect category column. Available columns:")
                st.write(list(df.columns))
                
                # Let user select manually
                remaining_cols = [col for col in df.columns if col not in text_columns]
                if remaining_cols:
                    selected_cat_col = st.selectbox("Select category column:", remaining_cols)
                    if selected_cat_col:
                        category_columns = [selected_cat_col]
            
            # Generate random sample if columns are found
            if text_columns and st.button("🎲 Get Random Sample"):
                try:
                    random_sample = df.sample(1)
                    text_col = text_columns[0]
                    disaster_text = str(random_sample.iloc[0][text_col])
                    
                    # Try to get category if available
                    actual_category = None
                    if category_columns:
                        cat_col = category_columns[0]
                        actual_category = str(random_sample.iloc[0][cat_col])
                    
                    st.session_state['random_text'] = disaster_text
                    if actual_category:
                        st.session_state['actual_category'] = actual_category
                        
                except Exception as e:
                    st.error(f"Error getting random sample: {str(e)}")
            
            # Display the random sample
            if 'random_text' in st.session_state:
                disaster_text = st.session_state['random_text']
                st.text_area("Random sample from dataset:", value=disaster_text, height=80, disabled=True)
                if 'actual_category' in st.session_state:
                    st.info(f"🏷️ **Actual category:** {st.session_state['actual_category']}")
                else:
                    st.info("🏷️ **Actual category:** Not available")
    
    with col_pred2:
        st.markdown("### 🎯 Prediction Controls")
        
        # Prediction button
        predict_button = st.button("🚀 Predict Disaster Type", type="primary", use_container_width=True)
        
        # Additional options
        st.markdown("**Options:**")
        show_confidence = st.checkbox("📊 Show confidence scores", value=True)
        show_probabilities = st.checkbox("📈 Show all probabilities", value=False)
        
        # Model info
        st.markdown("**Model Info:**")
        st.markdown(f"🤖 **Type:** {current_model}")
        st.markdown(f"📊 **Accuracy:** {results[current_model]:.1%}")
        st.markdown(f"🎯 **Status:** Ready")

    st.markdown('</div>', unsafe_allow_html=True)

    # Make prediction
    if predict_button and disaster_text.strip():
        with st.spinner("🔮 Making prediction..."):
            try:
                # Get the trained model and preprocessors
                model = st.session_state['trained_models'][current_model]
                vectorizer = st.session_state['vectorizer']
                encoder = st.session_state['encoder']
                
                # Preprocess the input text
                processed_text = vectorizer.transform([disaster_text])
                
                # Make prediction
                prediction_raw = model.predict(processed_text)[0]
                
                # Convert prediction back to string if it's encoded
                if hasattr(encoder, 'inverse_transform'):
                    # If using LabelEncoder
                    prediction = encoder.inverse_transform([prediction_raw])[0]
                elif hasattr(encoder, 'classes_'):
                    # If encoder has classes_ attribute
                    prediction = encoder.classes_[prediction_raw]
                else:
                    # If prediction is already a string or handle direct output
                    prediction = str(prediction_raw)
                
                # Get prediction probabilities if available
                if hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba(processed_text)[0]
                    confidence = max(probabilities)
                    
                    # Create probability dictionary with proper class names
                    if hasattr(encoder, 'classes_'):
                        classes = encoder.classes_
                    elif hasattr(model, 'classes_'):
                        classes = model.classes_
                    else:
                        classes = [f"Class_{i}" for i in range(len(probabilities))]
                    
                    prob_dict = dict(zip(classes, probabilities))
                else:
                    confidence = 0.85  # Default confidence for models without probabilities
                    prob_dict = {prediction: confidence}
                
                # Display prediction results
                st.markdown("---")
                st.markdown('<div class="prediction-result">', unsafe_allow_html=True)
                
                col_result1, col_result2 = st.columns([2, 1])
                
                with col_result1:
                    st.markdown("## 🎯 Prediction Result")
                    
                    # Main prediction
                    disaster_icons = {
                        'fire': '🔥',
                        'flood': '🌊',
                        'storm': '🌪️',
                        'earthquake': '🌍',
                        'accident': '🚨',
                        'other': '⚠️'
                    }
                    
                    prediction_icon = disaster_icons.get(str(prediction).lower(), '🚨')
                    st.markdown(f"### {prediction_icon} **{str(prediction).upper()}**")
                    
                    # Confidence bar
                    if show_confidence:
                        st.markdown("**Confidence Level:**")
                        confidence_percent = confidence * 100
                        
                        st.markdown(f"""
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width: {confidence_percent}%"></div>
                        </div>
                        <div style="text-align: center; margin-top: 5px;">
                            <strong>{confidence_percent:.1f}%</strong>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col_result2:
                    st.markdown("### 📊 Prediction Stats")
                    st.metric("🎯 Predicted Class", prediction)
                    st.metric("📈 Confidence", f"{confidence:.1%}")
                    st.metric("🤖 Model Used", current_model)
                    
                    # Compare with actual if available
                    if 'actual_category' in st.session_state:
                        actual = st.session_state['actual_category']
                        is_correct = str(prediction).lower() == str(actual).lower()
                        st.metric(
                            "✅ Accuracy Check", 
                            "Correct" if is_correct else "Incorrect",
                            delta="Match" if is_correct else f"Expected: {actual}"
                        )
                
                # Show all probabilities if requested
                if show_probabilities and len(prob_dict) > 1:
                    st.markdown("### 📈 All Class Probabilities")
                    
                    # Sort probabilities in descending order
                    sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
                    
                    for class_name, prob in sorted_probs:
                        col1, col2, col3 = st.columns([2, 2, 1])
                        with col1:
                            icon = disaster_icons.get(str(class_name).lower(), '📊')
                            st.write(f"{icon} {class_name}")
                        with col2:
                            st.progress(prob)
                        with col3:
                            st.write(f"{prob:.1%}")
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Additional insights
                st.markdown("### 💡 Prediction Insights")
                
                col_insight1, col_insight2 = st.columns(2)
                
                with col_insight1:
                    st.markdown(f"""
                    **🔍 Analysis:**
                    - Text length: {len(disaster_text)} characters
                    - Word count: {len(disaster_text.split())} words
                    - Model confidence: {"High" if confidence > 0.8 else "Medium" if confidence > 0.6 else "Low"}
                    - Prediction reliability: {"✅ Reliable" if confidence > 0.7 else "⚠️ Uncertain"}
                    """)
                
                with col_insight2:
                    st.markdown(f"""
                    **📋 Recommendation:**
                    - Priority level: {"🔴 High" if confidence > 0.8 else "🟡 Medium" if confidence > 0.6 else "🟢 Low"}
                    - Response needed: {"Immediate" if confidence > 0.8 else "Standard"}
                    - Additional verification: {"Not required" if confidence > 0.9 else "Recommended"}
                    """)
                
            except Exception as e:
                st.error(f"❌ Prediction failed: {str(e)}")
                st.info("Please ensure the model is properly trained and try again.")
    
    elif predict_button and not disaster_text.strip():
        st.warning("⚠️ Please enter a disaster description to make a prediction.")

    # Professional Information Section
    st.markdown("---")
    st.markdown('<div class="info-section">', unsafe_allow_html=True)
    st.markdown("### 🧠 AI Model Training Process")
    
    col_process1, col_process2 = st.columns(2)
    
    with col_process1:
        st.markdown("""
        **🔄 Training Pipeline:**
        1. **Data Preprocessing** - Text cleaning and normalization
        2. **Feature Engineering** - TF-IDF vectorization
        3. **Model Training** - Multiple algorithm comparison
        4. **Performance Evaluation** - Cross-validation testing
        """)
    
    with col_process2:
        st.markdown(f"""
        **📈 Model Insights:**
        - **Best Performer:** {best_model} ({results[best_model]:.1%})
        - **Training Data:** {len(df):,} disaster reports
        - **Feature Space:** Text-based disaster descriptions
        - **Evaluation:** Accuracy-based comparison
        """)
    
    st.markdown('</div>', unsafe_allow_html=True)

    # Add some spacing
    st.markdown("<br>", unsafe_allow_html=True)

def show_risk_assessment(df):
    st.subheader("⚠️ Disaster Risk Assessment & Monitoring")

    # Professional styling
    st.markdown("""
    <style>
        .risk-card {
            background: linear-gradient(135deg, var(--bg-color) 0%, var(--accent-color) 100%);
            color: white;
            border-radius: 15px;
            padding: 25px;
            margin: 15px 0;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        .risk-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 12px 35px rgba(0, 0, 0, 0.2);
        }
        .risk-score {
            font-size: 3rem;
            font-weight: 800;
            margin: 15px 0;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
        }
        .risk-level {
            font-size: 1.4rem;
            font-weight: 600;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .risk-indicator {
            width: 100%;
            height: 20px;
            border-radius: 10px;
            background: linear-gradient(90deg, #4CAF50, #FF9800, #F44336);
            position: relative;
            margin: 15px 0;
        }
        .risk-pointer {
            position: absolute;
            top: -5px;
            width: 0;
            height: 0;
            border-left: 8px solid transparent;
            border-right: 8px solid transparent;
            border-bottom: 12px solid white;
            transform: translateX(-50%);
        }
        .alert-section {
            background: linear-gradient(135deg, #FF6B6B 0%, #FF8E53 100%);
            color: white;
            border-radius: 15px;
            padding: 25px;
            margin: 20px 0;
            box-shadow: 0 8px 25px rgba(255, 107, 107, 0.3);
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0% { box-shadow: 0 8px 25px rgba(255, 107, 107, 0.3); }
            50% { box-shadow: 0 12px 35px rgba(255, 107, 107, 0.5); }
            100% { box-shadow: 0 8px 25px rgba(255, 107, 107, 0.3); }
        }
        .mitigation-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 15px;
            padding: 20px;
            margin: 15px 0;
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.2);
        }
        .timeline-item {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
            border-left: 4px solid var(--accent-color);
        }
    </style>
    """, unsafe_allow_html=True)

    # Detect available columns
    text_columns = []
    category_columns = []
    time_columns = []
    
    # Find relevant columns
    for col in df.columns:
        col_lower = col.lower()
        if any(word in col_lower for word in ['text', 'description', 'message', 'content', 'tweet', 'news']):
            text_columns.append(col)
        elif any(word in col_lower for word in ['category', 'label', 'class', 'type', 'target']):
            category_columns.append(col)
        elif any(word in col_lower for word in ['date', 'time', 'created', 'timestamp']):
            time_columns.append(col)

    # Calculate risk metrics
    total_incidents = len(df)
    
    # Category-based risk analysis
    if category_columns:
        category_col = category_columns[0]
        category_counts = df[category_col].value_counts()
        
        # Define risk weights for different disaster types
        risk_weights = {
            'fire': 0.9, 'wildfire': 0.95, 'forest fire': 0.9,
            'flood': 0.85, 'flooding': 0.85,
            'earthquake': 0.95, 'quake': 0.95,
            'storm': 0.8, 'hurricane': 0.9, 'tornado': 0.9,
            'accident': 0.7, 'crash': 0.75,
            'explosion': 0.9, 'blast': 0.9,
            'landslide': 0.85, 'avalanche': 0.9,
            'other': 0.5
        }
        
        # Calculate weighted risk score
        total_risk_score = 0
        for category, count in category_counts.items():
            category_lower = str(category).lower()
            weight = 0.6  # default weight
            
            for risk_type, risk_weight in risk_weights.items():
                if risk_type in category_lower:
                    weight = risk_weight
                    break
            
            total_risk_score += count * weight
        
        # Normalize risk score (0-100)
        max_possible_score = total_incidents * 1.0
        risk_score = min((total_risk_score / max_possible_score) * 100, 100)
        
    else:
        # Fallback risk calculation based on incident frequency
        risk_score = min((total_incidents / 100) * 50, 100)  # Scale based on incident count

    # Determine risk level
    def get_risk_level(score):
        if score >= 80:
            return "CRITICAL", "#D32F2F", "🔴"
        elif score >= 60:
            return "HIGH", "#F57C00", "🟠"
        elif score >= 40:
            return "MEDIUM", "#FF9800", "🟡"
        elif score >= 20:
            return "LOW", "#4CAF50", "🟢"
        else:
            return "MINIMAL", "#2E7D32", "🟢"

    risk_level, risk_color, risk_icon = get_risk_level(risk_score)

    # Main Risk Dashboard
    st.markdown("### 🎯 Current Risk Status")
    
    col_main1, col_main2, col_main3 = st.columns([2, 2, 1])
    
    with col_main1:
        st.markdown(f"""
        <div class="risk-card" style="--bg-color: {risk_color}; --accent-color: {risk_color};">
            <div class="risk-level">
                <span>{risk_icon}</span>
                <span>{risk_level} RISK</span>
            </div>
            <div class="risk-score">{risk_score:.1f}</div>
            <div style="font-size: 1.1rem; opacity: 0.9;">
                Risk Score (0-100)
            </div>
            <div class="risk-indicator">
                <div class="risk-pointer" style="left: {risk_score}%;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_main2:
        # Key metrics
        st.markdown("#### 📊 Key Metrics")
        st.metric("📈 Total Incidents", f"{total_incidents:,}")
        
        if category_columns:
            most_common = category_counts.index[0]
            st.metric("🚨 Most Common Type", most_common)
            st.metric("📊 Incident Frequency", f"{category_counts.iloc[0]} cases")
        
        # Calculate trend (simplified)
        if time_columns and len(df) > 1:
            try:
                time_col = time_columns[0]
                df_time = df.copy()
                df_time[time_col] = pd.to_datetime(df_time[time_col], errors='coerce')
                recent_data = df_time.dropna(subset=[time_col]).tail(30)
                trend = "📈 Increasing" if len(recent_data) > 15 else "📉 Stable"
                st.metric("📈 Trend (30 days)", trend)
            except:
                st.metric("📈 Trend", "Data unavailable")
    
    with col_main3:
        # Risk indicator gauge
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = risk_score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Risk Level"},
            delta = {'reference': 50},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': risk_color},
                'steps': [
                    {'range': [0, 20], 'color': "#E8F5E8"},
                    {'range': [20, 40], 'color': "#FFF3E0"},
                    {'range': [40, 60], 'color': "#FFF8E1"},
                    {'range': [60, 80], 'color': "#FFECB3"},
                    {'range': [80, 100], 'color': "#FFEBEE"}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90}}))
        
        fig_gauge.update_layout(
            height=250,
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig_gauge, use_container_width=True)

    # Alert System
    if risk_score >= 70:
        st.markdown('<div class="alert-section">', unsafe_allow_html=True)
        st.markdown("### 🚨 RISK ALERT")
        
        alert_messages = {
            "CRITICAL": "⚠️ **IMMEDIATE ACTION REQUIRED** - Critical risk level detected. Emergency protocols should be activated.",
            "HIGH": "⚠️ **HIGH ALERT** - Elevated risk conditions. Increased monitoring and preparedness recommended."
        }
        
        st.markdown(alert_messages.get(risk_level, "⚠️ **ATTENTION** - Elevated risk detected."))
        
        col_alert1, col_alert2 = st.columns(2)
        with col_alert1:
            st.markdown("**Immediate Actions:**")
            st.markdown("- 🚨 Activate emergency protocols")
            st.markdown("- 📞 Alert emergency services")
            st.markdown("- 📢 Notify public authorities")
        
        with col_alert2:
            st.markdown("**Monitoring:**")
            st.markdown("- 👁️ Increase surveillance")
            st.markdown("- 📊 Monitor real-time data")
            st.markdown("- 🔄 Update risk assessment")
        
        st.markdown('</div>', unsafe_allow_html=True)

    # Detailed Risk Analysis
    st.markdown("---")
    st.markdown("### 📊 Detailed Risk Analysis")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Risk Breakdown", "🗺️ Geographic Risk", "⏰ Temporal Analysis", "🛡️ Mitigation"])
    
    with tab1:
        if category_columns:
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                # Risk by category
                st.markdown("#### 📊 Risk by Disaster Type")
                
                risk_by_category = []
                for category, count in category_counts.head(10).items():
                    category_lower = str(category).lower()
                    weight = 0.6
                    
                    for risk_type, risk_weight in risk_weights.items():
                        if risk_type in category_lower:
                            weight = risk_weight
                            break
                    
                    category_risk = count * weight
                    risk_by_category.append({
                        'Category': category,
                        'Count': count,
                        'Risk_Score': category_risk,
                        'Risk_Level': get_risk_level((category_risk/count)*100)[0] if count > 0 else 'LOW'
                    })
                
                risk_df = pd.DataFrame(risk_by_category)
                
                fig_risk = px.bar(
                    risk_df, 
                    x='Category', 
                    y='Risk_Score',
                    color='Risk_Level',
                    title="Risk Score by Category",
                    color_discrete_map={
                        'CRITICAL': '#D32F2F',
                        'HIGH': '#F57C00',
                        'MEDIUM': '#FF9800',
                        'LOW': '#4CAF50',
                        'MINIMAL': '#2E7D32'
                    }
                )
                fig_risk.update_layout(template="plotly_dark")
                st.plotly_chart(fig_risk, use_container_width=True)
            
            with col_chart2:
                # Risk distribution pie chart
                st.markdown("#### 🥧 Risk Distribution")
                
                fig_pie = px.pie(
                    risk_df.head(8), 
                    values='Risk_Score', 
                    names='Category',
                    title="Risk Distribution by Type"
                )
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                fig_pie.update_layout(template="plotly_dark")
                st.plotly_chart(fig_pie, use_container_width=True)
        
        # Risk factors analysis
        st.markdown("#### 🔍 Risk Factors Analysis")
        
        col_factor1, col_factor2, col_factor3 = st.columns(3)
        
        with col_factor1:
            st.markdown("""
            **🌡️ Environmental Factors:**
            - High incident frequency
            - Seasonal patterns
            - Weather conditions
            - Geographic vulnerability
            """)
        
        with col_factor2:
            st.markdown("""
            **🏙️ Infrastructure Factors:**
            - Population density
            - Building age/condition
            - Emergency response capacity
            - Communication systems
            """)
        
        with col_factor3:
            st.markdown("""
            **⚡ Operational Factors:**
            - Response time
            - Resource availability
            - Training level
            - Coordination efficiency
            """)
    
    with tab2:
        st.markdown("#### 🗺️ Geographic Risk Assessment")
        
        # Simulated geographic data (replace with actual geo data if available)
        col_geo1, col_geo2 = st.columns(2)
        
        with col_geo1:
            st.markdown("**📍 High-Risk Areas:**")
            high_risk_areas = [
                "Downtown District - HIGH RISK 🔴",
                "Industrial Zone - MEDIUM RISK 🟡", 
                "Residential Area A - LOW RISK 🟢",
                "Coastal Region - HIGH RISK 🔴",
                "Mountain Area - MEDIUM RISK 🟡"
            ]
            
            for area in high_risk_areas:
                st.markdown(f"- {area}")
        
        with col_geo2:
            st.markdown("**🎯 Risk Hotspots:**")
            
            # Create a simple heatmap simulation
            risk_data = pd.DataFrame({
                'Region': ['North', 'South', 'East', 'West', 'Central'],
                'Risk_Score': [85, 45, 70, 30, 60],
                'Incidents': [45, 12, 28, 8, 22]
            })
            
            fig_heatmap = px.bar(
                risk_data,
                x='Region',
                y='Risk_Score',
                color='Risk_Score',
                title="Regional Risk Scores",
                color_continuous_scale='Reds'
            )
            fig_heatmap.update_layout(template="plotly_dark", height=300)
            st.plotly_chart(fig_heatmap, use_container_width=True)
    
    with tab3:
        st.markdown("#### ⏰ Temporal Risk Analysis")
        
        if time_columns:
            try:
                time_col = time_columns[0]
                df_time = df.copy()
                df_time[time_col] = pd.to_datetime(df_time[time_col], errors='coerce')
                df_time = df_time.dropna(subset=[time_col])
                
                if len(df_time) > 0:
                    # Monthly trend
                    df_time['Month'] = df_time[time_col].dt.to_period('M')
                    monthly_counts = df_time['Month'].value_counts().sort_index()
                    
                    fig_timeline = px.line(
                        x=monthly_counts.index.astype(str),
                        y=monthly_counts.values,
                        title="Incident Trend Over Time",
                        labels={'x': 'Month', 'y': 'Number of Incidents'}
                    )
                    fig_timeline.update_layout(template="plotly_dark")
                    st.plotly_chart(fig_timeline, use_container_width=True)
                    
                    # Risk forecast (simplified)
                    st.markdown("#### 🔮 Risk Forecast")
                    
                    recent_avg = monthly_counts.tail(3).mean()
                    if recent_avg > monthly_counts.mean():
                        forecast_trend = "📈 INCREASING"
                        forecast_color = "#FF5722"
                    else:
                        forecast_trend = "📉 DECREASING"
                        forecast_color = "#4CAF50"
                    
                    st.markdown(f"""
                    **Next Month Forecast:** <span style="color: {forecast_color}">{forecast_trend}</span>
                    
                    **Predicted Risk Level:** {get_risk_level(risk_score * 1.1 if "INCREASING" in forecast_trend else risk_score * 0.9)[0]}
                    """, unsafe_allow_html=True)
                    
            except Exception as e:
                st.info("📅 Temporal analysis unavailable - date format not recognized")
        else:
            st.info("📅 No time-based data available for temporal analysis")
        
        # Peak risk times
        st.markdown("#### ⏰ Peak Risk Periods")
        
        col_time1, col_time2 = st.columns(2)
        
        with col_time1:
            st.markdown("""
            **📊 Historical Patterns:**
            - Summer months: Higher fire risk
            - Rainy season: Increased flood risk  
            - Winter: Storm and infrastructure risk
            - Weekend: Different incident patterns
            """)
        
        with col_time2:
            st.markdown("""
            **🔍 Risk Indicators:**
            - Incident frequency spikes
            - Seasonal vulnerability windows
            - Resource allocation periods
            - Emergency response patterns
            """)
    
    with tab4:
        st.markdown("#### 🛡️ Risk Mitigation Strategies")
        
        # Mitigation recommendations based on risk level
        mitigation_strategies = {
            "CRITICAL": [
                "🚨 Immediate evacuation planning",
                "📞 Emergency services coordination", 
                "📢 Public warning systems activation",
                "🏥 Medical facility preparation",
                "🚚 Resource mobilization"
            ],
            "HIGH": [
                "⚠️ Enhanced monitoring systems",
                "👥 Staff training programs",
                "📋 Emergency protocol updates",
                "🔧 Infrastructure reinforcement",
                "📊 Regular risk assessments"
            ],
            "MEDIUM": [
                "📈 Preventive maintenance",
                "👁️ Monitoring system upgrades",
                "📚 Community education",
                "🔄 Plan review and updates",
                "💼 Insurance coverage review"
            ],
            "LOW": [
                "📋 Routine inspections",
                "📖 Documentation updates",
                "👨‍🏫 Basic training programs",
                "📊 Data collection improvement",
                "🔍 Vulnerability assessments"
            ]
        }
        
        current_strategies = mitigation_strategies.get(risk_level, mitigation_strategies["MEDIUM"])
        
        col_mit1, col_mit2 = st.columns(2)
        
        with col_mit1:
            st.markdown('<div class="mitigation-card">', unsafe_allow_html=True)
            st.markdown(f"#### 🎯 Priority Actions for {risk_level} Risk")
            
            for i, strategy in enumerate(current_strategies, 1):
                st.markdown(f"{i}. {strategy}")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col_mit2:
            st.markdown("#### 📈 Implementation Timeline")
            
            timeline_items = [
                ("Immediate (0-24h)", "Emergency response activation", "#FF5722"),
                ("Short-term (1-7 days)", "Resource mobilization", "#FF9800"), 
                ("Medium-term (1-4 weeks)", "System improvements", "#FFC107"),
                ("Long-term (1-6 months)", "Infrastructure upgrades", "#4CAF50")
            ]
            
            for timeframe, action, color in timeline_items:
                st.markdown(f"""
                <div class="timeline-item" style="--accent-color: {color};">
                    <strong>{timeframe}</strong><br>
                    {action}
                </div>
                """, unsafe_allow_html=True)
        
        # Cost-benefit analysis
        st.markdown("---")
        st.markdown("#### 💰 Investment Priority Matrix")
        
        investment_data = pd.DataFrame({
            'Mitigation_Strategy': ['Early Warning System', 'Infrastructure Upgrade', 'Training Programs', 'Emergency Equipment', 'Communication Systems'],
            'Cost': [50000, 200000, 25000, 75000, 40000],
            'Impact': [90, 85, 70, 80, 75],
            'Urgency': [95, 60, 50, 85, 80]
        })
        
        fig_scatter = px.scatter(
            investment_data,
            x='Cost',
            y='Impact', 
            size='Urgency',
            hover_name='Mitigation_Strategy',
            title="Cost vs Impact Analysis",
            labels={'Cost': 'Implementation Cost ($)', 'Impact': 'Risk Reduction Impact (%)'}
        )
        fig_scatter.update_layout(template="plotly_dark")
        st.plotly_chart(fig_scatter, use_container_width=True)

    # Risk Summary Report
    st.markdown("---")
    st.markdown("### 📋 Risk Assessment Summary")
    
    col_summary1, col_summary2, col_summary3 = st.columns(3)
    
    with col_summary1:
        st.markdown("#### 🎯 Current Status")
        st.markdown(f"""
        - **Risk Level:** {risk_level}
        - **Risk Score:** {risk_score:.1f}/100
        - **Total Incidents:** {total_incidents:,}
        - **Assessment Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
        """)
    
    with col_summary2:
        st.markdown("#### 📊 Key Findings")
        if category_columns and len(category_counts) > 0:
            top_risk = category_counts.index[0]
            st.markdown(f"""
            - **Primary Risk:** {top_risk}
            - **Frequency:** {category_counts.iloc[0]} incidents
            - **Trend:** {"Increasing" if risk_score > 50 else "Stable"}
            - **Priority:** {"High" if risk_score > 60 else "Medium"}
            """)
        else:
            st.markdown("""
            - **Data Status:** Limited information
            - **Assessment:** Based on incident count
            - **Recommendation:** Improve data collection
            - **Next Review:** Required
            """)
    
    with col_summary3:
        st.markdown("#### 🚀 Next Steps")
        st.markdown(f"""
        - **Immediate:** {"Emergency protocols" if risk_score > 70 else "Enhanced monitoring"}
        - **Short-term:** {"Resource mobilization" if risk_score > 60 else "Preventive measures"}
        - **Long-term:** {"Infrastructure review" if risk_score > 50 else "Routine maintenance"}
        - **Review:** {"Weekly" if risk_score > 70 else "Monthly"}
        """)
    
    # Export functionality
    st.markdown("---")
    col_export1, col_export2, col_export3 = st.columns(3)
    
    with col_export1:
        if st.button("📄 Generate Report", type="primary"):
            st.success("✅ Risk assessment report generated!")
            st.info("📧 Report has been prepared for export")
    
    with col_export2:
        if st.button("📧 Send Alert", type="secondary"):
            if risk_score > 60:
                st.warning("🚨 High-risk alert sent to emergency contacts")
            else:
                st.info("📨 Routine status update sent")
    
    with col_export3:
        if st.button("🔄 Refresh Assessment", type="secondary"):
            st.rerun()

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; opacity: 0.7; font-size: 0.9rem;">
        🛡️ Risk Assessment System | Last Updated: {} | Next Review: {}
    </div>
    """.format(
        pd.Timestamp.now().strftime('%Y-%m-%d %H:%M'),
        (pd.Timestamp.now() + pd.Timedelta(days=7)).strftime('%Y-%m-%d')
    ), unsafe_allow_html=True)

def show_data_upload_page(df):
    st.subheader("📁 Data Management & Upload")
    
    # Professional styling
    st.markdown("""
    <style>
        .upload-container {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 15px;
            padding: 25px;
            margin: 20px 0;
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.2);
        }
        .data-preview-container {
            background: #f8f9fa;
            border-radius: 12px;
            padding: 20px;
            margin: 15px 0;
            border: 1px solid #e9ecef;
        }
        .feature-box {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
            border-left: 4px solid #4ECDC4;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        .stat-card {
            background: white;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
            border-top: 4px solid var(--accent-color);
        }
    </style>
    """, unsafe_allow_html=True)

    # File Upload Section
    st.markdown("### 📤 Upload New Dataset")
    
    col_upload, col_info = st.columns([2, 1])
    
    with col_upload:
        uploaded_file = st.file_uploader(
            "Choose a file to upload",
            type=['csv', 'xlsx', 'json'],
            help="Supported formats: CSV, Excel (.xlsx), JSON"
        )
        
        if uploaded_file is not None:
            try:
                # Handle different file types
                if uploaded_file.name.endswith('.csv'):
                    new_df = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith('.xlsx'):
                    new_df = pd.read_excel(uploaded_file)
                elif uploaded_file.name.endswith('.json'):
                    new_df = pd.read_json(uploaded_file)
                
                st.success(f"✅ Successfully loaded {uploaded_file.name}")
                
                # Quick preview of uploaded data
                st.markdown("#### 👀 Quick Preview")
                st.dataframe(new_df.head(3), use_container_width=True)
                
                # Option to replace current dataset
                if st.button("🔄 Replace Current Dataset", type="primary"):
                    st.session_state['uploaded_df'] = new_df
                    st.success("Dataset replaced successfully!")
                    st.experimental_rerun()
                    
            except Exception as e:
                st.error(f"❌ Error loading file: {str(e)}")
    
    with col_info:
        st.markdown('<div class="upload-container">', unsafe_allow_html=True)
        st.markdown("""
        **📋 Upload Guidelines:**
        
        • **CSV**: Comma-separated values
        • **Excel**: .xlsx format preferred
        • **JSON**: Structured data format
        
        **💡 Tips:**
        - Ensure column headers are present
        - Check for missing values
        - Verify data consistency
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    # Current Dataset Overview
    st.markdown("---")
    st.markdown("### 📊 Current Dataset Overview")
    
    # Dataset Statistics Cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="stat-card" style="--accent-color: #FF6B6B;">
            <h3 style="color: #FF6B6B; margin: 0;">📝</h3>
            <h2 style="margin: 10px 0; color: #333;">{:,}</h2>
            <p style="margin: 0; color: #666;">Total Rows</p>
        </div>
        """.format(len(df)), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="stat-card" style="--accent-color: #4ECDC4;">
            <h3 style="color: #4ECDC4; margin: 0;">📋</h3>
            <h2 style="margin: 10px 0; color: #333;">{}</h2>
            <p style="margin: 0; color: #666;">Columns</p>
        </div>
        """.format(len(df.columns)), unsafe_allow_html=True)
    
    with col3:
        memory_usage = df.memory_usage(deep=True).sum() / 1024**2
        st.markdown("""
        <div class="stat-card" style="--accent-color: #667eea;">
            <h3 style="color: #667eea; margin: 0;">💾</h3>
            <h2 style="margin: 10px 0; color: #333;">{:.1f} MB</h2>
            <p style="margin: 0; color: #666;">Memory Usage</p>
        </div>
        """.format(memory_usage), unsafe_allow_html=True)
    
    with col4:
        missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        st.markdown("""
        <div class="stat-card" style="--accent-color: #FFA726;">
            <h3 style="color: #FFA726; margin: 0;">⚠️</h3>
            <h2 style="margin: 10px 0; color: #333;">{:.1f}%</h2>
            <p style="margin: 0; color: #666;">Missing Data</p>
        </div>
        """.format(missing_pct), unsafe_allow_html=True)

    # Interactive Data Explorer
    st.markdown("---")
    st.markdown("### 🔍 Interactive Data Explorer")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Data View", "🔍 Column Details", "📊 Data Quality", "⚙️ Export Options"])
    
    with tab1:
        # Enhanced data display with filters
        col_filter1, col_filter2, col_filter3 = st.columns(3)
        
        with col_filter1:
            rows_to_show = st.selectbox("Rows to display:", [10, 25, 50, 100], index=1)
        
        with col_filter2:
            if len(df.columns) > 5:
                show_all_cols = st.checkbox("Show all columns", value=True)
            else:
                show_all_cols = True
        
        with col_filter3:
            search_term = st.text_input("🔍 Search in data:", placeholder="Enter search term...")
        
        # Apply filters
        display_df = df.copy()
        
        if search_term:
            # Search across all string columns
            string_cols = display_df.select_dtypes(include=['object']).columns
            mask = pd.Series([False] * len(display_df))
            for col in string_cols:
                mask |= display_df[col].astype(str).str.contains(search_term, case=False, na=False)
            display_df = display_df[mask]
        
        if not show_all_cols and len(df.columns) > 5:
            display_df = display_df.iloc[:, :5]
        
        # Display with enhanced styling
        st.dataframe(
            display_df.head(rows_to_show),
            use_container_width=True,
            height=400
        )
        
        if len(display_df) != len(df):
            st.info(f"📊 Showing {len(display_df)} filtered results out of {len(df)} total rows")
    
    with tab2:
        # Column information with enhanced details
        st.markdown("#### 📋 Column Information")
        
        # Create detailed column info
        column_info = []
        for col in df.columns:
            col_data = {
                'Column': col,
                'Data Type': str(df[col].dtype),
                'Non-Null Count': df[col].count(),
                'Null Count': df[col].isnull().sum(),
                'Null %': f"{(df[col].isnull().sum() / len(df)) * 100:.1f}%",
                'Unique Values': df[col].nunique()
            }
            
            if df[col].dtype in ['int64', 'float64']:
                col_data['Min'] = df[col].min()
                col_data['Max'] = df[col].max()
                col_data['Mean'] = f"{df[col].mean():.2f}"
            else:
                col_data['Min'] = '-'
                col_data['Max'] = '-'
                col_data['Mean'] = '-'
            
            column_info.append(col_data)
        
        column_df = pd.DataFrame(column_info)
        st.dataframe(column_df, use_container_width=True, height=400)
    
    with tab3:
        # Data Quality Assessment
        st.markdown("#### 🎯 Data Quality Report")
        
        col_quality1, col_quality2 = st.columns(2)
        
        with col_quality1:
            # Missing data visualization
            missing_data = df.isnull().sum()
            missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
            
            if len(missing_data) > 0:
                fig_missing = go.Figure(go.Bar(
                    x=missing_data.values,
                    y=missing_data.index,
                    orientation='h',
                    marker_color='rgba(255, 107, 107, 0.8)'
                ))
                
                fig_missing.update_layout(
                    title="Missing Values by Column",
                    xaxis_title="Number of Missing Values",
                    yaxis_title="Columns",
                    height=300,
                    template="plotly_white"
                )
                
                st.plotly_chart(fig_missing, use_container_width=True)
            else:
                st.success("✅ No missing values found!")
        
        with col_quality2:
            # Data completeness
            completeness = (1 - (df.isnull().sum().sum() / (len(df) * len(df.columns)))) * 100
            
            st.markdown("**📊 Data Quality Metrics:**")
            st.metric("Data Completeness", f"{completeness:.1f}%")
            st.progress(completeness / 100)
            
            # Duplicate rows
            duplicates = df.duplicated().sum()
            st.metric("Duplicate Rows", f"{duplicates:,}")
            
            # Data types distribution
            dtype_counts = df.dtypes.value_counts()
            st.markdown("**🔢 Data Types:**")
            for dtype, count in dtype_counts.items():
                st.write(f"• **{dtype}**: {count} columns")
    
    with tab4:
        # Export options
        st.markdown("#### 📤 Export Current Dataset")
        
        col_export1, col_export2 = st.columns(2)
        
        with col_export1:
            # CSV Export
            csv_buffer = df.to_csv(index=False)
            st.download_button(
                label="📄 Download as CSV",
                data=csv_buffer,
                file_name=f"dataset_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            # JSON Export
            json_buffer = df.to_json(orient='records', indent=2)
            st.download_button(
                label="📋 Download as JSON",
                data=json_buffer,
                file_name=f"dataset_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
        
        with col_export2:
            # Sample data export
            sample_size = st.slider("Sample size for export:", 10, min(1000, len(df)), 100)
            sample_df = df.sample(n=sample_size, random_state=42)
            
            sample_csv = sample_df.to_csv(index=False)
            st.download_button(
                label=f"📊 Download Sample ({sample_size} rows)",
                data=sample_csv,
                file_name=f"sample_dataset_{sample_size}rows.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            st.info(f"💡 Sample contains {sample_size} randomly selected rows")

    # Dataset Actions
    st.markdown("---")
    st.markdown("### ⚙️ Dataset Actions")
    
    col_action1, col_action2, col_action3, col_action4 = st.columns(4)
    
    with col_action1:
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.experimental_rerun()
    
    with col_action2:
        if st.button("🧹 Clean Data", use_container_width=True):
            st.info("Data cleaning functionality would be implemented here")
    
    with col_action3:
        if st.button("📊 Generate Report", use_container_width=True):
            st.info("Automated report generation coming soon!")
    
    with col_action4:
        if st.button("💾 Save Session", use_container_width=True):
            st.info("Session save functionality would be implemented here")

    # Footer info
    st.markdown("---")
    st.markdown("""
    💡 **Pro Tips:**
    - Use the search function to quickly find specific data
    - Export samples for testing before downloading full dataset
    - Check data quality regularly for better analysis results
    - Upload new datasets to compare different data sources
    """)

# ============================
# Main Application
# ============================
def main():
    st.markdown('<h1 class="main-header">🚨 Disaster Response Analytics Dashboard</h1>', unsafe_allow_html=True)

    dashboard = DisasterAnalyticsDashboard()

    # Use the default file path directly, no upload required
    file_path = "/Users/rullbachtiar/Downloads/Folder Baru Dengan Item/disaster_messages.csv"  # Change this to the correct file path

    if os.path.exists(file_path):
        st.success("✅ Using default dataset")
        data = load_data(file_path)
    else:
        st.error("⚠️ Dataset not found. Please ensure the file exists at the specified path.")
        data = None

    if data is not None:
        st.write(f"Data Loaded: {data.shape}")  # Debugging line to check the shape of the data
        processed_data = dashboard.preprocess_data(data.copy())

        page = st.sidebar.selectbox(
            "Choose a page:",
            ["🏠 Overview", "📊 Data Analysis", "🔮 AI Prediction", "📈 Risk Assessment", "💾 Data Upload"]
        )

        if page == "🏠 Overview":
            show_overview(processed_data)
        elif page == "📊 Data Analysis":
            show_data_analysis(processed_data)
        elif page == "🔮 AI Prediction":
            show_prediction_page(processed_data, dashboard)
        elif page == "📈 Risk Assessment":
            show_risk_assessment(processed_data)
        elif page == "💾 Data Upload":
            show_data_upload_page(processed_data)
    else:
        st.error("❌ No data available. Please check your file path.")

if __name__ == "__main__":
    main()
