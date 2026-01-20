import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
from datetime import datetime
import pydeck as pdk
import os

# Set page configuration
st.set_page_config(
    page_title="World Famous Places Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
        padding-bottom: 1rem;
        border-bottom: 3px solid #1E3A8A;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #1E40AF;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 1rem;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 5px;
        font-weight: bold;
    }
    .stSelectbox, .stMultiSelect, .stSlider {
        margin-bottom: 1rem;
    }
    .image-container {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .placeholder-image {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 40px 20px;
        text-align: center;
        color: white;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and preprocess the dataset"""
    try:
        df = pd.read_csv('world_famous_places_2024.csv')
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Check for required columns
        required_columns = ['Place_Name', 'Country', 'Annual_Visitors_Millions', 
                           'Entry_Fee_USD', 'Tourism_Revenue_Million_USD', 
                           'Average_Visit_Duration_Hours', 'Year_Built']
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            st.warning(f"Missing columns: {missing_columns}. Trying to find alternatives...")
        
        # Clean string columns
        string_cols = df.select_dtypes(include=['object']).columns
        for col in string_cols:
            df[col] = df[col].astype(str).str.strip()
        
        # Convert numeric columns - handle non-numeric values
        numeric_cols = ['Annual_Visitors_Millions', 'Entry_Fee_USD', 
                       'Tourism_Revenue_Million_USD', 'Average_Visit_Duration_Hours']
        
        # Only convert columns that exist
        numeric_cols = [col for col in numeric_cols if col in df.columns]
        
        for col in numeric_cols:
            # Replace any non-numeric strings with NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Convert Year_Built to numeric, handle "Natural Formation" and similar
        if 'Year_Built' in df.columns:
            df['Year_Built'] = pd.to_numeric(df['Year_Built'], errors='coerce')
        
        # Calculate derived metrics only if we have the required columns
        if 'Tourism_Revenue_Million_USD' in df.columns and 'Annual_Visitors_Millions' in df.columns:
            # Avoid division by zero
            df['Revenue_Per_Visitor_USD'] = np.where(
                df['Annual_Visitors_Millions'] > 0,
                (df['Tourism_Revenue_Million_USD'] * 1_000_000) / (df['Annual_Visitors_Millions'] * 1_000_000),
                0
            )
            df['Revenue_Per_Visitor_USD'] = df['Revenue_Per_Visitor_USD'].round(2)
        
        if 'Average_Visit_Duration_Hours' in df.columns and 'Entry_Fee_USD' in df.columns:
            # Add small value to avoid division by zero
            df['Value_Ratio'] = df['Average_Visit_Duration_Hours'] / (df['Entry_Fee_USD'].replace(0, np.nan) + 0.01)
        
        # Calculate age - handle NaN values properly
        if 'Year_Built' in df.columns:
            current_year = datetime.now().year
            df['Age'] = current_year - df['Year_Built']
            # Mark natural formations as 0 age (instead of NaN)
            df['Age'] = df['Age'].fillna(0)
            
            # Create age categories - handle negative ages if any
            df['Age'] = df['Age'].clip(lower=0)
            
            bins = [0, 100, 500, 1000, 2000, 5000, 10000]
            labels = ['<100 years', '100-500 years', '500-1000 years', 
                     '1000-2000 years', '2000-5000 years', 'Ancient (>5000)']
            
            # Use try-except to handle edge cases
            try:
                df['Age_Category'] = pd.cut(df['Age'], bins=bins, labels=labels, include_lowest=True)
            except:
                # If cutting fails, create a simple category
                df['Age_Category'] = 'Unknown'
        else:
            df['Age'] = 0
            df['Age_Category'] = 'Unknown'
        
        # Create season categories
        if 'Best_Visit_Month' in df.columns:
            def get_season(month_range):
                if pd.isna(month_range):
                    return 'Not Specified'
                
                months = str(month_range).lower()
                if any(m in months for m in ['dec', 'jan', 'feb']):
                    return 'Winter'
                elif any(m in months for m in ['mar', 'apr', 'may']):
                    return 'Spring'
                elif any(m in months for m in ['june', 'july', 'aug']):
                    return 'Summer'
                elif any(m in months for m in ['sept', 'oct', 'nov']):
                    return 'Fall'
                else:
                    return 'Multiple/Varies'
            
            df['Best_Season'] = df['Best_Visit_Month'].apply(get_season)
        else:
            df['Best_Season'] = 'Not Specified'
        
        return df
    
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.error("Please ensure 'world_famous_places_2024.csv' is in the same directory.")
        return pd.DataFrame()  # Return empty dataframe instead of None

def get_place_image_url(place_name):
    """Get image URL for a place"""
    image_urls = {
        # United States
        'Times Square': 'https://upload.wikimedia.org/wikipedia/commons/4/47/New_york_times_square-terabass.jpg',
        'Statue of Liberty': 'https://upload.wikimedia.org/wikipedia/commons/a/a1/Statue_of_Liberty_7.jpg',
        'Central Park': 'https://upload.wikimedia.org/wikipedia/commons/8/83/Central_Park_New_York_City_skyline.jpg',
        'Las Vegas Strip': 'https://upload.wikimedia.org/wikipedia/commons/a/af/Las_Vegas_Skyline_at_night.jpg',
        'Empire State Building': 'https://upload.wikimedia.org/wikipedia/commons/1/10/Empire_State_Building_%28aerial_view%29.jpg',
        'Golden Gate Bridge': 'https://upload.wikimedia.org/wikipedia/commons/0/0c/GoldenGateBridge-001.jpg',
        'Lincoln Memorial': 'https://upload.wikimedia.org/wikipedia/commons/2/2c/Lincoln_Memorial_2006.jpg',
        'Grand Canyon': 'https://upload.wikimedia.org/wikipedia/commons/a/aa/Dawn_on_the_S_rim_of_the_Grand_Canyon_%288645178272%29.jpg',
        'Disneyland (California)': 'https://upload.wikimedia.org/wikipedia/commons/e/eb/Disneyland_entrance_sign.jpg',
        'Magic Kingdom (Orlando)': 'https://upload.wikimedia.org/wikipedia/commons/3/3d/Cinderella_Castle_2020.jpg',
        
        # France
        'Eiffel Tower': 'https://upload.wikimedia.org/wikipedia/commons/8/85/Tour_Eiffel_Wikimedia_Commons_%28cropped%29.jpg',
        'Louvre Museum': 'https://upload.wikimedia.org/wikipedia/commons/6/66/Louvre_Museum_Wikimedia_Commons.jpg',
        'Notre-Dame Cathedral': 'https://upload.wikimedia.org/wikipedia/commons/1/1e/Notre-Dame_de_Paris_2013-07-24.jpg',
        'Palace of Versailles': 'https://upload.wikimedia.org/wikipedia/commons/c/c3/Chateau_Versailles_Galerie_des_Glaces.jpg',
        
        # Italy
        'Colosseum': 'https://upload.wikimedia.org/wikipedia/commons/d/de/Colosseo_2020.jpg',
        'Leaning Tower of Pisa': 'https://upload.wikimedia.org/wikipedia/commons/6/66/The_Leaning_Tower_of_Pisa_SB.jpeg',
        
        # China
        'Great Wall of China': 'https://upload.wikimedia.org/wikipedia/commons/2/23/The_Great_Wall_of_China_at_Jinshanling-edit.jpg',
        'Forbidden City': 'https://upload.wikimedia.org/wikipedia/commons/3/34/Forbidden_City_Panorama.jpg',
        
        # India
        'Taj Mahal': 'https://upload.wikimedia.org/wikipedia/commons/1/1d/Taj_Mahal_%28Edited%29.jpeg',
        
        # Australia
        'Sydney Opera House': 'https://upload.wikimedia.org/wikipedia/commons/2/26/SydneyOperaHouse-20070401.jpg',
        
        # Peru
        'Machu Picchu': 'https://upload.wikimedia.org/wikipedia/commons/e/eb/Machu_Picchu%2C_Peru.jpg',
        
        # Cambodia
        'Angkor Wat': 'https://upload.wikimedia.org/wikipedia/commons/d/d7/Angkor_Wat.jpg',
        
        # Spain
        'Sagrada Familia': 'https://upload.wikimedia.org/wikipedia/commons/7/73/Sagrada_Familia_03.jpg',
        
        # Egypt
        'Great Pyramid of Giza': 'https://upload.wikimedia.org/wikipedia/commons/e/e3/Kheops-Pyramid.jpg',
        
        # Greece
        'Acropolis': 'https://upload.wikimedia.org/wikipedia/commons/4/4c/The_Parthenon_in_Athens.jpg',
        
        # UAE
        'Burj Khalifa': 'https://upload.wikimedia.org/wikipedia/commons/c/c7/Burj_Khalifa_2021.jpg',
        
        # UK
        'Big Ben': 'https://upload.wikimedia.org/wikipedia/commons/b/b7/Big_Ben_London.jpg',
        'Tower Bridge': 'https://upload.wikimedia.org/wikipedia/commons/6/63/Tower_Bridge_London_Feb_2006.jpg',
        'Buckingham Palace': 'https://upload.wikimedia.org/wikipedia/commons/2/2c/Buckingham_Palace_aerial_view_2016_%28cropped%29.jpg',
        
        # Brazil
        'Christ the Redeemer': 'https://upload.wikimedia.org/wikipedia/commons/4/4f/Christ_the_Redeemer_-_Cristo_Redentor.jpg',
    }
    
    return image_urls.get(place_name, None)

def display_place_image(place_name, place_type, country):
    """Display image or placeholder for a place"""
    
    image_url = get_place_image_url(place_name)
    
    if image_url:
        st.markdown(f'<div class="image-container">', unsafe_allow_html=True)
        st.image(image_url, caption=place_name, use_container_width=True)
        st.markdown(f'</div>', unsafe_allow_html=True)
    else:
        # Type to emoji mapping for placeholder
        type_to_emoji = {
            'Monument/Tower': '🗼',
            'Urban Landmark': '🏙️',
            'Museum': '🏛️',
            'Historic Monument': '🏺',
            'Monument/Mausoleum': '⚰️',
            'Cultural Building': '🏛️',
            'Archaeological Site': '🏺',
            'Historic Palace': '🏰',
            'Temple Complex': '🛕',
            'Cathedral': '⛪',
            'Park': '🌳',
            'Entertainment District': '🎪',
            'Skyscraper': '🏙️',
            'Bridge': '🌉',
            'Monument/Memorial': '🗽',
            'Theme Park': '🎢',
            'Natural Wonder': '🏞️',
            'Historic Tower': '🗼',
            'Clock Tower': '🕰️',
            'Palace': '🏰',
            'Monument/Statue': '🗽'
        }
        
        emoji = type_to_emoji.get(place_type, '🗺️')
        
        st.markdown(f'''
        <div class="placeholder-image">
            <div style="font-size: 4rem; margin-bottom: 10px;">{emoji}</div>
            <h3 style="margin: 5px 0;">{place_name}</h3>
            <p style="margin: 5px 0; opacity: 0.8;">{country}</p>
            <p style="margin: 5px 0; font-size: 0.9em; opacity: 0.7;">{place_type}</p>
        </div>
        ''', unsafe_allow_html=True)

def main():
    """Main Streamlit application"""
    
    # Title and header
    st.markdown('<h1 class="main-header">🌍 World Famous Places Dashboard</h1>', unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    
    if df.empty:
        st.error("""
        Failed to load data. Please ensure:
        1. The file 'world_famous_places_2024.csv' is in the same directory
        2. The file has the correct format
        3. You have read permissions for the file
        """)
        return
    
    # Check if dataframe has data
    if len(df) == 0:
        st.warning("No data loaded. Please check your CSV file.")
        return
    
    # Create sidebar
    with st.sidebar:
        st.markdown("### 🌍 Tourism Analytics")
        st.markdown("---")
        
        # Page selection
        page = st.selectbox(
            "Select Page",
            ["🏠 Overview", "📈 Analytics", "🗺️ Geographic Analysis", "💰 Financial Insights", "🔍 Detailed Explorer"]
        )
        
        # Filters
        st.markdown("### 🔍 Filters")
        
        # Region filter - check if column exists
        if 'Region' in df.columns:
            regions = ['All'] + sorted(df['Region'].unique().tolist())
            selected_region = st.selectbox("Select Region", regions)
        else:
            selected_region = 'All'
        
        # Type filter - check if column exists
        if 'Type' in df.columns:
            place_types = ['All'] + sorted(df['Type'].unique().tolist())
            selected_type = st.selectbox("Select Type", place_types)
        else:
            selected_type = 'All'
        
        # UNESCO filter - check if column exists
        if 'UNESCO_World_Heritage' in df.columns:
            unesco_filter = st.selectbox("UNESCO Status", ['All', 'Yes', 'No'])
        else:
            unesco_filter = 'All'
        
        # Visitor range filter - check if column exists
        if 'Annual_Visitors_Millions' in df.columns:
            min_visitors = float(df['Annual_Visitors_Millions'].min())
            max_visitors = float(df['Annual_Visitors_Millions'].max())
            
            # Handle case where min and max might be the same
            if min_visitors == max_visitors:
                max_visitors = min_visitors + 1
            
            min_val, max_val = st.slider(
                "Annual Visitors (Millions)",
                min_visitors,
                max_visitors,
                (min_visitors, max_visitors)
            )
        else:
            min_val, max_val = (0, 100)  # Default values
        
        # Entry fee filter - check if column exists
        if 'Entry_Fee_USD' in df.columns:
            min_fee = float(df['Entry_Fee_USD'].min())
            max_fee = float(df['Entry_Fee_USD'].max())
            
            if min_fee == max_fee:
                max_fee = min_fee + 1
            
            min_fee_val, max_fee_val = st.slider(
                "Entry Fee (USD)",
                min_fee,
                max_fee,
                (min_fee, max_fee)
            )
        else:
            min_fee_val, max_fee_val = (0, 100)
        
        st.markdown("---")
        st.markdown("### 📊 Quick Stats")
        
        # Display stats safely
        st.metric("Total Places", len(df))
        
        if 'Annual_Visitors_Millions' in df.columns:
            avg_visitors = df['Annual_Visitors_Millions'].mean()
            st.metric("Avg Annual Visitors", f"{avg_visitors:.1f}M")
        
        if 'Entry_Fee_USD' in df.columns:
            avg_fee = df['Entry_Fee_USD'].mean()
            st.metric("Avg Entry Fee", f"${avg_fee:.1f}")
    
    # Apply filters
    filtered_df = df.copy()
    
    if selected_region != 'All' and 'Region' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Region'] == selected_region]
    
    if selected_type != 'All' and 'Type' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Type'] == selected_type]
    
    if unesco_filter != 'All' and 'UNESCO_World_Heritage' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['UNESCO_World_Heritage'] == unesco_filter]
    
    # Apply numeric filters if columns exist
    if 'Annual_Visitors_Millions' in filtered_df.columns:
        filtered_df = filtered_df[
            (filtered_df['Annual_Visitors_Millions'] >= min_val) &
            (filtered_df['Annual_Visitors_Millions'] <= max_val)
        ]
    
    if 'Entry_Fee_USD' in filtered_df.columns:
        filtered_df = filtered_df[
            (filtered_df['Entry_Fee_USD'] >= min_fee_val) &
            (filtered_df['Entry_Fee_USD'] <= max_fee_val)
        ]
    
    # Page routing
    if page == "🏠 Overview":
        show_overview_page(filtered_df, df)
    elif page == "📈 Analytics":
        show_analytics_page(filtered_df)
    elif page == "🗺️ Geographic Analysis":
        show_geographic_page(filtered_df)
    elif page == "💰 Financial Insights":
        show_financial_page(filtered_df)
    elif page == "🔍 Detailed Explorer":
        show_explorer_page(filtered_df)

def show_overview_page(filtered_df, full_df):
    """Display the overview page"""
    
    st.markdown('<h2 class="sub-header">Global Tourism Overview</h2>', unsafe_allow_html=True)
    
    # Key metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if 'Tourism_Revenue_Million_USD' in filtered_df.columns:
            total_revenue = filtered_df['Tourism_Revenue_Million_USD'].sum()
            st.metric("Total Tourism Revenue", f"${total_revenue:,.0f}M")
        else:
            st.metric("Total Places", len(filtered_df))
    
    with col2:
        if 'Annual_Visitors_Millions' in filtered_df.columns:
            total_visitors = filtered_df['Annual_Visitors_Millions'].sum()
            st.metric("Total Annual Visitors", f"{total_visitors:,.1f}M")
        else:
            st.metric("Countries", filtered_df['Country'].nunique() if 'Country' in filtered_df.columns else 0)
    
    with col3:
        if 'Average_Visit_Duration_Hours' in filtered_df.columns:
            avg_duration = filtered_df['Average_Visit_Duration_Hours'].mean()
            st.metric("Avg Visit Duration", f"{avg_duration:.1f} hours")
        else:
            st.metric("Types", filtered_df['Type'].nunique() if 'Type' in filtered_df.columns else 0)
    
    with col4:
        if 'UNESCO_World_Heritage' in filtered_df.columns:
            unesco_count = filtered_df[filtered_df['UNESCO_World_Heritage'] == 'Yes'].shape[0]
            st.metric("UNESCO Sites", unesco_count)
        else:
            st.metric("Cities", filtered_df['City'].nunique() if 'City' in filtered_df.columns else 0)
    
    # Top performers
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🏆 Top 5 Most Visited")
        if 'Annual_Visitors_Millions' in filtered_df.columns and 'Place_Name' in filtered_df.columns:
            top_visited = filtered_df.nlargest(5, 'Annual_Visitors_Millions')[['Place_Name', 'Country', 'Annual_Visitors_Millions']]
            for idx, row in top_visited.iterrows():
                st.info(f"**{row['Place_Name']}** ({row['Country']}) - {row['Annual_Visitors_Millions']:.1f}M visitors")
        else:
            st.warning("Visitor data not available")
    
    with col2:
        st.markdown("### 💰 Top 5 Revenue Generators")
        if 'Tourism_Revenue_Million_USD' in filtered_df.columns and 'Place_Name' in filtered_df.columns:
            top_revenue = filtered_df.nlargest(5, 'Tourism_Revenue_Million_USD')[['Place_Name', 'Country', 'Tourism_Revenue_Million_USD']]
            for idx, row in top_revenue.iterrows():
                st.success(f"**{row['Place_Name']}** ({row['Country']}) - ${row['Tourism_Revenue_Million_USD']:,.0f}M")
        else:
            st.warning("Revenue data not available")
    
    # Visualization row 1 - only if we have data
    st.markdown("---")
    
    if 'Annual_Visitors_Millions' in filtered_df.columns and len(filtered_df) > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Visitors Distribution")
            fig = px.box(filtered_df, y='Annual_Visitors_Millions', 
                        title="Distribution of Annual Visitors")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'Type' in filtered_df.columns:
                st.markdown("### 🏛️ Places by Type")
                type_counts = filtered_df['Type'].value_counts()
                if len(type_counts) > 0:
                    fig = px.pie(values=type_counts.values, names=type_counts.index,
                                title="Distribution by Type")
                    st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Insufficient data for visualizations")
    
    # Visualization row 2
    if len(filtered_df) > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            if 'Best_Season' in filtered_df.columns:
                st.markdown("### 🌡️ Best Visiting Seasons")
                season_counts = filtered_df['Best_Season'].value_counts()
                if len(season_counts) > 0:
                    fig = px.bar(x=season_counts.index, y=season_counts.values,
                                title="Places by Best Visiting Season")
                    st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'UNESCO_World_Heritage' in filtered_df.columns:
                st.markdown("### 🏅 UNESCO vs Non-UNESCO")
                unesco_counts = filtered_df['UNESCO_World_Heritage'].value_counts()
                if len(unesco_counts) > 0:
                    fig = px.bar(x=unesco_counts.index, y=unesco_counts.values,
                                title="UNESCO World Heritage Sites")
                    st.plotly_chart(fig, use_container_width=True)
    
    # Data preview
    st.markdown("---")
    st.markdown("### 📋 Data Preview")
    st.dataframe(filtered_df.head(10), use_container_width=True)

def show_analytics_page(filtered_df):
    """Display the analytics page"""
    
    st.markdown('<h2 class="sub-header">📈 Detailed Analytics</h2>', unsafe_allow_html=True)
    
    # Check if we have enough data for analytics
    if len(filtered_df) < 2:
        st.warning("Not enough data for detailed analytics. Please adjust filters.")
        return
    
    # Tab layout
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Distributions", "📈 Trends", "🏛️ Categories", "🔗 Correlations"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            if 'Entry_Fee_USD' in filtered_df.columns:
                st.markdown("### Entry Fee Distribution")
                # Remove NaN values for histogram
                fee_data = filtered_df['Entry_Fee_USD'].dropna()
                if len(fee_data) > 0:
                    fig = px.histogram(filtered_df, x='Entry_Fee_USD', nbins=20,
                                    title="Distribution of Entry Fees",
                                    color_discrete_sequence=['#FF6B6B'])
                    st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'Average_Visit_Duration_Hours' in filtered_df.columns:
                st.markdown("### Visit Duration Distribution")
                duration_data = filtered_df['Average_Visit_Duration_Hours'].dropna()
                if len(duration_data) > 0:
                    fig = px.histogram(filtered_df, x='Average_Visit_Duration_Hours', nbins=20,
                                    title="Distribution of Visit Duration",
                                    color_discrete_sequence=['#4ECDC4'])
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            if 'Age' in filtered_df.columns and 'Annual_Visitors_Millions' in filtered_df.columns:
                st.markdown("### Age vs Visitors")
                # Filter out NaN values
                scatter_data = filtered_df.dropna(subset=['Age', 'Annual_Visitors_Millions'])
                if len(scatter_data) > 0:
                    fig = px.scatter(scatter_data, x='Age', y='Annual_Visitors_Millions',
                                  size='Annual_Visitors_Millions', 
                                  color='Region' if 'Region' in scatter_data.columns else None,
                                  hover_name='Place_Name',
                                  title="Age of Place vs Annual Visitors")
                    st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'Entry_Fee_USD' in filtered_df.columns and 'Average_Visit_Duration_Hours' in filtered_df.columns:
                st.markdown("### Entry Fee vs Duration")
                scatter_data = filtered_df.dropna(subset=['Entry_Fee_USD', 'Average_Visit_Duration_Hours'])
                if len(scatter_data) > 0:
                    fig = px.scatter(scatter_data, x='Entry_Fee_USD', y='Average_Visit_Duration_Hours',
                                  size='Annual_Visitors_Millions' if 'Annual_Visitors_Millions' in scatter_data.columns else None,
                                  color='Type' if 'Type' in scatter_data.columns else None,
                                  hover_name='Place_Name',
                                  title="Entry Fee vs Visit Duration")
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            if 'Type' in filtered_df.columns:
                st.markdown("### Average Metrics by Type")
                metrics_by_type = filtered_df.groupby('Type').agg({
                    'Annual_Visitors_Millions': 'mean',
                    'Entry_Fee_USD': 'mean',
                    'Average_Visit_Duration_Hours': 'mean'
                }).round(2)
                
                # Remove rows with all NaN
                metrics_by_type = metrics_by_type.dropna(how='all')
                
                if len(metrics_by_type) > 0:
                    # Display as bar chart for one metric
                    if 'Annual_Visitors_Millions' in metrics_by_type.columns:
                        fig = px.bar(metrics_by_type, x=metrics_by_type.index, y='Annual_Visitors_Millions',
                                    title="Average Annual Visitors by Type")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Show the table
                    st.dataframe(metrics_by_type)
        
        with col2:
            if 'Age_Category' in filtered_df.columns:
                st.markdown("### Age Categories Analysis")
                age_stats = filtered_df.groupby('Age_Category').agg({
                    'Place_Name': 'count',
                    'Annual_Visitors_Millions': 'mean',
                    'Entry_Fee_USD': 'mean'
                }).round(2)
                age_stats.columns = ['Count', 'Avg Visitors (M)', 'Avg Entry Fee ($)']
                
                # Remove rows with all NaN
                age_stats = age_stats.dropna(how='all')
                
                if len(age_stats) > 0:
                    fig = px.bar(age_stats, x=age_stats.index, y='Avg Visitors (M)',
                                title="Average Visitors by Age Category")
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.markdown("### Correlation Matrix")
        
        # Select numeric columns for correlation
        possible_numeric_cols = ['Annual_Visitors_Millions', 'Entry_Fee_USD',
                               'Tourism_Revenue_Million_USD', 'Average_Visit_Duration_Hours',
                               'Revenue_Per_Visitor_USD', 'Value_Ratio', 'Age']
        
        # Filter only columns that exist in the dataframe and have numeric data
        existing_numeric_cols = []
        for col in possible_numeric_cols:
            if col in filtered_df.columns:
                # Check if column has numeric data (not all NaN)
                if pd.api.types.is_numeric_dtype(filtered_df[col]):
                    if filtered_df[col].notna().sum() > 1:  # Need at least 2 non-NaN values
                        existing_numeric_cols.append(col)
        
        if len(existing_numeric_cols) >= 2:
            # Create correlation matrix
            corr_data = filtered_df[existing_numeric_cols].dropna()
            
            if len(corr_data) >= 2:
                corr_matrix = corr_data.corr()
                
                fig = px.imshow(corr_matrix,
                              text_auto=True,
                              aspect="auto",
                              color_continuous_scale='RdBu',
                              title="Correlation Matrix")
                st.plotly_chart(fig, use_container_width=True)
                
                # Show correlation insights
                st.markdown("#### 💡 Key Insights from Correlations:")
                insights = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_val = corr_matrix.iloc[i, j]
                        if not pd.isna(corr_val) and abs(corr_val) > 0.3:  # Moderate correlation
                            col1 = corr_matrix.columns[i]
                            col2 = corr_matrix.columns[j]
                            strength = "Strong" if abs(corr_val) > 0.7 else "Moderate"
                            direction = "positive" if corr_val > 0 else "negative"
                            insights.append(f"**{col1}** and **{col2}**: {strength} {direction} correlation ({corr_val:.3f})")
                
                if insights:
                    for insight in insights:
                        st.write(f"• {insight}")
                else:
                    st.info("No strong correlations (|r| > 0.3) found in the filtered data.")
            else:
                st.warning("Not enough numeric data for correlation analysis.")
        else:
            st.warning("Need at least 2 numeric columns for correlation analysis.")

def show_geographic_page(filtered_df):
    """Display the geographic analysis page"""
    
    st.markdown('<h2 class="sub-header">🗺️ Geographic Analysis</h2>', unsafe_allow_html=True)
    
    # Check if we have country data
    if 'Country' not in filtered_df.columns:
        st.warning("Country data not available for geographic analysis.")
        return
    
    # Extended country coordinates mapping (covering all countries in your dataset)
    country_coords = {
        'United States': (37.0902, -95.7129),
        'France': (46.6031, 1.8883),
        'Italy': (41.8719, 12.5674),
        'China': (35.8617, 104.1954),
        'United Kingdom': (55.3781, -3.4360),
        'Spain': (40.4637, -3.7492),
        'Australia': (-25.2744, 133.7751),
        'Peru': (-9.1900, -75.0152),
        'Egypt': (26.8206, 30.8025),
        'Greece': (39.0742, 21.8243),
        'United Arab Emirates': (23.4241, 53.8478),
        'Brazil': (-14.2350, -51.9253),
        'India': (20.5937, 78.9629),
        'Cambodia': (12.5657, 104.9910)
    }
    
    # Add coordinates to dataframe - use default for unknown countries
    filtered_df['lat'] = filtered_df['Country'].map(lambda x: country_coords.get(x, (0, 0))[0])
    filtered_df['lon'] = filtered_df['Country'].map(lambda x: country_coords.get(x, (0, 0))[1])
    
    # Filter out places with (0,0) coordinates if they exist
    map_data = filtered_df[~((filtered_df['lat'] == 0) & (filtered_df['lon'] == 0))]
    
    # Create tabs for different map views
    tab1, tab2, tab3 = st.tabs(["🌐 World Map", "📍 Regional View", "📊 Country Comparison"])
    
    with tab1:
        st.markdown("### World Tourism Hotspots")
        
        if len(map_data) > 0:
            # Bubble map
            fig = px.scatter_geo(map_data,
                               lat='lat',
                               lon='lon',
                               size='Annual_Visitors_Millions' if 'Annual_Visitors_Millions' in map_data.columns else None,
                               color='Region' if 'Region' in map_data.columns else None,
                               hover_name='Place_Name',
                               hover_data=['Country', 'Annual_Visitors_Millions', 'Tourism_Revenue_Million_USD'],
                               title="Global Distribution of Famous Places",
                               projection='natural earth')
            
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No geographic data available for mapping.")
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            if 'Region' in filtered_df.columns:
                st.markdown("### Regional Distribution")
                
                region_stats = filtered_df.groupby('Region').agg({
                    'Place_Name': 'count',
                    'Annual_Visitors_Millions': 'sum',
                    'Tourism_Revenue_Million_USD': 'sum'
                }).round(2)
                
                region_stats.columns = ['Count', 'Total Visitors (M)', 'Total Revenue ($M)']
                region_stats = region_stats.dropna()
                
                if len(region_stats) > 0:
                    fig = px.bar(region_stats, x=region_stats.index, y='Total Visitors (M)',
                                color='Total Visitors (M)',
                                title="Total Visitors by Region",
                                color_continuous_scale='Viridis')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show region statistics
                    st.dataframe(region_stats)
            else:
                st.warning("Region data not available.")
        
        with col2:
            st.markdown("### Top Countries")
            
            if 'Annual_Visitors_Millions' in filtered_df.columns:
                country_stats = filtered_df.groupby('Country').agg({
                    'Place_Name': 'count',
                    'Annual_Visitors_Millions': 'sum'
                })
                
                # Get top 10, but handle if less than 10
                top_n = min(10, len(country_stats))
                country_stats = country_stats.nlargest(top_n, 'Annual_Visitors_Millions')
                
                if len(country_stats) > 0:
                    fig = px.pie(country_stats, values='Annual_Visitors_Millions',
                                names=country_stats.index,
                                title=f"Visitor Distribution - Top {top_n} Countries")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                # Fallback to count of places per country
                country_counts = filtered_df['Country'].value_counts().head(10)
                if len(country_counts) > 0:
                    fig = px.bar(x=country_counts.index, y=country_counts.values,
                                title="Number of Places by Country (Top 10)")
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Country comparison tool
        st.markdown("### Country Comparison Tool")
        
        # Select countries to compare
        available_countries = sorted(filtered_df['Country'].unique())
        
        if len(available_countries) > 0:
            selected_countries = st.multiselect(
                "Select countries to compare:",
                available_countries,
                default=available_countries[:min(3, len(available_countries))]
            )
            
            if selected_countries:
                # Filter data for selected countries
                country_data = filtered_df[filtered_df['Country'].isin(selected_countries)]
                
                # Create comparison metrics based on available columns
                agg_dict = {'Place_Name': 'count'}
                
                if 'Annual_Visitors_Millions' in country_data.columns:
                    agg_dict['Annual_Visitors_Millions'] = 'sum'
                
                if 'Tourism_Revenue_Million_USD' in country_data.columns:
                    agg_dict['Tourism_Revenue_Million_USD'] = 'sum'
                
                if 'Entry_Fee_USD' in country_data.columns:
                    agg_dict['Entry_Fee_USD'] = 'mean'
                
                comparison_metrics = country_data.groupby('Country').agg(agg_dict).round(2)
                
                # Rename columns for display
                column_names = {
                    'Place_Name': 'Number of Places',
                    'Annual_Visitors_Millions': 'Total Visitors (M)',
                    'Tourism_Revenue_Million_USD': 'Total Revenue ($M)',
                    'Entry_Fee_USD': 'Avg Entry Fee ($)'
                }
                comparison_metrics = comparison_metrics.rename(columns=column_names)
                
                # Display comparison
                st.dataframe(comparison_metrics)
                
                # Visualization if we have data
                if len(comparison_metrics) > 1:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if 'Total Visitors (M)' in comparison_metrics.columns:
                            fig = px.bar(comparison_metrics, x=comparison_metrics.index,
                                       y='Total Visitors (M)',
                                       title="Total Visitors by Country",
                                       color=comparison_metrics.index)
                            st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        if 'Total Revenue ($M)' in comparison_metrics.columns:
                            fig = px.bar(comparison_metrics, x=comparison_metrics.index,
                                       y='Total Revenue ($M)',
                                       title="Total Revenue by Country",
                                       color=comparison_metrics.index)
                            st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Please select at least one country to compare.")
        else:
            st.warning("No country data available for comparison.")

def show_financial_page(filtered_df):
    """Display the financial insights page"""
    
    st.markdown('<h2 class="sub-header">💰 Financial Insights</h2>', unsafe_allow_html=True)
    
    # Check if we have financial data
    has_financial_data = 'Tourism_Revenue_Million_USD' in filtered_df.columns
    has_visitor_data = 'Annual_Visitors_Millions' in filtered_df.columns
    
    if not has_financial_data and not has_visitor_data:
        st.warning("No financial or visitor data available for analysis.")
        return
    
    # Key financial metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if has_financial_data:
            total_rev = filtered_df['Tourism_Revenue_Million_USD'].sum()
            st.metric("Total Revenue", f"${total_rev:,.0f}M")
        else:
            st.metric("Total Places", len(filtered_df))
    
    with col2:
        if has_financial_data:
            avg_rev_per_place = filtered_df['Tourism_Revenue_Million_USD'].mean()
            st.metric("Avg Revenue per Place", f"${avg_rev_per_place:,.0f}M")
    
    with col3:
        if 'Revenue_Per_Visitor_USD' in filtered_df.columns:
            avg_rev_per_visitor = filtered_df['Revenue_Per_Visitor_USD'].mean()
            st.metric("Avg Revenue per Visitor", f"${avg_rev_per_visitor:.2f}")
    
    with col4:
        if 'Value_Ratio' in filtered_df.columns:
            total_value = (filtered_df['Value_Ratio'] * filtered_df.get('Annual_Visitors_Millions', 1)).sum()
            st.metric("Total Value Index", f"{total_value:,.0f}")
    
    # Financial analysis tabs
    tab_names = []
    if has_financial_data:
        tab_names.append("📈 Revenue Analysis")
    if 'Entry_Fee_USD' in filtered_df.columns:
        tab_names.append("💵 Pricing Strategy")
    if 'Value_Ratio' in filtered_df.columns:
        tab_names.append("🎯 Value Analysis")
    if 'Revenue_Per_Visitor_USD' in filtered_df.columns:
        tab_names.append("📊 ROI Metrics")
    
    if tab_names:
        tabs = st.tabs(tab_names)
        tab_index = 0
        
        if has_financial_data and "📈 Revenue Analysis" in tab_names:
            with tabs[tab_index]:
                show_revenue_analysis(filtered_df)
            tab_index += 1
        
        if "💵 Pricing Strategy" in tab_names:
            with tabs[tab_index]:
                show_pricing_analysis(filtered_df)
            tab_index += 1
        
        if "🎯 Value Analysis" in tab_names:
            with tabs[tab_index]:
                show_value_analysis(filtered_df)
            tab_index += 1
        
        if "📊 ROI Metrics" in tab_names:
            with tabs[tab_index]:
                show_roi_analysis(filtered_df)
    else:
        st.warning("No financial metrics available for detailed analysis.")

def show_revenue_analysis(filtered_df):
    """Show revenue analysis tab content"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Top Revenue Generators")
        if 'Tourism_Revenue_Million_USD' in filtered_df.columns:
            top_n = min(10, len(filtered_df))
            top_revenue = filtered_df.nlargest(top_n, 'Tourism_Revenue_Million_USD')
            
            fig = px.bar(top_revenue, x='Place_Name', y='Tourism_Revenue_Million_USD',
                        title=f"Top {top_n} Revenue Generating Places",
                        color='Tourism_Revenue_Million_USD',
                        color_continuous_scale='Blues')
            fig.update_layout(xaxis_tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'Annual_Visitors_Millions' in filtered_df.columns and 'Tourism_Revenue_Million_USD' in filtered_df.columns:
            st.markdown("### Revenue vs Visitors")
            
            # Filter out NaN values
            scatter_data = filtered_df.dropna(subset=['Annual_Visitors_Millions', 'Tourism_Revenue_Million_USD'])
            
            if len(scatter_data) > 0:
                fig = px.scatter(scatter_data, x='Annual_Visitors_Millions',
                               y='Tourism_Revenue_Million_USD',
                               size='Annual_Visitors_Millions',
                               color='Region' if 'Region' in scatter_data.columns else None,
                               hover_name='Place_Name',
                               title="Revenue vs Number of Visitors")
                               # Removed trendline="ols" to avoid statsmodels dependency
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Calculate and show correlation
                correlation = scatter_data['Annual_Visitors_Millions'].corr(
                    scatter_data['Tourism_Revenue_Million_USD']
                )
                if not pd.isna(correlation):
                    st.info(f"Correlation between Visitors and Revenue: **{correlation:.3f}**")

def show_pricing_analysis(filtered_df):
    """Show pricing analysis tab content"""
    col1, col2 = st.columns(2)
    
    with col1:
        if 'Type' in filtered_df.columns and 'Entry_Fee_USD' in filtered_df.columns:
            st.markdown("### Entry Fee Analysis")
            
            # Entry fee distribution by type
            fee_by_type = filtered_df.groupby('Type')['Entry_Fee_USD'].agg(['mean', 'median', 'max']).round(2)
            fee_by_type.columns = ['Average', 'Median', 'Maximum']
            fee_by_type = fee_by_type.dropna()
            
            if len(fee_by_type) > 0:
                fig = px.bar(fee_by_type, x=fee_by_type.index, y='Average',
                            title="Average Entry Fee by Type")
                fig.update_layout(xaxis_tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
                
                st.dataframe(fee_by_type)
    
    with col2:
        if 'Entry_Fee_USD' in filtered_df.columns and 'Tourism_Revenue_Million_USD' in filtered_df.columns:
            st.markdown("### Fee vs Revenue Relationship")
            
            scatter_data = filtered_df.dropna(subset=['Entry_Fee_USD', 'Tourism_Revenue_Million_USD'])
            
            if len(scatter_data) > 0:
                fig = px.scatter(scatter_data, x='Entry_Fee_USD',
                               y='Tourism_Revenue_Million_USD',
                               size='Annual_Visitors_Millions' if 'Annual_Visitors_Millions' in scatter_data.columns else None,
                               color='Type' if 'Type' in scatter_data.columns else None,
                               hover_name='Place_Name',
                               title="Entry Fee vs Total Revenue")
                
                st.plotly_chart(fig, use_container_width=True)

def show_value_analysis(filtered_df):
    """Show value analysis tab content"""
    st.markdown("### Value for Money Analysis")
    
    # Ensure Value_Ratio exists
    if 'Value_Ratio' not in filtered_df.columns:
        # Calculate it if we have the components
        if 'Average_Visit_Duration_Hours' in filtered_df.columns and 'Entry_Fee_USD' in filtered_df.columns:
            filtered_df['Value_Ratio'] = filtered_df['Average_Visit_Duration_Hours'] / (filtered_df['Entry_Fee_USD'].replace(0, np.nan) + 0.01)
        else:
            st.warning("Cannot calculate value ratio without duration and fee data.")
            return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Best Value Places")
        top_n = min(10, len(filtered_df))
        best_value = filtered_df.nlargest(top_n, 'Value_Ratio')[['Place_Name', 'Country', 'Value_Ratio', 'Entry_Fee_USD']]
        best_value = best_value.dropna()
        
        if len(best_value) > 0:
            fig = px.bar(best_value, x='Place_Name', y='Value_Ratio',
                        title=f"Top {len(best_value)} Best Value Places",
                        color='Entry_Fee_USD',
                        color_continuous_scale='Greens')
            fig.update_layout(xaxis_tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Worst Value Places")
        worst_value = filtered_df.nsmallest(top_n, 'Value_Ratio')[['Place_Name', 'Country', 'Value_Ratio', 'Entry_Fee_USD']]
        worst_value = worst_value.dropna()
        
        if len(worst_value) > 0:
            fig = px.bar(worst_value, x='Place_Name', y='Value_Ratio',
                        title=f"Top {len(worst_value)} Worst Value Places",
                        color='Entry_Fee_USD',
                        color_continuous_scale='Reds')
            fig.update_layout(xaxis_tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
    
    # Value analysis insights
    st.markdown("#### 💡 Value Analysis Insights")
    
    if 'Value_Ratio' in filtered_df.columns:
        value_data = filtered_df['Value_Ratio'].dropna()
        if len(value_data) > 0:
            avg_value_score = value_data.mean()
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Average Value Score", f"{avg_value_score:.1f}")
            
            if 'Entry_Fee_USD' in filtered_df.columns:
                median_fee = filtered_df['Entry_Fee_USD'].median()
                free_places = (filtered_df['Entry_Fee_USD'] == 0).sum()
                with col2:
                    st.metric("Median Entry Fee", f"${median_fee:.1f}")
                with col3:
                    st.metric("Free Entry Places", free_places)

def show_roi_analysis(filtered_df):
    """Show ROI analysis tab content"""
    st.markdown("### Return on Investment Metrics")
    
    # Calculate ROI if we have the data
    if 'Tourism_Revenue_Million_USD' in filtered_df.columns and 'Entry_Fee_USD' in filtered_df.columns:
        # Avoid division by zero
        filtered_df['ROI_Ratio'] = filtered_df['Tourism_Revenue_Million_USD'] / (filtered_df['Entry_Fee_USD'].replace(0, np.nan) + 0.01)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Highest ROI Places")
            top_n = min(10, len(filtered_df))
            high_roi = filtered_df.nlargest(top_n, 'ROI_Ratio')[['Place_Name', 'Country', 'ROI_Ratio', 'Entry_Fee_USD']]
            high_roi = high_roi.dropna()
            
            if len(high_roi) > 0:
                fig = px.bar(high_roi, x='Place_Name', y='ROI_Ratio',
                            title=f"Top {len(high_roi)} ROI Places",
                            color='Entry_Fee_USD')
                fig.update_layout(xaxis_tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### ROI Distribution")
            
            roi_data = filtered_df['ROI_Ratio'].dropna()
            if len(roi_data) > 0:
                fig = px.histogram(filtered_df, x='ROI_Ratio', nbins=30,
                                title="Distribution of ROI Ratios",
                                color_discrete_sequence=['#FFA07A'])
                st.plotly_chart(fig, use_container_width=True)
        
        # ROI insights
        st.markdown("#### 📊 ROI Statistics")
        
        roi_stats = filtered_df['ROI_Ratio'].describe().round(2)
        st.dataframe(roi_stats)
    else:
        st.warning("Cannot calculate ROI without revenue and fee data.")

def show_explorer_page(filtered_df):
    """Display the detailed explorer page with images"""
    
    st.markdown('<h2 class="sub-header">🔍 Detailed Place Explorer</h2>', unsafe_allow_html=True)
    
    # Check if we have place names
    if 'Place_Name' not in filtered_df.columns:
        st.warning("Place name data not available for explorer.")
        return
    
    if len(filtered_df) == 0:
        st.warning("No data available with current filters.")
        return
    
    # Place selector
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Search and select place
        place_options = sorted(filtered_df['Place_Name'].unique())
        selected_place = st.selectbox("Search and select a place:", place_options)
    
    with col2:
        # Quick comparison
        st.markdown("### Compare with:")
        compare_options = ['None'] + [p for p in place_options if p != selected_place]
        compare_place = st.selectbox("Select place to compare:", compare_options, index=0)
    
    # Get selected place data
    place_data = filtered_df[filtered_df['Place_Name'] == selected_place].iloc[0]
    
    if compare_place != 'None':
        compare_data = filtered_df[filtered_df['Place_Name'] == compare_place].iloc[0]
        comparison_mode = True
    else:
        comparison_mode = False
    
    # Display place information
    if not comparison_mode:
        # Single place view
        st.markdown("---")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Display image or placeholder
            display_place_image(
                selected_place,
                place_data.get('Type', 'Place'),
                place_data.get('Country', '')
            )
        
        with col2:
            st.markdown(f"## {place_data['Place_Name']}")
            
            # Safely display location info
            location_parts = []
            if 'City' in place_data and pd.notna(place_data['City']):
                location_parts.append(str(place_data['City']))
            if 'Country' in place_data and pd.notna(place_data['Country']):
                location_parts.append(str(place_data['Country']))
            
            if location_parts:
                st.markdown(f"**📍 Location:** {', '.join(location_parts)}")
            
            if 'Type' in place_data and pd.notna(place_data['Type']):
                st.markdown(f"**🏛️ Type:** {place_data['Type']}")
            
            if 'Region' in place_data and pd.notna(place_data['Region']):
                st.markdown(f"**🌍 Region:** {place_data['Region']}")
            
            st.markdown("### 📊 Key Metrics")
            
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            
            with metric_col1:
                if 'Annual_Visitors_Millions' in place_data and pd.notna(place_data['Annual_Visitors_Millions']):
                    st.metric("Annual Visitors", f"{place_data['Annual_Visitors_Millions']:.1f}M")
            
            with metric_col2:
                if 'Tourism_Revenue_Million_USD' in place_data and pd.notna(place_data['Tourism_Revenue_Million_USD']):
                    st.metric("Tourism Revenue", f"${place_data['Tourism_Revenue_Million_USD']:,.0f}M")
            
            with metric_col3:
                if 'Entry_Fee_USD' in place_data and pd.notna(place_data['Entry_Fee_USD']):
                    st.metric("Entry Fee", f"${place_data['Entry_Fee_USD']:.0f}")
        
        # Detailed information
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📍 Details")
            details = {}
            
            if 'UNESCO_World_Heritage' in place_data:
                unesco_status = "✅ Yes" if place_data['UNESCO_World_Heritage'] == 'Yes' else "❌ No"
                details["UNESCO Status"] = unesco_status
            
            if 'Year_Built' in place_data:
                if pd.notna(place_data['Year_Built']):
                    details["Year Built"] = f"{int(place_data['Year_Built'])}"
                else:
                    details["Year Built"] = "Natural Formation"
            
            if 'Best_Visit_Month' in place_data and pd.notna(place_data['Best_Visit_Month']):
                details["Best Time to Visit"] = place_data['Best_Visit_Month']
            
            if 'Average_Visit_Duration_Hours' in place_data and pd.notna(place_data['Average_Visit_Duration_Hours']):
                details["Average Visit Duration"] = f"{place_data['Average_Visit_Duration_Hours']} hours"
            
            if 'Age' in place_data and pd.notna(place_data['Age']):
                details["Age"] = f"{int(place_data['Age'])} years" if place_data['Age'] > 0 else "Natural Formation"
            
            if 'Best_Season' in place_data and pd.notna(place_data['Best_Season']):
                details["Best Season"] = place_data['Best_Season']
            
            for key, value in details.items():
                st.markdown(f"**{key}:** {value}")
        
        with col2:
            st.markdown("### 💰 Financial Metrics")
            metrics = {}
            
            if 'Revenue_Per_Visitor_USD' in place_data and pd.notna(place_data['Revenue_Per_Visitor_USD']):
                metrics["Revenue per Visitor"] = f"${place_data['Revenue_Per_Visitor_USD']:.2f}"
            
            if 'Value_Ratio' in place_data and pd.notna(place_data['Value_Ratio']):
                metrics["Value Ratio"] = f"{place_data['Value_Ratio']:.2f} hours per dollar"
            
            if 'Revenue_Per_Visitor_USD' in place_data and place_data['Revenue_Per_Visitor_USD'] > 0:
                metrics["Visitors per Revenue Dollar"] = f"{1/place_data['Revenue_Per_Visitor_USD']:.4f}"
            
            for key, value in metrics.items():
                st.markdown(f"**{key}:** {value}")
        
        # What it's famous for
        if 'Famous_For' in place_data and pd.notna(place_data['Famous_For']):
            st.markdown("### 🌟 Famous For")
            st.info(place_data['Famous_For'])
        
        # Create radar chart for place characteristics if we have enough data
        st.markdown("### 📈 Place Profile")
        
        # Select metrics that exist and have numeric data
        possible_metrics = ['Annual_Visitors_Millions', 'Entry_Fee_USD', 
                          'Tourism_Revenue_Million_USD', 'Average_Visit_Duration_Hours']
        
        available_metrics = []
        normalized_values = []
        categories = []
        
        for metric in possible_metrics:
            if metric in filtered_df.columns and metric in place_data:
                if pd.api.types.is_numeric_dtype(filtered_df[metric]) and pd.notna(place_data[metric]):
                    val = place_data[metric]
                    min_val = filtered_df[metric].min()
                    max_val = filtered_df[metric].max()
                    
                    if max_val > min_val:
                        normalized = (val - min_val) / (max_val - min_val) * 100
                    else:
                        normalized = 50  # Default middle value
                    
                    available_metrics.append(metric)
                    normalized_values.append(normalized)
                    
                    # Create readable category names
                    category_names = {
                        'Annual_Visitors_Millions': 'Visitors',
                        'Entry_Fee_USD': 'Entry Fee',
                        'Tourism_Revenue_Million_USD': 'Revenue',
                        'Average_Visit_Duration_Hours': 'Visit Duration'
                    }
                    categories.append(category_names.get(metric, metric))
        
        if len(normalized_values) >= 3:  # Need at least 3 points for radar chart
            fig = go.Figure(data=go.Scatterpolar(
                r=normalized_values,
                theta=categories,
                fill='toself',
                name=place_data['Place_Name']
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )),
                showlegend=True,
                title="Place Characteristics (Normalized 0-100)"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough numeric data for radar chart visualization.")
    
    else:
        # Comparison view with images
        st.markdown("---")
        st.markdown(f"## 🆚 Comparison: {place_data['Place_Name']} vs {compare_data['Place_Name']}")
        
        # Display images side by side
        col1, col2 = st.columns(2)
        
        with col1:
            display_place_image(
                place_data['Place_Name'],
                place_data.get('Type', 'Place'),
                place_data.get('Country', '')
            )
        
        with col2:
            display_place_image(
                compare_data['Place_Name'],
                compare_data.get('Type', 'Place'),
                compare_data.get('Country', '')
            )
        
        # Create comparison metrics
        comparison_rows = []
        
        # Define metrics to compare
        metric_definitions = [
            ('Annual_Visitors_Millions', 'Annual Visitors (M)', lambda x: f"{x:.1f}"),
            ('Entry_Fee_USD', 'Entry Fee (USD)', lambda x: f"${x:.0f}"),
            ('Tourism_Revenue_Million_USD', 'Tourism Revenue (M USD)', lambda x: f"${x:,.0f}"),
            ('Average_Visit_Duration_Hours', 'Average Visit Duration (hours)', lambda x: f"{x:.1f}"),
            ('Revenue_Per_Visitor_USD', 'Revenue per Visitor (USD)', lambda x: f"${x:.2f}"),
            ('Value_Ratio', 'Value Ratio', lambda x: f"{x:.2f}"),
            ('Age', 'Age (years)', lambda x: f"{int(x)}" if pd.notna(x) else "N/A"),
            ('UNESCO_World_Heritage', 'UNESCO Status', lambda x: x if pd.notna(x) else "N/A")
        ]
        
        for metric_key, display_name, formatter in metric_definitions:
            if metric_key in place_data and metric_key in compare_data:
                place_val = place_data[metric_key] if pd.notna(place_data[metric_key]) else None
                compare_val = compare_data[metric_key] if pd.notna(compare_data[metric_key]) else None
                
                comparison_rows.append({
                    'Metric': display_name,
                    place_data['Place_Name']: formatter(place_val) if place_val is not None else "N/A",
                    compare_data['Place_Name']: formatter(compare_val) if compare_val is not None else "N/A"
                })
        
        if comparison_rows:
            comparison_df = pd.DataFrame(comparison_rows)
            
            # Display comparison table
            st.dataframe(comparison_df, use_container_width=True)
            
            # Visualization comparison for numeric metrics
            numeric_metrics = ['Annual_Visitors_Millions', 'Entry_Fee_USD', 
                             'Tourism_Revenue_Million_USD', 'Average_Visit_Duration_Hours']
            
            # Filter to only metrics that exist and have numeric data
            available_numeric = []
            place_vals = []
            compare_vals = []
            display_names = []
            
            for metric in numeric_metrics:
                if (metric in place_data and pd.notna(place_data[metric]) and 
                    metric in compare_data and pd.notna(compare_data[metric])):
                    
                    available_numeric.append(metric)
                    place_vals.append(place_data[metric])
                    compare_vals.append(compare_data[metric])
                    
                    # Create display names
                    name_map = {
                        'Annual_Visitors_Millions': 'Visitors (M)',
                        'Tourism_Revenue_Million_USD': 'Revenue (M USD)',
                        'Entry_Fee_USD': 'Entry Fee ($)',
                        'Average_Visit_Duration_Hours': 'Duration (hrs)'
                    }
                    display_names.append(name_map.get(metric, metric))
            
            if len(available_numeric) >= 2:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Bar chart comparison
                    fig = go.Figure(data=[
                        go.Bar(name=place_data['Place_Name'], 
                              x=display_names[:3],  # Show first 3 metrics
                              y=place_vals[:3]),
                        go.Bar(name=compare_data['Place_Name'],
                              x=display_names[:3],
                              y=compare_vals[:3])
                    ])
                    
                    fig.update_layout(barmode='group', title="Direct Comparison")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Radar chart comparison if we have enough points
                    if len(available_numeric) >= 3:
                        # Normalize values for radar chart
                        all_values = place_vals + compare_vals
                        min_val = min(all_values)
                        max_val = max(all_values)
                        
                        if max_val > min_val:
                            normalized_place = [(v - min_val) / (max_val - min_val) * 100 for v in place_vals]
                            normalized_compare = [(v - min_val) / (max_val - min_val) * 100 for v in compare_vals]
                        else:
                            normalized_place = [50] * len(place_vals)
                            normalized_compare = [50] * len(compare_vals)
                        
                        fig = go.Figure()
                        
                        fig.add_trace(go.Scatterpolar(
                            r=normalized_place,
                            theta=display_names,
                            fill='toself',
                            name=place_data['Place_Name']
                        ))
                        
                        fig.add_trace(go.Scatterpolar(
                            r=normalized_compare,
                            theta=display_names,
                            fill='toself',
                            name=compare_data['Place_Name']
                        ))
                        
                        fig.update_layout(
                            polar=dict(
                                radialaxis=dict(
                                    visible=True,
                                    range=[0, 100]
                                )),
                            showlegend=True,
                            title="Comparison Radar Chart"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
            
            # What they're famous for comparison
            col1, col2 = st.columns(2)
            
            with col1:
                if 'Famous_For' in place_data and pd.notna(place_data['Famous_For']):
                    st.markdown(f"### 🌟 {place_data['Place_Name']}")
                    st.info(place_data['Famous_For'])
            
            with col2:
                if 'Famous_For' in compare_data and pd.notna(compare_data['Famous_For']):
                    st.markdown(f"### 🌟 {compare_data['Place_Name']}")
                    st.info(compare_data['Famous_For'])
        else:
            st.warning("No comparable metrics available for these places.")
    
    # Download button for current view
    st.markdown("---")
    if comparison_mode:
        csv = comparison_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Comparison Data",
            data=csv,
            file_name=f"comparison_{place_data['Place_Name']}_vs_{compare_data['Place_Name']}.csv",
            mime="text/csv"
        )
    else:
        place_details = pd.DataFrame([place_data])
        csv = place_details.to_csv(index=False)
        st.download_button(
            label="📥 Download Place Details",
            data=csv,
            file_name=f"{place_data['Place_Name'].replace(' ', '_')}_details.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()