import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import random

# Page configuration
st.set_page_config(
    page_title="Demo Dashboard",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        font-weight: bold;
        padding: 1rem 0;
        border-bottom: 3px solid #1E3A8A;
    }
    .metric-card {
        background-color: #F8F9FA;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1E3A8A;
        margin: 0.5rem 0;
    }
    .drug-category {
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.2rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for interactivity
if 'selected_drug' not in st.session_state:
    st.session_state.selected_drug = None
if 'selected_region' not in st.session_state:
    st.session_state.selected_region = 'All Regions'

# Title and Header
st.markdown("<h1 class='main-header'>💊 Demo Dashboard</h1>", unsafe_allow_html=True)

# Sidebar for filters
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/206/206853.png", width=100)
    st.title("Filters & Controls")
    
    # Date range filter
    st.subheader("📅 Date Range")
    date_range = st.date_input(
        "Select Period",
        value=[datetime.now() - timedelta(days=365), datetime.now()],
        max_value=datetime.now()
    )
    
    # Region filter
    st.subheader("🌍 Region Filter")
    regions = ['North America', 'Europe', 'Asia Pacific', 'Latin America', 'Middle East', 'All Regions']
    selected_region = st.selectbox("Select Region", regions)
    st.session_state.selected_region = selected_region
    
    # Drug category filter
    st.subheader("📊 Drug Categories")
    categories = ['All Categories', 'Oncology', 'Cardiology', 'Neurology', 'Diabetes', 'Immunology']
    selected_category = st.multiselect("Select Drug Categories", categories, default=['All Categories'])
    
    # KPI threshold
    st.subheader("⚡ Performance Threshold")
    threshold = st.slider("Sales Target (in $M)", min_value=0, max_value=500, value=100)
    
    st.divider()
    st.info("💡 Click on charts to filter data")
    if st.button("🔄 Reset All Filters"):
        st.session_state.selected_drug = None
        st.session_state.selected_region = 'All Regions'
        st.rerun()

# Generate synthetic pharmaceutical data
@st.cache_data
def generate_pharma_data():
    np.random.seed(42)
    
    # Drugs data
    drugs = [
        {'Drug_ID': 'DRG001', 'Name': 'Oncovin Plus', 'Category': 'Oncology', 
         'Launch_Date': '2020-03-15', 'Patent_Expiry': '2035-06-30', 'Status': 'Approved'},
        {'Drug_ID': 'DRG002', 'Name': 'CardioSafe', 'Category': 'Cardiology', 
         'Launch_Date': '2019-07-22', 'Patent_Expiry': '2032-12-31', 'Status': 'Approved'},
        {'Drug_ID': 'DRG003', 'Name': 'NeuroRelief', 'Category': 'Neurology', 
         'Launch_Date': '2021-01-10', 'Patent_Expiry': '2036-08-15', 'Status': 'Phase III'},
        {'Drug_ID': 'DRG004', 'Name': 'GlucoBalance', 'Category': 'Diabetes', 
         'Launch_Date': '2018-11-05', 'Patent_Expiry': '2030-05-20', 'Status': 'Approved'},
        {'Drug_ID': 'DRG005', 'Name': 'ImmuneBoost', 'Category': 'Immunology', 
         'Launch_Date': '2022-06-18', 'Patent_Expiry': '2037-02-28', 'Status': 'Phase II'},
        {'Drug_ID': 'DRG006', 'Name': 'PainAway XR', 'Category': 'Neurology', 
         'Launch_Date': '2017-09-12', 'Patent_Expiry': '2028-11-30', 'Status': 'Approved'},
    ]
    
    # Sales data
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='M')
    sales_data = []
    
    for date in dates:
        for drug in drugs:
            base_sales = {
                'Oncology': 120, 'Cardiology': 85, 'Neurology': 70, 
                'Diabetes': 60, 'Immunology': 40
            }
            base = base_sales.get(drug['Category'], 50)
            growth = 1 + 0.02 * (date.month - 1)  # 2% monthly growth
            seasonal = 1.1 if date.month in [11, 12] else 0.95 if date.month in [6, 7] else 1
            
            for region in ['North America', 'Europe', 'Asia Pacific', 'Latin America', 'Middle East']:
                region_multiplier = {
                    'North America': 1.4,
                    'Europe': 1.2,
                    'Asia Pacific': 1.1,
                    'Latin America': 0.8,
                    'Middle East': 0.6
                }
                
                sales = base * growth * seasonal * region_multiplier[region] * np.random.uniform(0.9, 1.1)
                
                sales_data.append({
                    'Date': date,
                    'Drug_ID': drug['Drug_ID'],
                    'Drug_Name': drug['Name'],
                    'Category': drug['Category'],
                    'Region': region,
                    'Sales_USD_M': round(sales, 2),
                    'Units_Sold_K': round(sales * np.random.uniform(8, 12), 0),
                    'Market_Share_%': round(np.random.uniform(5, 25), 1)
                })
    
    # Clinical trials data
    trials = []
    phases = ['Phase I', 'Phase II', 'Phase III', 'Approval']
    for drug in drugs:
        for phase in phases:
            status = random.choice(['Active', 'Completed', 'On Hold', 'Terminated'])
            patients = random.randint(50, 500) if phase == 'Phase I' else \
                      random.randint(200, 2000) if phase == 'Phase II' else \
                      random.randint(1000, 5000)
            
            trials.append({
                'Drug_ID': drug['Drug_ID'],
                'Drug_Name': drug['Name'],
                'Phase': phase,
                'Status': status,
                'Patients_Enrolled': patients,
                'Success_Rate_%': round(random.uniform(60, 95), 1),
                'Estimated_Completion': (
                    datetime.now() + timedelta(days=random.randint(60, 730))
                ).strftime('%Y-%m-%d')
            })
    
    # Inventory data
    inventory = []
    for drug in drugs:
        for region in ['North America', 'Europe', 'Asia Pacific']:
            inventory.append({
                'Drug_ID': drug['Drug_ID'],
                'Drug_Name': drug['Name'],
                'Region': region,
                'Current_Stock_K': random.randint(50, 500),
                'Reorder_Level_K': random.randint(20, 100),
                'Lead_Time_Days': random.randint(14, 60),
                'Warehouse': random.choice(['WH-1', 'WH-2', 'WH-3'])
            })
    
    return pd.DataFrame(drugs), pd.DataFrame(sales_data), pd.DataFrame(trials), pd.DataFrame(inventory)

# Load data
drugs_df, sales_df, trials_df, inventory_df = generate_pharma_data()

# Filter data based on sidebar selections
if st.session_state.selected_region != 'All Regions':
    filtered_sales = sales_df[sales_df['Region'] == st.session_state.selected_region]
else:
    filtered_sales = sales_df

if 'All Categories' not in selected_category:
    filtered_sales = filtered_sales[filtered_sales['Category'].isin(selected_category)]

# Top Metrics Row
col1, col2, col3, col4 = st.columns(4)

with col1:
    total_sales = filtered_sales['Sales_USD_M'].sum()
    st.metric(
        label="💰 Total Sales",
        value=f"${total_sales:,.0f}M",
        delta=f"{(total_sales / len(filtered_sales['Date'].unique()) - 100):+.1f}% MoM"
    )

with col2:
    avg_market_share = filtered_sales['Market_Share_%'].mean()
    st.metric(
        label="📈 Avg Market Share",
        value=f"{avg_market_share:.1f}%",
        delta=f"{(avg_market_share - 15):+.1f}% vs Target"
    )

with col3:
    active_trials = len(trials_df[trials_df['Status'] == 'Active'])
    st.metric(
        label="🔬 Active Trials",
        value=active_trials,
        delta=f"+{random.randint(1, 5)} this quarter"
    )

with col4:
    low_inventory = inventory_df[inventory_df['Current_Stock_K'] < inventory_df['Reorder_Level_K']]
    st.metric(
        label="⚠️ Low Inventory Items",
        value=len(low_inventory),
        delta=f"-{random.randint(0, 2)} resolved"
    )

# Main Dashboard - 2 columns layout
col_left, col_right = st.columns([2, 1])

with col_left:
    # Sales Trend Chart
    st.subheader("📊 Monthly Sales Trend")
    
    monthly_sales = filtered_sales.groupby(['Date', 'Category'])['Sales_USD_M'].sum().reset_index()
    
    fig_sales = px.line(
        monthly_sales, 
        x='Date', 
        y='Sales_USD_M', 
        color='Category',
        title="Sales Trend by Drug Category",
        markers=True,
        height=400
    )
    fig_sales.update_layout(
        hovermode='x unified',
        xaxis_title="Month",
        yaxis_title="Sales ($ Millions)",
        legend_title="Drug Category"
    )
    st.plotly_chart(fig_sales, use_container_width=True)
    
    # Regional Performance Map
    st.subheader("🌍 Regional Sales Performance")
    
    region_sales = filtered_sales.groupby('Region')['Sales_USD_M'].sum().reset_index()
    
    # Create a custom color scale based on sales
    fig_map = go.Figure(data=go.Choropleth(
        locations=region_sales['Region'],
        z=region_sales['Sales_USD_M'].astype(float),
        locationmode='country names',
        colorscale='Blues',
        text=region_sales['Region'],
        hoverinfo='location+z',
        marker_line_color='darkgray',
        marker_line_width=0.5,
        colorbar_title="Sales ($M)"
    ))
    
    fig_map.update_layout(
        title_text='Pharmaceutical Sales by Region',
        geo=dict(
            showframe=False,
            showcoastlines=True,
            projection_type='equirectangular'
        ),
        height=400
    )
    st.plotly_chart(fig_map, use_container_width=True)

with col_right:
    # Top Performing Drugs
    st.subheader("🏆 Top Performing Drugs")
    
    top_drugs = filtered_sales.groupby('Drug_Name')['Sales_USD_M'].sum().nlargest(5).reset_index()
    
    fig_top = px.bar(
        top_drugs,
        y='Drug_Name',
        x='Sales_USD_M',
        orientation='h',
        color='Sales_USD_M',
        color_continuous_scale='Viridis',
        text='Sales_USD_M',
        height=300
    )
    fig_top.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        xaxis_title="Sales ($ Millions)",
        yaxis_title="",
        showlegend=False
    )
    fig_top.update_traces(texttemplate='$%{x:.0f}M', textposition='outside')
    st.plotly_chart(fig_top, use_container_width=True)
    
    # Clinical Trials Status
    st.subheader("🔬 Clinical Trials Overview")
    
    trials_summary = trials_df.groupby(['Phase', 'Status']).size().reset_index(name='Count')
    
    fig_trials = px.sunburst(
        trials_summary,
        path=['Phase', 'Status'],
        values='Count',
        color='Phase',
        color_discrete_sequence=px.colors.qualitative.Set3,
        height=300
    )
    fig_trials.update_layout(margin=dict(t=0, l=0, r=0, b=0))
    st.plotly_chart(fig_trials, use_container_width=True)
    
    # Inventory Alert
    st.subheader("📦 Inventory Status")
    
    inventory_status = inventory_df.copy()
    inventory_status['Status'] = np.where(
        inventory_status['Current_Stock_K'] < inventory_status['Reorder_Level_K'],
        'Low Stock',
        'Adequate'
    )
    
    status_count = inventory_status['Status'].value_counts()
    fig_inventory = go.Figure(data=[go.Pie(
        labels=status_count.index,
        values=status_count.values,
        hole=.4,
        marker_colors=['#FF6B6B', '#4ECDC4']
    )])
    fig_inventory.update_layout(height=250, showlegend=True)
    st.plotly_chart(fig_inventory, use_container_width=True)

# Bottom Section - Detailed Data Tables
st.divider()
tab1, tab2, tab3 = st.tabs(["📋 Drug Portfolio", "📈 Sales Details", "📊 Clinical Trials"])

# with tab1:
#     col1, col2 = st.columns(2)
    
#     with col1:
#         st.dataframe(
#             drugs_df[['Name', 'Category', 'Launch_Date', 'Patent_Expiry', 'Status']]
#             .sort_values('Launch_Date', ascending=False)
#             .style.background_gradient(subset=['Category'], cmap='Pastel1')
#             .format({'Launch_Date': lambda x: x.strftime('%Y-%m-%d') 
#                     if pd.notnull(x) else ''}),
#             height=300
#         )

# with tab1:
#     col1, col2 = st.columns(2)
    
#     with col1:
#         # 1. Define a function to map categories to colors
#         def color_category(val):
#             color_map = {
#                 'Oncology': 'background-color: #FFD1DC',   # Light Pink
#                 'Cardiology': 'background-color: #B3E5FC', # Light Blue
#                 'Neurology': 'background-color: #C8E6C9',  # Light Green
#                 'Diabetes': 'background-color: #FFF9C4',   # Light Yellow
#                 'Immunology': 'background-color: #E1BEE7'  # Light Purple
#             }
#             return color_map.get(val, '')

#         # 2. Apply the styling using .map() instead of background_gradient
#         st.dataframe(
#             drugs_df[['Name', 'Category', 'Launch_Date', 'Patent_Expiry', 'Status']]
#             .sort_values('Launch_Date', ascending=False)
#             .style.map(color_category, subset=['Category']) # Use .map for text columns
#             .format({
#                 'Launch_Date': lambda x: x.strftime('%Y-%m-%d') if pd.notnull(x) else '',
#                 'Patent_Expiry': lambda x: x.strftime('%Y-%m-%d') if pd.notnull(x) else ''
#             }),
#             height=300,
#             use_container_width=True
#         )
    
#     with col2:
#         # Patent Expiry Timeline
#         st.subheader("📅 Patent Expiry Timeline")
#         drugs_df['Years_to_Expiry'] = (
#             pd.to_datetime(drugs_df['Patent_Expiry']) - pd.Timestamp.now()
#         ).dt.days / 365.25
        
#         fig_patent = px.timeline(
#             drugs_df,
#             x_start="Launch_Date",
#             x_end="Patent_Expiry",
#             y="Name",
#             color="Years_to_Expiry",
#             color_continuous_scale="RdYlGn_r",
#             title="Drug Patent Timeline"
#         )
#         fig_patent.update_layout(height=350)
#         st.plotly_chart(fig_patent, use_container_width=True)

# with tab1:
#     col1, col2 = st.columns(2)
    
#     with col1:
#         # Define custom color logic
#         def color_category(val):
#             color_map = {
#                 'Oncology': 'background-color: #FFD1DC',
#                 'Cardiology': 'background-color: #B3E5FC',
#                 'Neurology': 'background-color: #C8E6C9',
#                 'Diabetes': 'background-color: #FFF9C4',
#                 'Immunology': 'background-color: #E1BEE7'
#             }
#             return color_map.get(val, '')

#         st.dataframe(
#             drugs_df[['Name', 'Category', 'Launch_Date', 'Patent_Expiry', 'Status']]
#             .sort_values('Launch_Date', ascending=False)
#             .style.map(color_category, subset=['Category'])  # Fixed: uses .map instead of gradient
#             .format({
#                 'Launch_Date': lambda x: x.strftime('%Y-%m-%d') if pd.notnull(x) else '',
#                 'Patent_Expiry': lambda x: x.strftime('%Y-%m-%d') if pd.notnull(x) else ''
#             }),
#             height=300,
#             use_container_width=True
#         )

#     with col2:
#         st.subheader("📅 Patent Expiry Timeline")
#         # Recalculate expiry using the converted dates
#         drugs_df['Years_to_Expiry'] = (
#             drugs_df['Patent_Expiry'] - pd.Timestamp.now()
#         ).dt.days / 365.25
        
#         fig_patent = px.timeline(
#             drugs_df,
#             x_start="Launch_Date",
#             x_end="Patent_Expiry",
#             y="Name",
#             color="Years_to_Expiry",
#             color_continuous_scale="RdYlGn_r",
#             title="Drug Patent Timeline"
#         )
#         fig_patent.update_layout(height=350)
#         st.plotly_chart(fig_patent, use_container_width=True)

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        # 1. Define the color map
        def color_category(val):
            color_map = {
                'Oncology': 'background-color: #FFD1DC',
                'Cardiology': 'background-color: #B3E5FC',
                'Neurology': 'background-color: #C8E6C9',
                'Diabetes': 'background-color: #FFF9C4',
                'Immunology': 'background-color: #E1BEE7'
            }
            return color_map.get(val, '')

        # 2. Display DataFrame WITHOUT .format()
        # The dates are already strings like '2020-03-15', so they display correctly automatically.
        st.dataframe(
            drugs_df[['Name', 'Category', 'Launch_Date', 'Patent_Expiry', 'Status']]
            .sort_values('Launch_Date', ascending=False)
            .style.map(color_category, subset=['Category']), 
            height=300,
            use_container_width=True
        )

    with col2:
        st.subheader("📅 Patent Expiry Timeline")
        
        # We convert to datetime just for this calculation locally
        # This prevents breaking the original dataframe
        drugs_df['Years_to_Expiry'] = (
            pd.to_datetime(drugs_df['Patent_Expiry']) - pd.Timestamp.now()
        ).dt.days / 365.25
        
        fig_patent = px.timeline(
            drugs_df,
            x_start="Launch_Date",
            x_end="Patent_Expiry",
            y="Name",
            color="Years_to_Expiry",
            color_continuous_scale="RdYlGn_r",
            title="Drug Patent Timeline"
        )
        fig_patent.update_layout(height=350)
        st.plotly_chart(fig_patent, use_container_width=True)

# with tab2:
#     # Interactive sales table with filtering
#     st.subheader("Detailed Sales Data")
    
#     # Add quarter calculation
#     filtered_sales['Quarter'] = filtered_sales['Date'].dt.to_period('Q').astype(str)
    
#     # Group by multiple dimensions
#     pivot_sales = pd.pivot_table(
#         filtered_sales,
#         values='Sales_USD_M',
#         index=['Drug_Name', 'Category'],
#         columns='Quarter',
#         aggfunc='sum',
#         fill_value=0
#     ).reset_index()
    
#     st.dataframe(
#         pivot_sales.style
#         .background_gradient(subset=pivot_sales.columns[2:], cmap='YlOrRd')
#         .format('${:,.0f}'),
#         height=400,
#         use_container_width=True
#     )
    
#     # Download button
#     csv = pivot_sales.to_csv(index=False).encode('utf-8')
#     st.download_button(
#         label="📥 Download Sales Data",
#         data=csv,
#         file_name="pharma_sales_data.csv",
#         mime="text/csv",
#     )

with tab2:
    # Interactive sales table with filtering
    st.subheader("Detailed Sales Data")
    
    # Add quarter calculation
    filtered_sales['Quarter'] = filtered_sales['Date'].dt.to_period('Q').astype(str)
    
    # Group by multiple dimensions
    pivot_sales = pd.pivot_table(
        filtered_sales,
        values='Sales_USD_M',
        index=['Drug_Name', 'Category'],
        columns='Quarter',
        aggfunc='sum',
        fill_value=0
    ).reset_index()
    
    # Fix: Apply formatting only to numeric columns (skip first 2 columns)
    st.dataframe(
        pivot_sales.style
        .background_gradient(subset=pivot_sales.columns[2:], cmap='YlOrRd')
        .format('${:,.0f}', subset=pivot_sales.columns[2:]),
        height=400,
        use_container_width=True
    )
    
    # Download button
    csv = pivot_sales.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Sales Data",
        data=csv,
        file_name="pharma_sales_data.csv",
        mime="text/csv",
    )

with tab3:
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.dataframe(
            trials_df.sort_values(['Phase', 'Success_Rate_%'], ascending=[True, False])
            .style.bar(subset=['Success_Rate_%'], color='#5fba7d')
            .highlight_max(subset=['Success_Rate_%'], color='#fffd75')
            .highlight_min(subset=['Success_Rate_%'], color='#ff7f7f'),
            height=400,
            use_container_width=True
        )
    
    with col2:
        st.subheader("Trial Metrics")
        
        # Calculate trial metrics
        total_patients = trials_df['Patients_Enrolled'].sum()
        avg_success_rate = trials_df['Success_Rate_%'].mean()
        
        st.metric("Total Patients", f"{total_patients:,}")
        st.metric("Avg Success Rate", f"{avg_success_rate:.1f}%")
        
        # Phase distribution
        phase_dist = trials_df['Phase'].value_counts()
        st.write("**Phase Distribution:**")
        for phase, count in phase_dist.items():
            st.progress(count/len(trials_df), text=f"{phase}: {count}")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>💊 <b>Pharma Analytics Dashboard</b> | Data refreshed: {}</p>
    <p style='font-size: 0.8em;'>For internal use only. All drug names are fictional.</p>
</div>
""".format(datetime.now().strftime("%Y-%m-%d %H:%M")), unsafe_allow_html=True)


# streamlit run dashboard.py

