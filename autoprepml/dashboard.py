"""Interactive Dashboard module for AutoPrepML using Plotly and Streamlit"""
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from pathlib import Path
import json


def create_plotly_dashboard(df: pd.DataFrame, 
                            title: str = "AutoPrepML Interactive Dashboard",
                            output_path: Optional[str] = None) -> str:
    """Create an interactive Plotly dashboard.
    
    Args:
        df: DataFrame to visualize
        title: Dashboard title
        output_path: Optional path to save HTML file
        
    Returns:
        HTML string of the dashboard
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import plotly.express as px
    except ImportError as e:
        raise ImportError(
            "Plotly is required for interactive dashboards.\n"
            "Install with: pip install plotly"
        ) from e

    # Create figure with subplots
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    n_plots = min(4, len(numeric_cols))  # Show up to 4 distribution plots

    if n_plots == 0:
        # No numeric columns, create basic info dashboard
        fig = go.Figure()
        fig.add_annotation(
            text=f"No numeric columns to visualize<br>Rows: {len(df)}, Columns: {len(df.columns)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20)
        )
    else:
        # Create subplots for distributions
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[f"{col} Distribution" for col in numeric_cols[:n_plots]],
            specs=[[{"type": "histogram"}, {"type": "box"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )

        # Add histogram
        if len(numeric_cols) > 0:
            fig.add_trace(
                go.Histogram(x=df[numeric_cols[0]], name=numeric_cols[0]),
                row=1, col=1
            )

        # Add box plot
        if len(numeric_cols) > 1:
            fig.add_trace(
                go.Box(y=df[numeric_cols[1]], name=numeric_cols[1]),
                row=1, col=2
            )

        # Add scatter plot
        if len(numeric_cols) >= 2:
            fig.add_trace(
                go.Scatter(
                    x=df[numeric_cols[0]], 
                    y=df[numeric_cols[1]],
                    mode='markers',
                    name=f"{numeric_cols[0]} vs {numeric_cols[1]}"
                ),
                row=2, col=1
            )

        # Add value counts for first categorical
        if cat_cols:
            value_counts = df[cat_cols[0]].value_counts().head(10)
            fig.add_trace(
                go.Bar(x=value_counts.index, y=value_counts.values, name=cat_cols[0]),
                row=2, col=2
            )

    fig.update_layout(
        title=title,
        height=800,
        showlegend=True,
        template="plotly_white"
    )

    html = fig.to_html(full_html=True, include_plotlyjs='cdn')

    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(html, encoding='utf-8')
        print(f"✅ Plotly dashboard saved to: {output_path}")

    return html


def create_correlation_heatmap(df: pd.DataFrame, 
                               output_path: Optional[str] = None) -> str:
    """Create an interactive correlation heatmap.
    
    Args:
        df: DataFrame with numeric columns
        output_path: Optional path to save HTML file
        
    Returns:
        HTML string of the heatmap
    """
    try:
        import plotly.graph_objects as go
    except ImportError as e:
        raise ImportError(
            "Plotly is required. Install with: pip install plotly"
        ) from e

    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.shape[1] < 2:
        raise ValueError("Need at least 2 numeric columns for correlation heatmap")

    corr_matrix = numeric_df.corr()

    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="Correlation")
    ))

    fig.update_layout(
        title="Correlation Heatmap",
        height=600,
        template="plotly_white"
    )

    html = fig.to_html(full_html=True, include_plotlyjs='cdn')

    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(html, encoding='utf-8')
        print(f"✅ Correlation heatmap saved to: {output_path}")

    return html


def create_missing_data_plot(df: pd.DataFrame,
                             output_path: Optional[str] = None) -> str:
    """Create interactive visualization of missing data.
    
    Args:
        df: DataFrame to analyze
        output_path: Optional path to save HTML file
        
    Returns:
        HTML string of the plot
    """
    try:
        import plotly.graph_objects as go
    except ImportError as e:
        raise ImportError(
            "Plotly is required. Install with: pip install plotly"
        ) from e

    missing_counts = df.isnull().sum()
    missing_pct = (missing_counts / len(df)) * 100

    # Filter columns with missing values
    missing_cols = missing_counts[missing_counts > 0].sort_values(ascending=True)

    if missing_cols.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No missing values detected!",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20, color="green")
        )
    else:
        fig = go.Figure(data=[
            go.Bar(
                y=missing_cols.index,
                x=missing_cols.values,
                orientation='h',
                text=[f"{missing_pct[col]:.1f}%" for col in missing_cols.index],
                textposition='auto',
            )
        ])

        fig.update_layout(
            title="Missing Data by Column",
            xaxis_title="Number of Missing Values",
            yaxis_title="Column",
            height=max(400, len(missing_cols) * 30),
            template="plotly_white"
        )

    html = fig.to_html(full_html=True, include_plotlyjs='cdn')

    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(html, encoding='utf-8')
        print(f"✅ Missing data plot saved to: {output_path}")

    return html


def generate_streamlit_app(output_path: str = "streamlit_app.py") -> None:
    """Generate a Streamlit app file for interactive data exploration.
    
    Args:
        output_path: Path to save the Streamlit app file
    """
    app_code = '''"""
AutoPrepML Interactive Dashboard - Streamlit App
Run with: streamlit run streamlit_app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from autoprepml import AutoPrepML, AutoEDA
from autoprepml.feature_engine import AutoFeatureEngine

st.set_page_config(page_title="AutoPrepML Dashboard", layout="wide")

st.title("🚀 AutoPrepML Interactive Dashboard")
st.markdown("Upload your data and explore preprocessing options interactively")

# Sidebar
st.sidebar.header("Configuration")

# File upload
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    st.sidebar.success(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔍 EDA", "⚙️ Preprocessing", "🎯 Feature Engineering"])
    
    with tab1:
        st.header("Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Rows", len(df))
        col2.metric("Columns", len(df.columns))
        col3.metric("Missing Values", df.isnull().sum().sum())
        col4.metric("Duplicates", df.duplicated().sum())
        
        st.subheader("Data Preview")
        st.dataframe(df.head(20), use_container_width=True)
        
        st.subheader("Column Information")
        col_info = pd.DataFrame({
            'Column': df.columns,
            'Type': df.dtypes.values,
            'Non-Null': df.count().values,
            'Null Count': df.isnull().sum().values,
            'Unique': [df[col].nunique() for col in df.columns]
        })
        st.dataframe(col_info, use_container_width=True)
    
    with tab2:
        st.header("Exploratory Data Analysis")
        
        # Run AutoEDA
        if st.button("Run Automated EDA"):
            with st.spinner("Analyzing data..."):
                eda = AutoEDA(df)
                results = eda.analyze()
                
                st.subheader("📋 Automated Insights")
                for insight in eda.get_insights():
                    st.info(insight)
                
                # Numeric distributions
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_cols:
                    st.subheader("Distribution Plots")
                    selected_col = st.selectbox("Select column", numeric_cols)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        fig = px.histogram(df, x=selected_col, title=f"{selected_col} Distribution")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        fig = px.box(df, y=selected_col, title=f"{selected_col} Box Plot")
                        st.plotly_chart(fig, use_container_width=True)
                
                # Correlations
                if len(numeric_cols) >= 2:
                    st.subheader("Correlation Heatmap")
                    corr_matrix = df[numeric_cols].corr()
                    fig = px.imshow(corr_matrix, 
                                   text_auto=True,
                                   aspect="auto",
                                   color_continuous_scale='RdBu',
                                   color_continuous_midpoint=0)
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("Data Preprocessing")
        
        st.subheader("Cleaning Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            handle_missing = st.selectbox(
                "Handle Missing Values",
                ["None", "Drop Rows", "Mean Imputation", "Median Imputation"]
            )
            
            handle_outliers = st.checkbox("Remove Outliers (IQR method)")
            
            scale_features = st.checkbox("Scale Numeric Features")
            if scale_features:
                scaler_type = st.radio("Scaler", ["Standard", "MinMax"])
        
        with col2:
            remove_duplicates = st.checkbox("Remove Duplicate Rows")
            
            encode_categorical = st.checkbox("Encode Categorical Variables")
            if encode_categorical:
                encoding_method = st.radio("Method", ["Label Encoding", "One-Hot Encoding"])
        
        if st.button("Apply Preprocessing"):
            with st.spinner("Processing..."):
                prep = AutoPrepML(df)
                
                # Apply selected operations
                if handle_missing != "None":
                    if handle_missing == "Drop Rows":
                        df_clean = prep.df.dropna()
                    else:
                        method = 'mean' if handle_missing == "Mean Imputation" else 'median'
                        from autoprepml.cleaning import impute_missing
                        df_clean = impute_missing(prep.df, method=method)
                else:
                    df_clean = prep.df
                
                st.success("✅ Preprocessing complete!")
                st.dataframe(df_clean.head(), use_container_width=True)
                
                # Download button
                csv = df_clean.to_csv(index=False)
                st.download_button(
                    "Download Preprocessed Data",
                    csv,
                    "preprocessed_data.csv",
                    "text/csv"
                )
    
    with tab4:
        st.header("Feature Engineering")
        
        target_col = st.selectbox("Select Target Column (optional)", ["None"] + list(df.columns))
        target_col = None if target_col == "None" else target_col
        
        st.subheader("Feature Creation Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            create_polynomial = st.checkbox("Create Polynomial Features")
            create_interactions = st.checkbox("Create Interaction Features")
            create_ratios = st.checkbox("Create Ratio Features")
        
        with col2:
            create_binned = st.checkbox("Create Binned Features")
            create_aggregations = st.checkbox("Create Aggregation Features")
            create_datetime = st.checkbox("Extract DateTime Features")
        
        if st.button("Generate Features"):
            with st.spinner("Engineering features..."):
                fe = AutoFeatureEngine(df, target_column=target_col)
                
                if create_polynomial:
                    fe.create_polynomial_features(degree=2, interaction_only=True)
                
                if create_interactions:
                    fe.create_interactions(max_interactions=10)
                
                if create_ratios:
                    fe.create_ratio_features(max_ratios=10)
                
                if create_binned:
                    fe.create_binned_features(n_bins=5)
                
                if create_aggregations:
                    fe.create_aggregation_features()
                
                if create_datetime:
                    fe.create_datetime_features()
                
                df_engineered = fe.get_features()
                summary = fe.get_summary()
                
                st.success(f"✅ Created {summary['features_created']} new features!")
                
                st.subheader("Feature Engineering Summary")
                st.json(summary)
                
                st.subheader("Engineered Dataset Preview")
                st.dataframe(df_engineered.head(), use_container_width=True)
                
                # Download
                csv = df_engineered.to_csv(index=False)
                st.download_button(
                    "Download Engineered Data",
                    csv,
                    "engineered_data.csv",
                    "text/csv"
                )

else:
    st.info("👈 Upload a CSV file to get started!")
    
    st.markdown("""
    ## Features
    
    - **Overview**: Quick dataset statistics and preview
    - **EDA**: Automated exploratory data analysis with visualizations
    - **Preprocessing**: Interactive data cleaning and transformation
    - **Feature Engineering**: Automated feature creation and selection
    
    ## Quick Start
    
    1. Upload your CSV file using the sidebar
    2. Explore the data in different tabs
    3. Apply preprocessing and feature engineering
    4. Download the processed data
    """)
'''
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(app_code, encoding='utf-8')
    print(f"✅ Streamlit app generated: {output_path}")
    print(f"   Run with: streamlit run {output_path}")


class InteractiveDashboard:
    """Interactive dashboard class for AutoPrepML.
    
    Provides methods to create various interactive visualizations
    and generate Streamlit apps.
    """
    
    def __init__(self, df: pd.DataFrame):
        """Initialize InteractiveDashboard.
        
        Args:
            df: DataFrame to visualize
            
        Raises:
            ValueError: If input is not a DataFrame or is empty
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        
        if df.empty:
            raise ValueError("DataFrame cannot be empty")
        
        self.df = df
    
    def create_dashboard(
        self,
        title: str = "AutoPrepML Interactive Dashboard",
        output_path: str = "dashboard.html",
    ) -> str:
        """Create comprehensive interactive dashboard.
        
        Args:
            title: Title to display in the dashboard
            output_path: Path to save HTML file
            
        Returns:
            HTML string
        """
        return create_plotly_dashboard(
            self.df,
            title=title,
            output_path=output_path,
        )
    
    def create_correlation_heatmap(self, output_path: str = "correlation.html") -> str:
        """Create correlation heatmap.
        
        Args:
            output_path: Path to save HTML file
            
        Returns:
            HTML string
        """
        return create_correlation_heatmap(self.df, output_path=output_path)
    
    def create_missing_data_plot(self, output_path: str = "missing_data.html") -> str:
        """Create missing data visualization.
        
        Args:
            output_path: Path to save HTML file
            
        Returns:
            HTML string
        """
        return create_missing_data_plot(self.df, output_path=output_path)

    # Backwards compatibility alias (deprecated)
    def create_missing_plot(self, output_path: str = "missing_data.html") -> str:
        """Alias for create_missing_data_plot for backward compatibility."""
        return self.create_missing_data_plot(output_path=output_path)
    
    def generate_streamlit_app(self, output_path: str = "streamlit_app.py") -> None:
        """Generate Streamlit app file.
        
        Args:
            output_path: Path to save Python file
        """
        generate_streamlit_app(output_path)
