"""
Demo: Interactive Dashboard - Plotly and Streamlit Integration
"""
import pandas as pd
import numpy as np
from autoprepml import InteractiveDashboard, create_plotly_dashboard, generate_streamlit_app

# Create sample dataset
np.random.seed(42)
n_samples = 500

data = {
    'customer_id': range(1, n_samples + 1),
    'age': np.random.randint(18, 80, n_samples),
    'annual_income': np.random.normal(60000, 25000, n_samples),
    'credit_score': np.random.randint(300, 850, n_samples),
    'monthly_spending': np.random.uniform(500, 5000, n_samples),
    'years_customer': np.random.randint(0, 20, n_samples),
    'num_products': np.random.randint(1, 5, n_samples),
    'satisfaction_score': np.random.uniform(1, 10, n_samples),
    'region': np.random.choice(['North', 'South', 'East', 'West'], n_samples),
    'customer_segment': np.random.choice(['Premium', 'Standard', 'Basic'], n_samples),
    'churn': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
}

# Introduce some missing values
data['credit_score'][np.random.choice(n_samples, 30, replace=False)] = np.nan
data['satisfaction_score'][np.random.choice(n_samples, 20, replace=False)] = np.nan

df = pd.DataFrame(data)

print("=" * 80)
print("Interactive Dashboard Demo - Plotly & Streamlit")
print("=" * 80)

print(f"\n📊 Sample Dataset:")
print(f"  Shape: {df.shape}")
print(f"  Columns: {list(df.columns)}")

# Initialize Interactive Dashboard
dashboard = InteractiveDashboard(df)

# 1. Create Main Plotly Dashboard
print("\n" + "=" * 80)
print("🎨 Creating Interactive Plotly Dashboard")
print("=" * 80)
html_dashboard = dashboard.create_dashboard('dashboard.html')
print("  ✅ Main dashboard saved: dashboard.html")
print("     Open in browser to interact with the visualizations!")

# 2. Create Correlation Heatmap
print("\n🎨 Creating Correlation Heatmap")
print("=" * 80)
try:
    html_corr = dashboard.create_correlation_heatmap('correlation_heatmap.html')
    print("  ✅ Correlation heatmap saved: correlation_heatmap.html")
except Exception as e:
    print(f"  ⚠️  Could not create heatmap: {e}")

# 3. Create Missing Data Visualization
print("\n🎨 Creating Missing Data Visualization")
print("=" * 80)
html_missing = dashboard.create_missing_plot('missing_data.html')
print("  ✅ Missing data plot saved: missing_data.html")

# 4. Generate Streamlit App
print("\n" + "=" * 80)
print("🚀 Generating Streamlit App")
print("=" * 80)
dashboard.generate_streamlit_app('streamlit_dashboard.py')
print("  ✅ Streamlit app generated: streamlit_dashboard.py")
print("     Run with: streamlit run streamlit_dashboard.py")

# 5. Using convenience functions
print("\n" + "=" * 80)
print("⚡ Using Convenience Functions")
print("=" * 80)

# Create quick dashboard with convenience function
create_plotly_dashboard(df, title="Customer Analysis Dashboard", output_path="quick_dashboard.html")
print("  ✅ Quick dashboard created: quick_dashboard.html")

# Generate Streamlit app template
generate_streamlit_app("custom_streamlit_app.py")
print("  ✅ Custom Streamlit app: custom_streamlit_app.py")

# Show instructions
print("\n" + "=" * 80)
print("📖 How to Use the Generated Files")
print("=" * 80)

print("\n1. Plotly Dashboards (HTML files):")
print("   - Open any .html file in your web browser")
print("   - Interactive features:")
print("     * Zoom in/out on plots")
print("     * Hover for detailed values")
print("     * Pan across visualizations")
print("     * Download plots as images")

print("\n2. Streamlit App:")
print("   - Install streamlit if not already: pip install streamlit")
print("   - Run: streamlit run streamlit_dashboard.py")
print("   - Features:")
print("     * Upload CSV files")
print("     * Interactive EDA")
print("     * Data preprocessing controls")
print("     * Feature engineering")
print("     * Download processed data")

print("\n" + "=" * 80)
print("💡 Plotly Features Available")
print("=" * 80)
print("  ✓ Interactive histograms and distributions")
print("  ✓ Box plots for outlier detection")
print("  ✓ Scatter plots for relationships")
print("  ✓ Correlation heatmaps")
print("  ✓ Bar charts for categorical data")
print("  ✓ Missing data visualizations")

print("\n" + "=" * 80)
print("💡 Streamlit App Features")
print("=" * 80)
print("  ✓ File upload interface")
print("  ✓ Dataset overview with metrics")
print("  ✓ Automated EDA integration")
print("  ✓ Interactive preprocessing controls")
print("  ✓ Feature engineering options")
print("  ✓ Download processed data")

print("\n" + "=" * 80)
print("✨ Demo Complete!")
print("=" * 80)

print("\nGenerated Files:")
print("  📄 dashboard.html - Main interactive dashboard")
print("  📄 correlation_heatmap.html - Correlation analysis")
print("  📄 missing_data.html - Missing value visualization")
print("  📄 quick_dashboard.html - Quick dashboard example")
print("  🐍 streamlit_dashboard.py - Full Streamlit app")
print("  🐍 custom_streamlit_app.py - Customizable Streamlit template")

print("\nNext steps:")
print("  1. Open the HTML files in your browser")
print("  2. Run the Streamlit app:")
print("     $ streamlit run streamlit_dashboard.py")
print("  3. Upload your own CSV files to explore")
print("  4. Customize the Streamlit app for your needs")

print("\n💡 Tip: For production deployments:")
print("  - Deploy Streamlit apps to Streamlit Cloud (free)")
print("  - Host Plotly dashboards on GitHub Pages")
print("  - Share interactive visualizations with stakeholders")
