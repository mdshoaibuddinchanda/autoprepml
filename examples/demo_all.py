"""
Demo: Multi-Modal Data Preprocessing with AutoPrepML
Shows all 4 data types: Tabular, Text, Time Series, and Graph
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from autoprepml import AutoPrepML  # Tabular
from autoprepml import TextPrepML  # Text/NLP
from autoprepml import TimeSeriesPrepML  # Time Series
from autoprepml import GraphPrepML  # Graph

print("=" * 80)
print("🚀 AutoPrepML - MULTI-MODAL DATA PREPROCESSING DEMO")
print("=" * 80)
print("\nSupported Data Types:")
print("  1. Tabular (CSV/Excel/JSON)")
print("  2. Text/NLP (documents, reviews, tweets)")
print("  3. Time Series (temporal data)")
print("  4. Graph (networks, relationships)")
print("=" * 80)

# ============================================================================
# 1. TABULAR DATA
# ============================================================================
print("\n" + "=" * 80)
print("📊 1. TABULAR DATA PREPROCESSING")
print("=" * 80)

# Sample tabular data (classification task)
tabular_data = {
    'age': [25, 30, np.nan, 45, 50, 35, 28, 1000],  # Outlier: 1000
    'income': [50000, 60000, 55000, np.nan, 80000, 65000, 52000, 75000],
    'category': ['A', 'B', 'A', 'C', 'B', 'A', 'A', 'C'],
    'target': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no']
}
df_tabular = pd.DataFrame(tabular_data)

print("\nInitializing AutoPrepML for tabular data...")
prep_tabular = AutoPrepML(df_tabular)

print("Detecting issues...")
issues = prep_tabular.detect(target_col='target')
missing_count = sum(v['count'] for v in issues['missing_values'].values())
print(f"  ✓ Missing values: {missing_count}")
print(f"  ✓ Outliers: {issues['outliers']['outlier_count']}")
print(f"  ✓ Class imbalance ratio: {issues['class_imbalance']['imbalance_ratio']:.2f}")

print("Cleaning data automatically...")
cleaned, target = prep_tabular.clean(task='classification', target_col='target', auto=True)
print(f"  ✓ Shape: {df_tabular.shape} → {cleaned.shape}")

# ============================================================================
# 2. TEXT/NLP DATA
# ============================================================================
print("\n" + "=" * 80)
print("📝 2. TEXT/NLP DATA PREPROCESSING")
print("=" * 80)

# Sample text data
text_data = {
    'id': [1, 2, 3, 4, 5],
    'review': [
        'This product is AMAZING! https://example.com',
        'Terrible quality <html>Bad</html>',
        'Great service! Very satisfied.',
        'ok',  # Too short
        'Perfect! Highly recommend to everyone.'
    ]
}
df_text = pd.DataFrame(text_data)

print("\nInitializing TextPrepML...")
prep_text = TextPrepML(df_text, text_column='review')

print("Detecting text issues...")
text_issues = prep_text.detect_issues()
print(f"  ✓ Contains URLs: {text_issues['contains_urls']}")
print(f"  ✓ Contains HTML: {text_issues['contains_html']}")
print(f"  ✓ Very short texts: {text_issues['very_short']}")

print("Cleaning text...")
prep_text.clean_text(lowercase=True, remove_urls=True, remove_html=True)
prep_text.remove_stopwords()
prep_text.filter_by_length(min_length=10)
print(f"  ✓ Shape: {df_text.shape} → {prep_text.df.shape}")

# ============================================================================
# 3. TIME SERIES DATA
# ============================================================================
print("\n" + "=" * 80)
print("⏰ 3. TIME SERIES DATA PREPROCESSING")
print("=" * 80)

# Sample time series data with gaps
dates = pd.date_range('2024-01-01', periods=20, freq='D')
dates_with_gaps = [dates[i] for i in range(len(dates)) if i not in [5, 6, 12]]

df_timeseries = pd.DataFrame({
    'date': dates_with_gaps,
    'value': [100 + i*5 + np.random.normal(0, 5) for i in range(len(dates_with_gaps))]
})

print("\nInitializing TimeSeriesPrepML...")
prep_ts = TimeSeriesPrepML(df_timeseries, timestamp_column='date', value_column='value')

print("Detecting time series issues...")
ts_issues = prep_ts.detect_issues()
print(f"  ✓ Total records: {ts_issues['total_records']}")
print(f"  ✓ Detected gaps: {ts_issues['detected_gaps']}")
print(f"  ✓ Is chronological: {ts_issues['is_chronological']}")

print("Preprocessing time series...")
prep_ts.fill_missing_timestamps(freq='D')
prep_ts.interpolate_missing(method='linear')
prep_ts.add_time_features()
prep_ts.add_lag_features(lags=[1, 7])
print(f"  ✓ Shape: {df_timeseries.shape} → {prep_ts.df.shape}")
print("  ✓ Features added: time features + lag features")

# ============================================================================
# 4. GRAPH DATA
# ============================================================================
print("\n" + "=" * 80)
print("🕸️  4. GRAPH DATA PREPROCESSING")
print("=" * 80)

# Sample graph data
nodes = pd.DataFrame({
    'id': [1, 2, 3, 4, 5, 5],  # Duplicate ID
    'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Eve2']
})

edges = pd.DataFrame({
    'source': [1, 1, 2, 3, 6],  # Node 6 doesn't exist (dangling)
    'target': [2, 3, 3, 4, 7]   # Node 7 doesn't exist (dangling)
})

print("\nInitializing GraphPrepML...")
prep_graph = GraphPrepML(
    nodes_df=nodes, 
    edges_df=edges,
    node_id_col='id',
    source_col='source',
    target_col='target'
)

print("Detecting graph issues...")
graph_issues = prep_graph.detect_issues()
print(f"  ✓ Duplicate nodes: {graph_issues['nodes']['duplicate_node_ids']}")
print(f"  ✓ Dangling edges: {graph_issues['edges']['dangling_edges']}")
print(f"  ✓ Self-loops: {graph_issues['edges']['self_loops']}")

print("Cleaning graph data...")
prep_graph.validate_node_ids()
prep_graph.validate_edges(remove_self_loops=True, remove_dangling=True)
prep_graph.add_node_features()
prep_graph.identify_components()
print(f"  ✓ Nodes: {nodes.shape} → {prep_graph.nodes_df.shape}")
print(f"  ✓ Edges: {edges.shape} → {prep_graph.edges_df.shape}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("✨ MULTI-MODAL PREPROCESSING COMPLETE!")
print("=" * 80)

print("\n📊 Results Summary:")
print(f"""
1. Tabular Data:
   • Original: {df_tabular.shape[0]} rows, {df_tabular.shape[1]} columns
   • Cleaned: {cleaned.shape[0]} rows, {cleaned.shape[1]} columns
   • Issues fixed: missing values, outliers, encoding

2. Text/NLP Data:
   • Original: {df_text.shape[0]} reviews
   • Cleaned: {prep_text.df.shape[0]} reviews
   • Operations: URL/HTML removal, stopwords, filtering

3. Time Series Data:
   • Original: {df_timeseries.shape[0]} data points
   • Enhanced: {prep_ts.df.shape[0]} data points, {prep_ts.df.shape[1]} features
   • Operations: gap filling, interpolation, feature engineering

4. Graph Data:
   • Nodes: {nodes.shape[0]} → {prep_graph.nodes_df.shape[0]}
   • Edges: {edges.shape[0]} → {prep_graph.edges_df.shape[0]}
   • Operations: validation, deduplication, feature extraction
""")

print("=" * 80)
print("💡 AutoPrepML is now ready to handle ANY data type!")
print("=" * 80)

print("\n📚 Next Steps:")
print("  • Use cleaned tabular data for ML models (sklearn, xgboost)")
print("  • Use preprocessed text for NLP models (transformers, BERT)")
print("  • Use time series features for forecasting (ARIMA, Prophet, LSTM)")
print("  • Use graph features for network analysis (PageRank, GNN)")

print("\n🔗 More Information:")
print("  • Documentation: docs/")
print("  • Examples: examples/demo_*.py")
print("  • Tests: tests/test_*.py")
print("  • Summary: MULTI_MODAL_SUMMARY.md")

print("\n" + "=" * 80)
