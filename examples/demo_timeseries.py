"""
Demo: Time Series Data Preprocessing with AutoPrepML
Example: Sales forecasting with missing timestamps and outlier handling
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from autoprepml.timeseries import TimeSeriesPrepML

# Generate sample data: Daily sales with gaps and outliers
dates = []
sales = []

# Generate dates with intentional gaps
current_date = datetime(2024, 1, 1)
for i in range(60):
    # Skip some dates to create gaps (weekends or missing data)
    if i in [5, 6, 12, 13, 19, 20, 35, 36]:
        current_date += timedelta(days=1)
        continue

    dates.append(current_date)

    # Generate sales with trend and seasonality
    trend = 100 + i * 2  # Upward trend
    seasonality = 20 * np.sin(i / 7 * 2 * np.pi)  # Weekly pattern
    noise = np.random.normal(0, 10)
    value = trend + seasonality + noise

    # Add some outliers
    if i in [15, 30, 45]:
        value *= 3  # Spike

    sales.append(max(0, value))
    current_date += timedelta(days=1)

# Add some duplicate timestamps
dates.append(dates[5])
sales.append(sales[5] + 10)

# Create DataFrame
df = pd.DataFrame({"date": dates, "sales": sales})

print("=" * 80)
print("📈 TIME SERIES PREPROCESSING DEMO - Daily Sales Data")
print("=" * 80)

# Initialize TimeSeriesPrepML
print("\n1️⃣  Initializing TimeSeriesPrepML...")
prep = TimeSeriesPrepML(df, timestamp_column="date", value_column="sales")
print(f"✓ Loaded {len(prep.df)} data points")
print(f"✓ Date range: {prep.df['date'].min().date()} to {prep.df['date'].max().date()}")

# Detect issues
print("\n2️⃣  Detecting time series data quality issues...")
issues = prep.detect_issues()
print(f"✓ Duplicate timestamps: {issues['duplicate_timestamps']}")
print(f"✓ Detected gaps: {issues['detected_gaps']}")
print(f"✓ Is chronological: {issues['is_chronological']}")
print(f"✓ Missing values: {issues['missing_values']}")
print(f"✓ Negative values: {issues['negative_values']}")
print(f"✓ Total records: {issues['total_records']}")

# Sort by time
print("\n3️⃣  Sorting by timestamp...")
prep.sort_by_time()
print("✓ Data sorted chronologically")

# Remove duplicate timestamps
print("\n4️⃣  Handling duplicate timestamps...")
original_count = len(prep.df)
prep.remove_duplicate_timestamps(aggregate="mean")
removed = original_count - len(prep.df)
print(f"✓ Aggregated {removed} duplicate timestamps using mean")

# Fill missing timestamps
print("\n5️⃣  Filling missing timestamps...")
original_count = len(prep.df)
prep.fill_missing_timestamps(freq="D")  # Daily frequency
added = len(prep.df) - original_count
print(f"✓ Added {added} missing dates")
print(f"✓ Complete date range: {len(prep.df)} days")

# Interpolate missing values
print("\n6️⃣  Interpolating missing sales values...")
missing_before = prep.df["sales"].isnull().sum()
prep.interpolate_missing(method="linear")
missing_after = prep.df["sales"].isnull().sum()
print(f"✓ Filled {missing_before - missing_after} missing values using linear interpolation")

# Detect outliers
print("\n7️⃣  Detecting outliers...")
prep.detect_outliers(method="zscore", threshold=3.0)
outlier_count = prep.df["sales_is_outlier"].sum()
print(f"✓ Detected {outlier_count} outliers using z-score method")

if outlier_count > 0:
    outlier_values = prep.df[prep.df["sales_is_outlier"]]["sales"].values
    print(f"   Outlier values: {outlier_values[:5]}")

# Add time features
print("\n8️⃣  Extracting time-based features...")
prep.add_time_features()
print("✓ Added features:")
time_features = [
    "date_year",
    "date_month",
    "date_day",
    "date_dayofweek",
    "date_hour",
    "date_quarter",
    "date_is_weekend",
]
for feature in time_features:
    print(f"   - {feature}")

# Add lag features
print("\n9️⃣  Creating lag features...")
prep.add_lag_features(lags=[1, 7, 30])
print("✓ Added lag features: 1-day, 7-day (weekly), 30-day (monthly)")

# Add rolling features
print("\n🔟 Creating rolling window statistics...")
prep.add_rolling_features(windows=[7, 30], functions=["mean", "std"])
print("✓ Added rolling features:")
print("   - 7-day rolling mean & std")
print("   - 30-day rolling mean & std")

# Display sample with features
print("\n📊 Sample data with engineered features:")
print("-" * 80)
sample_df = prep.df[
    ["date", "sales", "date_dayofweek", "date_is_weekend", "sales_lag_1", "sales_rolling_mean_7"]
].tail(5)
print(sample_df.to_string(index=False))

# Generate report
print("\n📈 Generating preprocessing report...")
report = prep.report()
print(f"✓ Original shape: {report['original_shape']}")
print(f"✓ Current shape:  {report['current_shape']}")
print(f"✓ Operations performed: {len(report['logs'])}")

# Save cleaned data
output_file = "sales_cleaned.csv"
prep.df.to_csv(output_file, index=False)
print(f"\n💾 Saved cleaned time series to: {output_file}")

# Summary statistics
print("\n" + "=" * 80)
print("✨ TIME SERIES PREPROCESSING COMPLETE!")
print("=" * 80)
print("Statistics:")
print(f"   • Total data points: {len(prep.df)}")
print(f"   • Date range: {prep.df['date'].max() - prep.df['date'].min()}")
print(f"   • Average sales: ${prep.df['sales'].mean():.2f}")
print(f"   • Min sales: ${prep.df['sales'].min():.2f}")
print(f"   • Max sales: ${prep.df['sales'].max():.2f}")
print(f"   • Outliers detected: {outlier_count}")
print(f"   • Weekend days: {prep.df['date_is_weekend'].sum()}")

print("\n💡 Use Cases:")
print("   • Sales forecasting")
print("   • Demand prediction")
print("   • Trend analysis")
print("   • Seasonality detection")
print("   • Anomaly detection")

print("\n📊 Ready for ML models:")
print("   • ARIMA, SARIMA")
print("   • Prophet")
print("   • LSTM, GRU")
print("   • XGBoost, LightGBM (with lag features)")
print("=" * 80)
