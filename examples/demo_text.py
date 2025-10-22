"""
Demo: Text/NLP Data Preprocessing with AutoPrepML
Example: Cleaning and analyzing customer reviews/tweets
"""

import pandas as pd
from autoprepml.text import TextPrepML

# Sample data: Customer product reviews
reviews_data = {
    'review_id': list(range(1, 16)),
    'review_text': [
        'This product is AMAZING! Best purchase ever!!! http://example.com/review',
        'Terrible quality. Complete waste of money. Do NOT buy! 😡',
        '<html><body>Great product, highly recommended!</body></html>',
        'Contact support@company.com for issues. Not satisfied.',
        'ok',  # Too short
        'The product arrived quickly and works as expected. Very happy with the purchase.',
        'This product is AMAZING! Best purchase ever!!!',  # Duplicate
        'Absolutely love it! 5 stars ⭐⭐⭐⭐⭐',
        'Meh... it\'s okay I guess. Nothing special really.',
        'WORST PRODUCT EVER!!! AVOID AT ALL COSTS!!!',
        '   ',  # Empty
        'Good value for money. Shipping was fast. Would recommend to friends and family.',
        'Broke after 2 weeks. Poor quality control. Very disappointed!!!',
        'Perfect! Exactly what I needed. Great customer service too.',
        'Not bad, but could be better. The price is a bit high for what you get.'
    ]
}

df = pd.DataFrame(reviews_data)

print("=" * 80)
print("📝 TEXT/NLP PREPROCESSING DEMO - Customer Reviews")
print("=" * 80)

# Initialize TextPrepML
print("\n1️⃣  Initializing TextPrepML...")
prep = TextPrepML(df, text_column='review_text')
print(f"✓ Loaded {len(prep.df)} reviews")

# Detect issues
print("\n2️⃣  Detecting text data quality issues...")
issues = prep.detect_issues()
print(f"✓ Missing text: {issues['missing_text']}")
print(f"✓ Empty strings: {issues['empty_strings']}")
print(f"✓ Very short texts (< 5 chars): {issues['very_short']}")
print(f"✓ Contains URLs: {issues['contains_urls']}")
print(f"✓ Contains emails: {issues['contains_emails']}")
print(f"✓ Contains HTML: {issues['contains_html']}")
print(f"✓ Average length: {issues['avg_length']:.1f} characters")

# Clean text
print("\n3️⃣  Cleaning text data...")
prep.clean_text(
    lowercase=True,
    remove_urls=True,
    remove_emails=True,
    remove_html=True,
    remove_extra_spaces=True
)
print("✓ Cleaned text: removed URLs, emails, HTML tags, normalized case")

# Show before/after examples
print("\n   Before: ", df['review_text'].iloc[0][:60])
print("   After:  ", prep.df['review_text'].iloc[0][:60])

# Remove stopwords
print("\n4️⃣  Removing stopwords...")
prep.remove_stopwords()
print("✓ Removed common English stopwords")

# Filter by length
print("\n5️⃣  Filtering by length...")
original_count = len(prep.df)
prep.filter_by_length(min_length=10, max_length=500)
print(f"✓ Kept {len(prep.df)}/{original_count} reviews (removed too short/long)")

# Remove duplicates
print("\n6️⃣  Removing duplicate reviews...")
original_count = len(prep.df)
prep.remove_duplicates(keep='first')
print(f"✓ Kept {len(prep.df)}/{original_count} reviews (removed duplicates)")

# Extract features
print("\n7️⃣  Extracting text features...")
prep.extract_features()
print("✓ Added features:")
feature_cols = [col for col in prep.df.columns if col.startswith('review_text_')]
for col in feature_cols[:6]:
    print(f"   - {col}")

# Tokenize
print("\n8️⃣  Tokenizing text...")
prep.tokenize(method='word')
print("✓ Created word tokens")

# Sample token output
sample_tokens = prep.df['review_text_tokens'].iloc[0][:5]
print(f"   Sample tokens: {sample_tokens}")

# Get vocabulary
print("\n9️⃣  Analyzing vocabulary...")
vocab = prep.get_vocabulary(top_n=10)
print("✓ Top 10 most common words:")
for word, count in list(vocab.items())[:10]:
    print(f"   {word}: {count}")

# Detect language
print("\n🔟 Detecting language...")
prep.detect_language()
english_count = (prep.df['review_text_language'] == 'english').sum()
print(f"✓ Detected {english_count}/{len(prep.df)} reviews as English")

# Generate report
print("\n📊 Generating preprocessing report...")
report = prep.report()
print(f"✓ Original shape: {report['original_shape']}")
print(f"✓ Current shape:  {report['current_shape']}")
print(f"✓ Operations performed: {len(report['logs'])}")

# Display sample results
print("\n📄 Sample Cleaned Reviews:")
print("-" * 80)
for idx in range(min(3, len(prep.df))):
    row = prep.df.iloc[idx]
    print(f"\n#{row['review_id']}")
    print(f"Text: {row['review_text'][:70]}...")
    print(f"Length: {row['review_text_length']} chars, "
          f"Words: {row['review_text_word_count']}, "
          f"Language: {row['review_text_language']}")

# Save cleaned data
output_file = 'reviews_cleaned.csv'
prep.df.to_csv(output_file, index=False)
print(f"\n💾 Saved cleaned reviews to: {output_file}")

# Summary statistics
print("\n" + "=" * 80)
print("✨ TEXT PREPROCESSING COMPLETE!")
print("=" * 80)
print("📈 Statistics:")
print(f"   • Total reviews processed: {len(prep.df)}")
print(f"   • Average review length: {prep.df['review_text_length'].mean():.1f} chars")
print(f"   • Average words per review: {prep.df['review_text_word_count'].mean():.1f}")
print(f"   • Total vocabulary size: {len(vocab)} unique words")

print("\n💡 Use Cases:")
print("   • Sentiment analysis")
print("   • Topic modeling")
print("   • Text classification")
print("   • Customer feedback analysis")
print("=" * 80)
