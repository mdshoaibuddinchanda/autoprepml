"""Tests for text preprocessing module"""
import pytest
import pandas as pd
from autoprepml.text import TextPrepML


@pytest.fixture
def sample_text_df():
    """Create sample text DataFrame"""
    return pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'text': [
            'This is a GREAT product! http://example.com',
            'Contact us at support@example.com for help',
            '<html><body>Nice review</body></html>',
            'Too short',
            'The quick brown fox jumps over the lazy dog. ' * 100  # Very long text
        ]
    })


def test_textprepml_init(sample_text_df):
    """Test TextPrepML initialization"""
    prep = TextPrepML(sample_text_df, text_column='text')
    
    assert prep.text_column == 'text'
    assert len(prep.df) == 5
    assert prep.log == []


def test_textprepml_init_invalid_column(sample_text_df):
    """Test initialization with invalid column"""
    with pytest.raises(ValueError, match="not found"):
        TextPrepML(sample_text_df, text_column='nonexistent')


def test_detect_issues(sample_text_df):
    """Test text issue detection"""
    prep = TextPrepML(sample_text_df, text_column='text')
    issues = prep.detect_issues()
    
    assert 'missing_text' in issues
    assert 'contains_urls' in issues
    assert 'contains_emails' in issues
    assert 'contains_html' in issues
    assert issues['contains_urls'] >= 1
    assert issues['contains_emails'] >= 1
    assert issues['contains_html'] >= 1


def test_clean_text(sample_text_df):
    """Test text cleaning"""
    prep = TextPrepML(sample_text_df, text_column='text')
    result = prep.clean_text(
        lowercase=True,
        remove_urls=True,
        remove_emails=True,
        remove_html=True
    )
    
    # Check URLs removed
    assert 'http://' not in result['text'].iloc[0]
    
    # Check emails removed
    assert '@' not in result['text'].iloc[1]
    
    # Check HTML removed
    assert '<html>' not in result['text'].iloc[2]
    
    # Check lowercase
    assert 'GREAT' not in result['text'].iloc[0]
    assert 'great' in result['text'].iloc[0].lower()


def test_clean_text_special_chars():
    """Test cleaning special characters"""
    df = pd.DataFrame({'text': ['Hello!!! @#$ World???']})
    prep = TextPrepML(df, text_column='text')
    result = prep.clean_text(remove_special_chars=True)
    
    assert '@' not in result['text'].iloc[0]
    assert '#' not in result['text'].iloc[0]
    assert '$' not in result['text'].iloc[0]


def test_clean_text_numbers():
    """Test removing numbers"""
    df = pd.DataFrame({'text': ['I have 123 apples and 456 oranges']})
    prep = TextPrepML(df, text_column='text')
    result = prep.clean_text(remove_numbers=True)
    
    assert '123' not in result['text'].iloc[0]
    assert '456' not in result['text'].iloc[0]


def test_remove_stopwords():
    """Test stopword removal"""
    df = pd.DataFrame({'text': ['this is a test of the system']})
    prep = TextPrepML(df, text_column='text')
    result = prep.remove_stopwords()
    
    # Common stopwords should be removed
    text = result['text'].iloc[0]
    assert 'this' not in text
    assert 'is' not in text
    assert 'a' not in text
    assert 'the' not in text
    # Content words should remain
    assert 'test' in text
    assert 'system' in text


def test_remove_stopwords_custom_list():
    """Test stopword removal with custom list"""
    df = pd.DataFrame({'text': ['hello world foo bar']})
    prep = TextPrepML(df, text_column='text')
    result = prep.remove_stopwords(stopwords=['hello', 'foo'])
    
    text = result['text'].iloc[0]
    assert 'hello' not in text
    assert 'foo' not in text
    assert 'world' in text
    assert 'bar' in text


def test_tokenize_word():
    """Test word tokenization"""
    df = pd.DataFrame({'text': ['Hello world this is a test']})
    prep = TextPrepML(df, text_column='text')
    result = prep.tokenize(method='word')
    
    assert 'text_tokens' in result.columns
    tokens = result['text_tokens'].iloc[0]
    assert isinstance(tokens, list)
    assert len(tokens) == 6
    assert 'Hello' in tokens


def test_tokenize_sentence():
    """Test sentence tokenization"""
    df = pd.DataFrame({'text': ['First sentence. Second sentence! Third?']})
    prep = TextPrepML(df, text_column='text')
    result = prep.tokenize(method='sentence')
    
    assert 'text_tokens' in result.columns
    tokens = result['text_tokens'].iloc[0]
    assert isinstance(tokens, list)
    assert len(tokens) > 1


def test_extract_features():
    """Test feature extraction"""
    df = pd.DataFrame({'text': ['Hello World! 123']})
    prep = TextPrepML(df, text_column='text')
    result = prep.extract_features()
    
    # Check feature columns created
    assert 'text_length' in result.columns
    assert 'text_word_count' in result.columns
    assert 'text_upper_count' in result.columns
    assert 'text_digit_count' in result.columns
    assert 'text_special_char_count' in result.columns
    assert 'text_avg_word_len' in result.columns
    
    # Verify feature values
    assert result['text_length'].iloc[0] > 0
    assert result['text_word_count'].iloc[0] == 3
    assert result['text_upper_count'].iloc[0] >= 2  # H and W
    assert result['text_digit_count'].iloc[0] == 3


def test_detect_language():
    """Test language detection"""
    df = pd.DataFrame({'text': ['The quick brown fox is and are', 'Another English sentence the of']})
    prep = TextPrepML(df, text_column='text')
    result = prep.detect_language()
    
    assert 'text_language' in result.columns
    # Should have language column (simple heuristic may vary)
    assert 'text_language' in result.columns


def test_remove_duplicates(sample_text_df):
    """Test duplicate removal"""
    # Add a duplicate
    df_with_dup = pd.concat([sample_text_df, sample_text_df.iloc[[0]]], ignore_index=True)
    
    prep = TextPrepML(df_with_dup, text_column='text')
    original_len = len(prep.df)
    result = prep.remove_duplicates()
    
    assert len(result) < original_len


def test_filter_by_length():
    """Test length filtering"""
    df = pd.DataFrame({
        'text': [
            'Short',
            'This is a medium length text',
            'X' * 6000  # Very long
        ]
    })
    
    prep = TextPrepML(df, text_column='text')
    result = prep.filter_by_length(min_length=10, max_length=100)
    
    # Only medium text should remain
    assert len(result) == 1
    assert 'medium' in result['text'].iloc[0]


def test_get_vocabulary():
    """Test vocabulary extraction"""
    df = pd.DataFrame({'text': ['hello world hello test world hello']})
    prep = TextPrepML(df, text_column='text')
    vocab = prep.get_vocabulary(top_n=3)
    
    assert isinstance(vocab, dict)
    assert 'hello' in vocab
    assert vocab['hello'] == 3  # Most common
    assert vocab['world'] == 2


def test_report(sample_text_df):
    """Test report generation"""
    prep = TextPrepML(sample_text_df, text_column='text')
    prep.clean_text()
    prep.extract_features()
    
    report = prep.report()
    
    assert 'original_shape' in report
    assert 'current_shape' in report
    assert 'text_column' in report
    assert 'logs' in report
    assert 'issues' in report
    assert len(report['logs']) > 0


def test_empty_strings():
    """Test handling empty strings"""
    df = pd.DataFrame({'text': ['', 'valid text', '   ', None]})
    prep = TextPrepML(df, text_column='text')
    issues = prep.detect_issues()
    
    assert issues['empty_strings'] >= 1
    assert issues['missing_text'] >= 1


def test_chained_operations():
    """Test chaining multiple operations"""
    df = pd.DataFrame({
        'text': [
            'This is <b>HTML</b> with http://url.com and EMAIL@test.com',
            'Another test sentence with CAPS'
        ]
    })
    
    prep = TextPrepML(df, text_column='text')
    prep.clean_text(lowercase=True, remove_html=True, remove_urls=True, remove_emails=True)
    prep.remove_stopwords()
    result = prep.extract_features()
    
    # Check all operations applied
    assert '<b>' not in prep.df['text'].iloc[0]
    assert 'http://' not in prep.df['text'].iloc[0]
    assert '@' not in prep.df['text'].iloc[0]
    assert 'CAPS' not in prep.df['text'].iloc[1]
    assert 'text_length' in result.columns
