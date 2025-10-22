"""Text/NLP preprocessing module for AutoPrepML"""
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from collections import Counter


class TextPrepML:
    """Text preprocessing class for NLP tasks.
    
    Example:
        >>> prep = TextPrepML(df, text_column='review')
        >>> clean_df = prep.clean()
        >>> prep.save_report('text_report.html')
    """
    
    def __init__(self, df: pd.DataFrame, text_column: str):
        """Initialize TextPrepML.
        
        Args:
            df: Input DataFrame
            text_column: Name of column containing text data
        """
        self.df = df.copy()
        self.text_column = text_column
        self.original_df = df.copy()
        self.log = []
        
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in DataFrame")
    
    def detect_issues(self) -> Dict[str, Any]:
        """Detect text data quality issues.
        
        Returns:
            Dictionary with detected issues
        """
        text_series = self.df[self.text_column].dropna()
        
        issues = {
            'missing_text': int(self.df[self.text_column].isnull().sum()),
            'empty_strings': int((text_series == '').sum()),
            'very_short': int((text_series.str.len() < 5).sum()),
            'very_long': int((text_series.str.len() > 5000).sum()),
            'contains_urls': int(text_series.str.contains(r'http[s]?://|www\.', regex=True, na=False).sum()),
            'contains_emails': int(text_series.str.contains(r'\S+@\S+', regex=True, na=False).sum()),
            'contains_html': int(text_series.str.contains(r'<[^>]+>', regex=True, na=False).sum()),
            'avg_length': float(text_series.str.len().mean()) if len(text_series) > 0 else 0,
            'median_length': float(text_series.str.len().median()) if len(text_series) > 0 else 0,
        }
        
        self.log.append({'action': 'detect_issues', 'result': issues})
        return issues
    
    def clean_text(self, 
                   lowercase: bool = True,
                   remove_urls: bool = True,
                   remove_emails: bool = True,
                   remove_html: bool = True,
                   remove_special_chars: bool = False,
                   remove_numbers: bool = False,
                   remove_extra_spaces: bool = True) -> pd.DataFrame:
        """Clean text data.
        
        Args:
            lowercase: Convert to lowercase
            remove_urls: Remove URLs
            remove_emails: Remove email addresses
            remove_html: Remove HTML tags
            remove_special_chars: Remove special characters
            remove_numbers: Remove numbers
            remove_extra_spaces: Remove extra whitespace
            
        Returns:
            DataFrame with cleaned text
        """
        text_col = self.df[self.text_column].fillna('')
        
        # Remove HTML tags
        if remove_html:
            text_col = text_col.str.replace(r'<[^>]+>', ' ', regex=True)
        
        # Remove URLs
        if remove_urls:
            text_col = text_col.str.replace(r'http[s]?://\S+|www\.\S+', ' ', regex=True)
        
        # Remove emails
        if remove_emails:
            text_col = text_col.str.replace(r'\S+@\S+', ' ', regex=True)
        
        # Remove special characters
        if remove_special_chars:
            text_col = text_col.str.replace(r'[^a-zA-Z0-9\s]', ' ', regex=True)
        
        # Remove numbers
        if remove_numbers:
            text_col = text_col.str.replace(r'\d+', ' ', regex=True)
        
        # Convert to lowercase
        if lowercase:
            text_col = text_col.str.lower()
        
        # Remove extra spaces
        if remove_extra_spaces:
            text_col = text_col.str.replace(r'\s+', ' ', regex=True).str.strip()
        
        self.df[self.text_column] = text_col
        self.log.append({'action': 'clean_text', 'details': {
            'lowercase': lowercase,
            'remove_urls': remove_urls,
            'remove_html': remove_html
        }})
        
        return self.df
    
    def remove_stopwords(self, stopwords: Optional[List[str]] = None, language: str = 'english') -> pd.DataFrame:
        """Remove stopwords from text.
        
        Args:
            stopwords: Custom list of stopwords (if None, uses default English)
            language: Language for default stopwords ('english')
            
        Returns:
            DataFrame with stopwords removed
        """
        if stopwords is None:
            # Default English stopwords
            stopwords = [
                'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your',
                'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she',
                'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
                'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that',
                'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
                'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',
                'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through',
                'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
                'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once'
            ]
        
        stopwords_set = set(stopwords)
        
        def remove_stops(text):
            if pd.isna(text) or text == '':
                return text
            words = text.split()
            return ' '.join([w for w in words if w.lower() not in stopwords_set])
        
        self.df[self.text_column] = self.df[self.text_column].apply(remove_stops)
        self.log.append({'action': 'remove_stopwords', 'count': len(stopwords)})
        
        return self.df
    
    def tokenize(self, method: str = 'word') -> pd.DataFrame:
        """Tokenize text into words or sentences.
        
        Args:
            method: 'word' or 'sentence'
            
        Returns:
            DataFrame with tokenized text (as list column)
        """
        text_col = self.df[self.text_column].fillna('')
        
        if method == 'word':
            # Simple word tokenization
            self.df[f'{self.text_column}_tokens'] = text_col.str.split()
        elif method == 'sentence':
            # Simple sentence tokenization
            self.df[f'{self.text_column}_tokens'] = text_col.str.split(r'[.!?]+')
        else:
            raise ValueError(f"Unknown tokenization method: {method}")
        
        self.log.append({'action': 'tokenize', 'method': method})
        return self.df
    
    def extract_features(self) -> pd.DataFrame:
        """Extract numerical features from text.
        
        Returns:
            DataFrame with added text features
        """
        text_col = self.df[self.text_column].fillna('')
        
        # Length features
        self.df[f'{self.text_column}_length'] = text_col.str.len()
        self.df[f'{self.text_column}_word_count'] = text_col.str.split().str.len()
        
        # Character features
        self.df[f'{self.text_column}_upper_count'] = text_col.str.count(r'[A-Z]')
        self.df[f'{self.text_column}_digit_count'] = text_col.str.count(r'\d')
        self.df[f'{self.text_column}_special_char_count'] = text_col.str.count(r'[!@#$%^&*(),.?":{}|<>]')
        
        # Average word length
        def avg_word_len(text):
            words = text.split()
            return np.mean([len(w) for w in words]) if words else 0
        
        self.df[f'{self.text_column}_avg_word_len'] = text_col.apply(avg_word_len)
        
        self.log.append({'action': 'extract_features', 'features': 6})
        return self.df
    
    def detect_language(self) -> pd.DataFrame:
        """Detect language of text (simple heuristic).
        
        Returns:
            DataFrame with language detection column
        """
        # Simple English detection (can be enhanced with langdetect library)
        text_col = self.df[self.text_column].fillna('')
        
        common_english_words = {'the', 'is', 'are', 'and', 'of', 'to', 'in', 'a', 'for'}
        
        def is_likely_english(text):
            words = set(text.lower().split())
            overlap = len(words & common_english_words)
            return 'english' if overlap >= 2 else 'other'
        
        self.df[f'{self.text_column}_language'] = text_col.apply(is_likely_english)
        self.log.append({'action': 'detect_language', 'method': 'heuristic'})
        
        return self.df
    
    def remove_duplicates(self, keep: str = 'first') -> pd.DataFrame:
        """Remove duplicate text entries.
        
        Args:
            keep: 'first', 'last', or False (remove all duplicates)
            
        Returns:
            DataFrame with duplicates removed
        """
        original_len = len(self.df)
        self.df = self.df.drop_duplicates(subset=[self.text_column], keep=keep)
        removed = original_len - len(self.df)
        
        self.log.append({'action': 'remove_duplicates', 'removed': removed})
        return self.df
    
    def filter_by_length(self, min_length: int = 5, max_length: int = 5000) -> pd.DataFrame:
        """Filter text by length constraints.
        
        Args:
            min_length: Minimum text length
            max_length: Maximum text length
            
        Returns:
            Filtered DataFrame
        """
        original_len = len(self.df)
        text_lengths = self.df[self.text_column].str.len()
        self.df = self.df[(text_lengths >= min_length) & (text_lengths <= max_length)]
        removed = original_len - len(self.df)
        
        self.log.append({'action': 'filter_by_length', 'removed': removed, 'min': min_length, 'max': max_length})
        return self.df
    
    def get_vocabulary(self, top_n: int = 100) -> Dict[str, int]:
        """Get most common words in corpus.
        
        Args:
            top_n: Number of top words to return
            
        Returns:
            Dictionary of word frequencies
        """
        all_text = ' '.join(self.df[self.text_column].fillna(''))
        words = all_text.lower().split()
        counter = Counter(words)
        return dict(counter.most_common(top_n))
    
    def report(self) -> Dict[str, Any]:
        """Generate preprocessing report.
        
        Returns:
            Report dictionary
        """
        return {
            'original_shape': self.original_df.shape,
            'current_shape': self.df.shape,
            'text_column': self.text_column,
            'logs': self.log,
            'issues': self.detect_issues()
        }
    
    def save_report(self, output_path: str) -> None:
        """Save preprocessing report to file.
        
        Args:
            output_path: Path to save report (supports .json, .html)
        """
        from .reports import generate_json_report, generate_html_report
        
        report = self.report()
        
        if output_path.endswith('.json'):
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(generate_json_report(report))
        elif output_path.endswith('.html'):
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(generate_html_report(report))
        else:
            raise ValueError("Output path must end with .json or .html")
