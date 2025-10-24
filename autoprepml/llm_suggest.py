"""LLM integration for AutoPrepML - AI-powered data preprocessing suggestions"""
import os
from typing import Optional, Dict, Any, List
import pandas as pd
import json
from enum import Enum

try:
    from .config_manager import AutoPrepMLConfig
    HAS_CONFIG_MANAGER = True
except ImportError:
    HAS_CONFIG_MANAGER = False


class LLMProvider(Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    OLLAMA = "ollama"  # Local LLM


class LLMSuggestor:
    """LLM-powered suggestions for data preprocessing.
    
    Supports multiple providers:
    - OpenAI (GPT-4, GPT-3.5)
    - Anthropic (Claude)
    - Google (Gemini)
    - Ollama (Local LLMs: llama2, mistral, etc.)
    
    Example:
        >>> # With OpenAI
        >>> suggestor = LLMSuggestor(provider='openai', api_key='sk-...')
        >>> suggestions = suggestor.suggest_fix(df, column='age', issue_type='missing')
        
        >>> # With local Ollama
        >>> suggestor = LLMSuggestor(provider='ollama', model='llama2')
        >>> suggestions = suggestor.analyze_dataframe(df)
    """
    
    def __init__(
        self,
        provider: str = "openai",
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ):
        """Initialize LLM Suggestor with fully dynamic configuration.
        
        Args:
            provider: LLM provider ('openai', 'anthropic', 'google', 'ollama')
            api_key: API key for the provider (not needed for Ollama).
                    Priority: 1. Parameter 2. Config file 3. Environment variable
            model: Model name (any valid model for the provider).
                   If not provided, checks <PROVIDER>_MODEL env var, then uses default.
                   Examples: 'gpt-4o', 'claude-3-5-sonnet-20241022', 'gemini-2.5-flash', 'llama3.2'
            base_url: Custom base URL (for Ollama or custom endpoints).
                      Default: Environment variable <PROVIDER>_BASE_URL or provider default
            temperature: Sampling temperature (0-1, higher = more creative).
                        Default: Environment variable <PROVIDER>_TEMPERATURE or 0.7
            max_tokens: Maximum tokens in response.
                       Default: Environment variable <PROVIDER>_MAX_TOKENS or 500
        
        Environment Variables for Dynamic Configuration:
            - <PROVIDER>_API_KEY: API key (e.g., GOOGLE_API_KEY)
            - <PROVIDER>_MODEL: Model name (e.g., GOOGLE_MODEL=gemini-2.5-pro)
            - <PROVIDER>_BASE_URL: Custom endpoint URL
            - <PROVIDER>_TEMPERATURE: Temperature setting (0.0-1.0)
            - <PROVIDER>_MAX_TOKENS: Max output tokens
            - <PROVIDER>_DEFAULT_MODEL: Default model if none specified
            - GOOGLE_SAFETY_LEVEL: Safety filter level (BLOCK_NONE, BLOCK_LOW, etc.)
        """
        self.provider = LLMProvider(provider.lower())
        
        # Try to get API key from multiple sources
        if api_key:
            self.api_key = api_key
        elif HAS_CONFIG_MANAGER:
            # Try config manager first
            self.api_key = AutoPrepMLConfig.get_api_key(provider.lower())
        else:
            # Fallback to environment variable
            self.api_key = os.getenv(f"{provider.upper()}_API_KEY")
        
        # Warn if API key is missing (except for Ollama)
        if not self.api_key and self.provider != LLMProvider.OLLAMA:
            print(f"⚠️  Warning: No API key found for {self.provider.value}")
            print(f"   Set it with: autoprepml-config --set {self.provider.value}")
            print(f"   Or set environment variable: {provider.upper()}_API_KEY")
        
        # Allow dynamic configuration via environment variables
        provider_upper = self.provider.value.upper()
        self.temperature = (
            temperature if temperature is not None 
            else float(os.getenv(f"{provider_upper}_TEMPERATURE", "0.7"))
        )
        self.max_tokens = (
            max_tokens if max_tokens is not None 
            else int(os.getenv(f"{provider_upper}_MAX_TOKENS", "500"))
        )
        self.base_url = (
            base_url or 
            os.getenv(f"{provider_upper}_BASE_URL") or
            self._get_default_base_url()
        )
        
        # Set model with priority: parameter > env var > default
        self.model = model or self._get_default_model()
        
        # Initialize client
        self.client = self._initialize_client()
    
    def _get_default_base_url(self) -> Optional[str]:
        """Get default base URL for each provider"""
        defaults = {
            LLMProvider.OPENAI: "https://api.openai.com/v1",
            LLMProvider.ANTHROPIC: "https://api.anthropic.com",
            LLMProvider.GOOGLE: None,  # Uses Google's SDK default
            LLMProvider.OLLAMA: "http://localhost:11434"
        }
        return defaults.get(self.provider)
        
    def _get_default_model(self) -> str:
        """Get default model for each provider.
        
        Note: These are fallback defaults. Users can override by passing model parameter
        or setting environment variables: <PROVIDER>_MODEL (e.g., GOOGLE_MODEL=gemini-2.5-pro)
        """
        # Check environment variable first for dynamic model selection
        if (env_model := os.getenv(f"{self.provider.value.upper()}_MODEL")):
            return env_model

        # Fallback to reasonable defaults if no model specified
        defaults = {
            LLMProvider.OPENAI: os.getenv("OPENAI_DEFAULT_MODEL", "gpt-4o"),
            LLMProvider.ANTHROPIC: os.getenv("ANTHROPIC_DEFAULT_MODEL", "claude-3-5-sonnet-20241022"),
            LLMProvider.GOOGLE: os.getenv("GOOGLE_DEFAULT_MODEL", "gemini-2.5-flash"),
            LLMProvider.OLLAMA: os.getenv("OLLAMA_DEFAULT_MODEL", "llama3.2")
        }
        return defaults[self.provider]
    
    def _initialize_client(self):
        """Initialize the appropriate LLM client with dynamic configuration"""
        try:
            if self.provider == LLMProvider.OPENAI:
                from openai import OpenAI
                client_kwargs = {"api_key": self.api_key}
                if self.base_url and self.base_url != "https://api.openai.com/v1":
                    client_kwargs["base_url"] = self.base_url
                return OpenAI(**client_kwargs)

            elif self.provider == LLMProvider.ANTHROPIC:
                from anthropic import Anthropic
                client_kwargs = {"api_key": self.api_key}
                if self.base_url and self.base_url != "https://api.anthropic.com":
                    client_kwargs["base_url"] = self.base_url
                return Anthropic(**client_kwargs)

            elif self.provider == LLMProvider.GOOGLE:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                # Model is set dynamically, can be any valid Gemini model
                return genai.GenerativeModel(self.model)

            elif self.provider == LLMProvider.OLLAMA:
                try:
                    import ollama
                    # Ollama client can use custom base URL
                    return ollama
                except ImportError as e:
                    raise ImportError(
                        "Ollama not installed. Install with: pip install ollama\n"
                        "Also make sure Ollama is running locally: https://ollama.ai"
                    ) from e
        except ImportError as e:
            raise ImportError(
                f"Provider '{self.provider.value}' requires additional packages.\n"
                f"Install with: pip install autoprepml[llm] or pip install {self.provider.value}\n"
                f"Original error: {str(e)}"
            ) from e
    
    def _call_llm(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Call LLM with unified interface"""
        try:
            if self.provider == LLMProvider.OPENAI:
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                return response.choices[0].message.content
                
            elif self.provider == LLMProvider.ANTHROPIC:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    system=system_prompt or "You are a data preprocessing expert.",
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text
                
            elif self.provider == LLMProvider.GOOGLE:
                full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
                
                # Allow custom safety settings via environment variable
                safety_level = os.getenv("GOOGLE_SAFETY_LEVEL", "BLOCK_NONE")
                
                response = self.client.generate_content(
                    full_prompt,
                    generation_config={
                        "temperature": self.temperature,
                        "max_output_tokens": self.max_tokens
                    },
                    safety_settings={
                        "HARM_CATEGORY_HATE_SPEECH": safety_level,
                        "HARM_CATEGORY_HARASSMENT": safety_level,
                        "HARM_CATEGORY_SEXUALLY_EXPLICIT": safety_level,
                        "HARM_CATEGORY_DANGEROUS_CONTENT": safety_level
                    }
                )
                # Check if response was blocked
                if response.candidates and response.candidates[0].finish_reason != 1:  # 1 = STOP (success)
                    return f"Response blocked by safety filters (reason: {response.candidates[0].finish_reason}). Try rephrasing your query or adjust GOOGLE_SAFETY_LEVEL environment variable."
                return response.text
                
            elif self.provider == LLMProvider.OLLAMA:
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})
                
                # Support custom Ollama host via base_url or environment
                chat_kwargs = {
                    "model": self.model,
                    "messages": messages,
                    "options": {
                        "temperature": self.temperature,
                        "num_predict": self.max_tokens
                    }
                }
                
                # Add host if custom base_url is set
                if self.base_url and self.base_url != "http://localhost:11434":
                    chat_kwargs["host"] = self.base_url
                
                response = self.client.chat(**chat_kwargs)
                return response['message']['content']
                
        except Exception as e:
            return f"Error calling {self.provider.value}: {str(e)}"
    
    def suggest_fix(
        self,
        df: pd.DataFrame,
        column: Optional[str] = None,
        issue_type: str = 'missing'
    ) -> str:
        """Generate LLM-powered suggestions for data cleaning.
        
        Args:
            df: Input DataFrame
            column: Specific column to analyze
            issue_type: Type of issue ('missing', 'outlier', 'imbalance', 'duplicates')
            
        Returns:
            AI-generated suggestion text
        """
        # Gather column information
        if column and column in df.columns:
            col_info = self._get_column_info(df, column)
        else:
            col_info = self._get_dataframe_summary(df)
        
        system_prompt = """You are an expert data scientist specializing in data preprocessing 
        and cleaning for machine learning. Provide concise, actionable recommendations."""
        
        prompt = f"""
Analyze this data quality issue and provide specific preprocessing recommendations:

**Issue Type**: {issue_type}
**Column**: {column or 'Multiple columns'}

**Data Characteristics**:
{json.dumps(col_info, indent=2)}

**Task**: Provide:
1. Root cause analysis of the issue
2. 2-3 specific preprocessing strategies (with pros/cons)
3. Recommended approach with rationale
4. Code snippet example if applicable

Keep response concise and actionable (max 300 words).
"""
        
        return self._call_llm(prompt, system_prompt)
    
    def analyze_dataframe(
        self,
        df: pd.DataFrame,
        task: str = 'classification',
        target_col: Optional[str] = None
    ) -> Dict[str, Any]:
        """Comprehensive DataFrame analysis with preprocessing recommendations.
        
        Args:
            df: Input DataFrame
            task: ML task ('classification', 'regression', 'clustering')
            target_col: Target column name (if applicable)
            
        Returns:
            Dictionary with analysis and recommendations
        """
        summary = self._get_dataframe_summary(df, target_col)

        system_prompt = """You are an expert ML engineer. Analyze data and provide 
        a comprehensive preprocessing pipeline recommendation."""

        prompt = f"""
Analyze this dataset and recommend a complete preprocessing pipeline:

**ML Task**: {task}
**Target Column**: {target_col or 'Not specified'}

**Dataset Summary**:
{json.dumps(summary, indent=2)}

**Provide**:
1. Data Quality Assessment (score 1-10)
2. Critical Issues (prioritized list)
3. Recommended Preprocessing Pipeline (step-by-step)
4. Feature Engineering Suggestions
5. Potential Pitfalls to Avoid

Format as JSON with these keys: quality_score, critical_issues, pipeline_steps, 
feature_suggestions, warnings
"""

        response_text = self._call_llm(prompt, system_prompt)

        # Try to parse as JSON, fallback to text
        try:
            # Extract JSON if embedded in markdown code blocks
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()
            elif "```" in response_text:
                json_start = response_text.find("```") + 3
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()

            return json.loads(response_text)
        except Exception:
            return {"raw_response": response_text}
    
    def explain_cleaning_step(
        self,
        action: str,
        details: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate natural language explanation of a cleaning step.
        
        Args:
            action: Action type (e.g., 'imputed_missing', 'scaled_features')
            details: Dictionary with action details
            context: Optional context about the data
            
        Returns:
            Human-readable explanation
        """
        system_prompt = """You are explaining data preprocessing steps to a non-technical 
        audience. Be clear, concise, and explain why each step matters."""
        
        prompt = f"""
Explain this data preprocessing step in simple terms:

**Action**: {action}
**Details**: {json.dumps(details, indent=2)}
**Context**: {json.dumps(context, indent=2) if context else 'Not provided'}

Provide:
1. What was done (1 sentence)
2. Why it was necessary (1 sentence)
3. Impact on the data (1 sentence)

Total: max 3 sentences, non-technical language.
"""
        
        return self._call_llm(prompt, system_prompt)
    
    def suggest_features(
        self,
        df: pd.DataFrame,
        task: str = 'classification',
        target_col: Optional[str] = None
    ) -> List[str]:
        """Suggest new features to create based on existing data.
        
        Args:
            df: Input DataFrame
            task: ML task type
            target_col: Target column
            
        Returns:
            List of feature engineering suggestions
        """
        summary = self._get_dataframe_summary(df, target_col)

        system_prompt = """You are a feature engineering expert. Suggest creative, 
        impactful features based on domain knowledge and ML best practices."""

        prompt = f"""
Suggest feature engineering strategies for this dataset:

**Task**: {task}
**Columns**: {list(df.columns)}
**Summary**: {json.dumps(summary, indent=2)}

Suggest 5-10 new features to create, including:
- Feature name
- Calculation/creation method
- Expected impact on model performance

Return as a JSON array of objects with keys: name, method, impact
"""

        response_text = self._call_llm(prompt, system_prompt)

        try:
            # Parse JSON response
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()

            features = json.loads(response_text)
            return features if isinstance(features, list) else [response_text]
        except Exception:
            return [response_text]
    
    def _get_column_info(self, df: pd.DataFrame, column: str) -> Dict[str, Any]:
        """Extract detailed information about a column"""
        if column not in df.columns:
            return {
                "error": f"Column '{column}' not found in DataFrame",
                "available_columns": list(df.columns)
            }
        
        col = df[column]

        info = {
            "dtype": str(col.dtype),
            "missing_count": int(col.isnull().sum()),
            "missing_pct": float(col.isnull().sum() / len(df) * 100),
            "unique_values": int(col.nunique()),
            "total_rows": len(df)
        }

        # Numeric columns
        if pd.api.types.is_numeric_dtype(col):
            info |= {
                "mean": None if col.isnull().all() else float(col.mean()),
                "median": None if col.isnull().all() else float(col.median()),
                "std": None if col.isnull().all() else float(col.std()),
                "min": None if col.isnull().all() else float(col.min()),
                "max": None if col.isnull().all() else float(col.max()),
                "sample_values": col.dropna().head(5).tolist(),
            }
        else:
            # Categorical columns
            value_counts = col.value_counts().head(5)
            info |= {
                "top_values": value_counts.to_dict(),
                "sample_values": col.dropna().head(5).tolist(),
            }

        return info
    
    def _get_dataframe_summary(
        self,
        df: pd.DataFrame,
        target_col: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get comprehensive DataFrame summary"""
        summary = {
            "shape": {"rows": df.shape[0], "columns": df.shape[1]},
            "columns": list(df.columns),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "missing_pct": (df.isnull().sum() / len(df) * 100).to_dict(),
            "duplicate_rows": int(df.duplicated().sum()),
            "memory_usage_mb": float(df.memory_usage(deep=True).sum() / 1024 / 1024)
        }
        
        # Numeric columns summary
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            summary["numeric_summary"] = df[numeric_cols].describe().to_dict()
        
        # Categorical columns summary
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            summary["categorical_summary"] = {
                col: df[col].value_counts().head(5).to_dict()
                for col in categorical_cols[:5]  # Limit to first 5
            }
        
        # Target column analysis (if specified)
        if target_col and target_col in df.columns:
            target_info = {
                "value_counts": df[target_col].value_counts().to_dict(),
                "unique_values": int(df[target_col].nunique())
            }
            if pd.api.types.is_numeric_dtype(df[target_col]):
                target_info["distribution"] = {
                    "mean": float(df[target_col].mean()),
                    "std": float(df[target_col].std())
                }
            summary["target_column"] = target_info
        
        return summary
    
    def suggest_column_rename(self, df: pd.DataFrame, column: str) -> str:
        """Suggest a better column name based on data content.
        
        Args:
            df: DataFrame
            column: Column to analyze and rename
            
        Returns:
            Suggested column name
        """
        if column not in df.columns:
            return f"Error: Column '{column}' not found"
        
        col_info = self._get_column_info(df, column)
        
        prompt = f"""Suggest a clear, descriptive column name for this data column.

Current name: {column}
Data type: {col_info['dtype']}
Unique values: {col_info['unique_values']}
Sample values: {col_info.get('sample_values', [])}

Provide ONLY the suggested column name in snake_case format (e.g., customer_age, product_price).
No explanations, just the name."""
        
        suggestion = self._call_llm(prompt)
        
        # Extract just the column name (remove any explanations)
        suggested_name = suggestion.strip().split('\n')[0].strip()
        # Remove any quotes or extra characters
        suggested_name = suggested_name.replace('"', '').replace("'", '').strip()
        
        return suggested_name
    
    def suggest_all_column_renames(self, df: pd.DataFrame) -> Dict[str, str]:
        """Suggest better names for all columns.
        
        Args:
            df: DataFrame
            
        Returns:
            Dictionary mapping old names to suggested new names
        """
        rename_map = {}
        
        for col in df.columns:
            try:
                suggested = self.suggest_column_rename(df, col)
                if suggested and suggested != col:
                    rename_map[col] = suggested
            except Exception as e:
                print(f"Could not rename '{col}': {e}")
        
        return rename_map
    
    def explain_data_quality_issues(self, df: pd.DataFrame) -> str:
        """Explain detected data quality issues in natural language.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Natural language explanation of issues
        """
        summary = self._get_dataframe_summary(df)
        
        prompt = f"""Analyze this dataset and explain any data quality issues found:

Dataset shape: {summary['shape']['rows']} rows, {summary['shape']['columns']} columns
Missing values: {sum(summary['missing_values'].values())} total
Duplicate rows: {summary['duplicate_rows']}

Column missing values: {dict(list(summary['missing_values'].items())[:10])}

Provide a brief, actionable summary of data quality issues and recommendations."""
        
        return self._call_llm(prompt)
    
    def generate_data_documentation(self, df: pd.DataFrame) -> str:
        """Generate comprehensive markdown documentation for the dataset.
        
        Args:
            df: DataFrame to document
            
        Returns:
            Markdown formatted documentation
        """
        summary = self._get_dataframe_summary(df)
        
        prompt = f"""Generate comprehensive markdown documentation for this dataset:

Dataset: {summary['shape']['rows']} rows × {summary['shape']['columns']} columns

Columns: {', '.join(summary['columns'][:20])}
Data types: {summary['dtypes']}

Create markdown documentation with:
1. Overview section
2. Column descriptions (purpose, type, range)
3. Data quality notes
4. Recommended preprocessing steps

Format as clean, professional markdown."""
        
        return self._call_llm(prompt)
    
    def suggest_preprocessing_pipeline(self, df: pd.DataFrame, task: str = 'classification') -> str:
        """Suggest a complete preprocessing pipeline for ML task.
        
        Args:
            df: DataFrame
            task: ML task ('classification', 'regression', 'clustering')
            
        Returns:
            Suggested preprocessing steps
        """
        summary = self._get_dataframe_summary(df)
        
        prompt = f"""Suggest a complete preprocessing pipeline for this dataset:

Task: {task}
Shape: {summary['shape']['rows']} rows, {summary['shape']['columns']} columns
Missing values: {sum(summary['missing_values'].values())}
Duplicates: {summary['duplicate_rows']}

Numeric columns: {len([k for k,v in summary['dtypes'].items() if 'int' in str(v) or 'float' in str(v)])}
Categorical columns: {len([k for k,v in summary['dtypes'].items() if 'object' in str(v)])}

Provide a step-by-step preprocessing pipeline with:
1. Data cleaning steps
2. Feature engineering suggestions
3. Encoding/scaling methods
4. Feature selection recommendations

Be specific and actionable."""
        
        return self._call_llm(prompt)


# Convenience functions for backward compatibility
def suggest_fix(
    df: pd.DataFrame,
    column: Optional[str] = None,
    issue_type: str = 'missing',
    provider: str = 'openai',
    api_key: Optional[str] = None
) -> str:
    """Generate LLM-powered suggestions for data cleaning.
    
    Args:
        df: Input DataFrame
        column: Optional specific column to analyze
        issue_type: Type of issue ('missing', 'outlier', 'imbalance')
        provider: LLM provider ('openai', 'anthropic', 'google', 'ollama')
        api_key: API key (optional if set in environment)
        
    Returns:
        Suggestion text string
        
    Example:
        >>> suggestions = suggest_fix(df, column='age', issue_type='missing', provider='openai')
        >>> # Or with local LLM
        >>> suggestions = suggest_fix(df, column='age', provider='ollama')
    """
    suggestor = LLMSuggestor(provider=provider, api_key=api_key)
    return suggestor.suggest_fix(df, column, issue_type)


def explain_cleaning_step(
    action: str,
    details: Dict[str, Any],
    provider: str = 'openai',
    api_key: Optional[str] = None
) -> str:
    """Generate natural language explanation of a cleaning step.
    
    Args:
        action: Action type (e.g., 'imputed_missing', 'scaled_features')
        details: Dictionary with action details
        provider: LLM provider
        api_key: API key (optional)
        
    Returns:
        Human-readable explanation
    """
    suggestor = LLMSuggestor(provider=provider, api_key=api_key)
    return suggestor.explain_cleaning_step(action, details)


def suggest_column_rename(
    df: pd.DataFrame,
    column: str,
    provider: str = 'openai',
    api_key: Optional[str] = None
) -> str:
    """Suggest better column name based on data content.
    
    Args:
        df: DataFrame
        column: Column to rename
        provider: LLM provider
        api_key: API key (optional)
        
    Returns:
        Suggested column name
    """
    suggestor = LLMSuggestor(provider=provider, api_key=api_key)
    return suggestor.suggest_column_rename(df, column)


def generate_data_documentation(
    df: pd.DataFrame,
    provider: str = 'openai',
    api_key: Optional[str] = None
) -> str:
    """Generate comprehensive data documentation.
    
    Args:
        df: DataFrame to document
        provider: LLM provider
        api_key: API key (optional)
        
    Returns:
        Markdown documentation
    """
    suggestor = LLMSuggestor(provider=provider, api_key=api_key)
    return suggestor.generate_data_documentation(df)
