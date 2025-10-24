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
        temperature: float = 0.7,
        max_tokens: int = 500
    ):
        """Initialize LLM Suggestor.
        
        Args:
            provider: LLM provider ('openai', 'anthropic', 'google', 'ollama')
            api_key: API key for the provider (not needed for Ollama).
                    If not provided, will check:
                    1. AutoPrepML config file (~/.autoprepml/config.json)
                    2. Environment variables (OPENAI_API_KEY, etc.)
            model: Model name (e.g., 'gpt-4', 'claude-3', 'gemini-pro', 'llama2')
            base_url: Custom base URL (for Ollama or custom endpoints)
            temperature: Sampling temperature (0-1, higher = more creative)
            max_tokens: Maximum tokens in response
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
        
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = base_url
        
        # Set default models
        self.model = model or self._get_default_model()
        
        # Initialize client
        self.client = self._initialize_client()
        
    def _get_default_model(self) -> str:
        """Get default model for each provider"""
        defaults = {
            LLMProvider.OPENAI: "gpt-4",
            LLMProvider.ANTHROPIC: "claude-3-sonnet-20240229",
            LLMProvider.GOOGLE: "gemini-pro",
            LLMProvider.OLLAMA: "llama2"
        }
        return defaults[self.provider]
    
    def _initialize_client(self):
        """Initialize the appropriate LLM client"""
        try:
            if self.provider == LLMProvider.OPENAI:
                from openai import OpenAI
                return OpenAI(api_key=self.api_key)

            elif self.provider == LLMProvider.ANTHROPIC:
                from anthropic import Anthropic
                return Anthropic(api_key=self.api_key)

            elif self.provider == LLMProvider.GOOGLE:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                return genai.GenerativeModel(self.model)

            elif self.provider == LLMProvider.OLLAMA:
                try:
                    import ollama
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
                response = self.client.generate_content(
                    full_prompt,
                    generation_config={
                        "temperature": self.temperature,
                        "max_output_tokens": self.max_tokens
                    }
                )
                return response.text
                
            elif self.provider == LLMProvider.OLLAMA:
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})
                
                response = self.client.chat(
                    model=self.model,
                    messages=messages,
                    options={
                        "temperature": self.temperature,
                        "num_predict": self.max_tokens
                    }
                )
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
