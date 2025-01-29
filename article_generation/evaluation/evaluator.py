"""Article quality evaluator using LLMs."""

import os
from typing import Dict, Any, List, Optional
from anthropic import AsyncAnthropic
import logging

logger = logging.getLogger(__name__)

class ArticleEvaluator:
    """Evaluates article quality using a stronger LLM."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the evaluator.
        
        Args:
            api_key: Optional API key. If not provided, will try to get from environment.
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("API key must be provided either directly or via ANTHROPIC_API_KEY environment variable")
            
        self.client = AsyncAnthropic(api_key=self.api_key)
        # Use a more powerful model for evaluation
        self.model = os.getenv("ANTHROPIC_EVAL_MODEL", "claude-3-opus-20240229")
        self.max_tokens = int(os.getenv("EVAL_MAX_TOKENS", "4096"))
        self.temperature = float(os.getenv("EVAL_TEMPERATURE", "0.3"))  # Lower temperature for more consistent evaluation
    
    async def evaluate_article(
        self,
        content: str,
        title: str,
        keywords: List[str],
        min_length: Optional[int] = None,
        max_length: Optional[int] = None
    ) -> Dict[str, Any]:
        """Evaluate an article's quality.
        
        Args:
            content: The article content
            title: The article title
            keywords: Expected keywords
            min_length: Minimum expected length
            max_length: Maximum expected length
            
        Returns:
            Dictionary containing evaluation results
        """
        prompt = f"""You are an expert content evaluator specializing in SEO-optimized articles.
Please evaluate this article thoroughly and provide a structured critique.

Article Title: {title}
Expected Keywords: {', '.join(keywords)}
Length Requirements: {min_length or 'Not specified'} - {max_length or 'Not specified'} words

Article Content:
{content}

Evaluate the following aspects and provide a score from 1-10 for each:

1. Structure and Formatting
- Does it follow proper markdown formatting?
- Are sections clearly organized?
- Is the content well-structured?

2. Content Quality
- Is the content comprehensive and informative?
- Is it engaging and well-written?
- Does it provide value to the reader?

3. SEO Optimization
- Are keywords naturally incorporated?
- Is the content optimized for search engines?
- Does it follow SEO best practices?

4. Regional Relevance
- Is the content relevant to Coatzacoalcos and the Gulf coast region?
- Does it address local challenges and context?
- Is it tailored to the target audience?

For each aspect:
1. Provide a score (1-10)
2. List specific strengths
3. List areas for improvement
4. Give actionable recommendations

Finally, provide:
1. An overall score (1-10)
2. A summary of the main issues (if any)
3. Top 3 recommendations for improvement

Format your response as JSON with the following structure:
{{
    "structure_score": 8,
    "structure_strengths": ["..."],
    "structure_improvements": ["..."],
    "structure_recommendations": ["..."],
    // ... similar for other aspects
    "overall_score": 7,
    "main_issues": ["..."],
    "top_recommendations": ["..."]
}}"""

        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system="You are an expert content evaluator. Provide detailed, objective evaluations in JSON format.",
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            
            # Extract and parse JSON from response
            content = response.content[0].text
            try:
                evaluation = eval(content)  # Safe since we control the input format
                return evaluation
            except Exception as e:
                logger.error(f"Failed to parse evaluation response: {e}")
                return {
                    "error": "Failed to parse evaluation",
                    "raw_response": content
                }
                
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return {
                "error": str(e)
            } 