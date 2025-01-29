"""Article generation using Anthropic's Claude."""

import os
from typing import Dict, List, Optional
from anthropic import AsyncAnthropic
from dotenv import load_dotenv
import logging

from ..evaluation.trace_logger import TraceLogger
from ..evaluation.evaluator import ArticleEvaluator

load_dotenv()
logger = logging.getLogger(__name__)

class ArticleGenerator:
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the ArticleGenerator.
        
        Args:
            api_key: Optional API key. If not provided, will try to get from environment.
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("API key must be provided either directly or via ANTHROPIC_API_KEY environment variable")
            
        self.client = AsyncAnthropic(api_key=self.api_key)
        self.model = os.getenv("ANTHROPIC_MODEL", "claude-3-opus-20240229")
        self.max_tokens = int(os.getenv("MAX_TOKENS", "4096"))
        self.temperature = float(os.getenv("TEMPERATURE", "0.7"))
        
        # Initialize evaluation tools
        self.trace_logger = TraceLogger()
        self.evaluator = ArticleEvaluator(api_key=self.api_key)

    async def generate_article(
        self,
        title: str,
        keywords: List[str],
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
    ) -> Dict[str, str]:
        """Generate an SEO-optimized article with the given title.
        
        Args:
            title: The exact title to use for the article
            keywords: List of keywords to include in the article
            min_length: Minimum word length (optional)
            max_length: Maximum word length (optional)
            
        Returns:
            Dict containing the article title, content, and keywords
        """
        prompt = self._create_seo_prompt(title, keywords, min_length, max_length)
        
        try:
            print("\n=== Making API Request ===")
            print(f"Model: {self.model}")
            print(f"Max tokens: {self.max_tokens}")
            print(f"Temperature: {self.temperature}")
            print(f"API Key (first 10 chars): {self.api_key[:10]}...")
            print(f"API Key length: {len(self.api_key)}")
            
            message = await self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system="You are an expert SEO content writer specializing in creating high-quality, engaging articles.",
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            
            print("\n=== API Response ===")
            print(f"Response type: {type(message)}")
            if hasattr(message, 'content'):
                print(f"Response content type: {type(message.content)}")
                print(f"Response content: {message.content}")
            
            content = message.content[0].text if hasattr(message, 'content') else message.completion
            
            result = {
                "title": title,
                "content": content,
                "keywords": keywords
            }
            
            # Log the successful generation
            self.trace_logger.log_trace(
                title=title,
                keywords=keywords,
                prompt=prompt,
                response=result
            )
            
            # Evaluate the article
            evaluation = await self.evaluator.evaluate_article(
                content=content,
                title=title,
                keywords=keywords,
                min_length=min_length,
                max_length=max_length
            )
            
            # Add evaluation to result
            result["evaluation"] = evaluation
            
            return result
            
        except Exception as e:
            print("\n=== API Error in generate_article ===")
            print(f"Error type: {type(e)}")
            print(f"Error message: {str(e)}")
            if hasattr(e, 'response'):
                print(f"Response status: {e.response.status_code}")
                print(f"Response body: {e.response.text}")
            if hasattr(e, 'request'):
                print(f"Request details: {e.request}")
                
            # Log the failed generation
            self.trace_logger.log_trace(
                title=title,
                keywords=keywords,
                prompt=prompt,
                response={},
                error=e
            )
            
            raise e from None

    def _create_seo_prompt(
        self,
        title: str,
        keywords: List[str],
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
    ) -> str:
        """Create an SEO-optimized prompt for article generation.
        
        Args:
            title: The exact title to use for the article
            keywords: List of keywords to include
            min_length: Minimum word length (optional)
            max_length: Maximum word length (optional)
            
        Returns:
            The formatted prompt string
        """
        min_length = min_length or int(os.getenv("MIN_ARTICLE_LENGTH", "1200"))
        max_length = max_length or int(os.getenv("MAX_ARTICLE_LENGTH", "3000"))
        
        return f"""IMPORTANT: Generate an SEO-optimized article that MUST follow these EXACT formatting requirements.

The article MUST start with this EXACT title:
# {title}

Then, it MUST use these EXACT section headers in this EXACT order:
1. ## Introduction
2. ## Body Content
3. ## Conclusion

Any deviation from these exact headers will cause the article to be rejected.

Regional Context:
- Target audience: Residential and commercial clients in Coatzacoalcos and nearby cities (Minatitlán)
- Consider the coastal environment (Gulf of Mexico) and its challenges
- Address high humidity and corrosion issues common in the region
- Reference local industrial infrastructure and petrochemical industry presence
- Include regional business opportunities and service coverage area

Required Content Structure:

## Introduction
- Hook readers in the first paragraph
- Include primary keyword within first 100 words
- Establish local context and relevance
- Preview main points
- At least 300 words

## Body Content
- Length: Between {min_length} and {max_length} words
- Use H3 subheadings for subsections
- Include these keywords naturally: {', '.join(keywords)}
- Use short, scannable paragraphs (2-4 sentences)
- Include relevant statistics and data when possible
- Address specific regional challenges and solutions
- At least 700 words

## Conclusion
- Summarize key points
- Include clear call-to-action
- Reinforce local expertise and service value
- At least 200 words

Format Requirements:
- Use proper markdown formatting
- The article MUST start with exactly '# {title}'
- Main sections MUST use exactly '## Introduction', '## Body Content', and '## Conclusion'
- Use '### ' for subsections
- Include at least 3 numbered lists
- Include at least 10 bullet points

Remember to:
- Maintain a professional yet approachable tone
- Focus on local relevance and specific regional challenges
- Include practical examples relevant to the Gulf coast region
- Address both residential and commercial service aspects
- Emphasize expertise in dealing with coastal environmental challenges
- Use natural language that resonates with local readers
- Include location-specific details when relevant

Do not include any YAML frontmatter, metadata, schema markup, or technical SEO elements in the output. Just provide the clean article content in markdown format.""" 