"""Article generation using Anthropic's Claude."""

import os
from typing import Dict, List, Optional

from anthropic import AsyncAnthropic
from dotenv import load_dotenv

load_dotenv()

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

    async def generate_article(
        self,
        topic: str,
        keywords: List[str],
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
    ) -> Dict[str, str]:
        """Generate an SEO-optimized article on the given topic."""
        prompt = self._create_seo_prompt(topic, keywords, min_length, max_length)
        
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
            
            return {
                "title": topic,
                "content": content,
                "keywords": keywords
            }
        except Exception as e:
            print("\n=== API Error in generate_article ===")
            print(f"Error type: {type(e)}")
            print(f"Error message: {str(e)}")
            if hasattr(e, 'response'):
                print(f"Response status: {e.response.status_code}")
                print(f"Response body: {e.response.text}")
            if hasattr(e, 'request'):
                print(f"Request details: {e.request}")
            raise

    def _create_seo_prompt(
        self,
        topic: str,
        keywords: List[str],
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
    ) -> str:
        """Create an SEO-optimized prompt for article generation."""
        min_length = min_length or int(os.getenv("MIN_ARTICLE_LENGTH", "1200"))
        max_length = max_length or int(os.getenv("MAX_ARTICLE_LENGTH", "3000"))
        
        return f"""Generate a comprehensive, SEO-optimized article about {topic} for a multiservice company based in Coatzacoalcos, Mexico.

Regional Context:
- Target audience: Residential and commercial clients in Coatzacoalcos and nearby cities (Minatitlán)
- Consider the coastal environment (Gulf of Mexico) and its challenges
- Address high humidity and corrosion issues common in the region
- Reference local industrial infrastructure and petrochemical industry presence
- Include regional business opportunities and service coverage area

Article Structure:
1. Introduction:
   - Hook readers in the first paragraph
   - Include primary keyword within first 100 words
   - Establish local context and relevance
   - Preview main points

2. Body Content:
   - Length: Between {min_length} and {max_length} words
   - Organize with H2 and H3 subheadings
   - Include these keywords naturally: {', '.join(keywords)}
   - Use short, scannable paragraphs (2-4 sentences)
   - Include relevant statistics and data when possible
   - Address specific regional challenges and solutions

3. Conclusion:
   - Summarize key points
   - Include clear call-to-action
   - Reinforce local expertise and service value

Format the article in Markdown, using proper heading hierarchy (H1 → H2 → H3).

Remember to:
- Maintain a professional yet approachable tone
- Focus on local relevance and specific regional challenges
- Include practical examples relevant to the Gulf coast region
- Address both residential and commercial service aspects
- Emphasize expertise in dealing with coastal environmental challenges
- Use natural language that resonates with local readers
- Include location-specific details when relevant

Do not include any YAML frontmatter, metadata, schema markup, or technical SEO elements in the output. Just provide the clean article content in markdown format.""" 