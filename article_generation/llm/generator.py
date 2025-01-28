"""Article generation using Anthropic's Claude."""

import os
from typing import Dict, List, Optional

from anthropic import AsyncAnthropic
from dotenv import load_dotenv

load_dotenv()

class ArticleGenerator:
    def __init__(self):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is not set")
            
        self.client = AsyncAnthropic(api_key=api_key)
        self.model = os.getenv("ANTHROPIC_MODEL", "claude-3-sonnet-20240229")
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
            message = await self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system="You are an expert SEO content writer specializing in creating high-quality, engaging articles.",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            return {
                "title": topic,
                "content": message.content[0].text,
                "keywords": keywords
            }
        except Exception as e:
            print(f"Error in generate_article: {str(e)}")
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

Article Structure Requirements:
1. Title (H1):
   - Create an attention-grabbing, SEO-friendly title under 65 characters
   - Include primary keyword naturally

2. Meta Description:
   - Write a compelling 150-160 character meta description
   - Include primary keyword and call-to-action

3. Introduction:
   - Hook readers in the first paragraph
   - Include primary keyword within first 100 words
   - Establish local context and relevance
   - Preview main points

4. Body Content:
   - Length: Between {min_length} and {max_length} words
   - Organize with H2 and H3 subheadings
   - Include these keywords naturally: {', '.join(keywords)}
   - Use short, scannable paragraphs (2-4 sentences)
   - Include relevant statistics and data when possible
   - Address specific regional challenges and solutions
   - Add internal link suggestions in [brackets]

5. Technical SEO Elements:
   - Use proper heading hierarchy (H1 → H2 → H3)
   - Include location-specific keywords
   - Add schema markup suggestions
   - Suggest image placement points with alt text

6. Conclusion:
   - Summarize key points
   - Include clear call-to-action
   - Reinforce local expertise and service value

7. Additional Metadata:
   - Suggest 3-5 category tags
   - Estimate reading time
   - Add relevant internal linking opportunities

Format the article in Markdown, using proper heading hierarchy and including all metadata in a front matter section.

Remember to:
- Maintain a professional yet approachable tone
- Focus on local relevance and specific regional challenges
- Include practical examples relevant to the Gulf coast region
- Address both residential and commercial service aspects
- Emphasize expertise in dealing with coastal environmental challenges
- Use natural language that resonates with local readers
- Include location-specific details when relevant""" 