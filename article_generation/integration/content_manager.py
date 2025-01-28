"""Content manager for integrating article and image generation."""

import os
from typing import Dict, List, Optional

from ..llm.generator import ArticleGenerator
from ..image_gen.generator import ImageGenerator

class ContentManager:
    def __init__(self):
        self.article_generator = ArticleGenerator()
        self.image_generator = ImageGenerator()
        
    async def generate_content(
        self,
        topic: str,
        keywords: List[str],
        num_images: int = 1,
        aspect_ratio: str = "16:9",
    ) -> Dict:
        """Generate a complete article with associated images."""
        # Generate the article
        article = await self.article_generator.generate_article(topic, keywords)
        
        # Generate image based on article content
        image_prompt = self.image_generator._create_image_prompt(article["content"])
        images = await self.image_generator.generate_image(
            prompt=image_prompt,
            aspect_ratio=aspect_ratio,
            num_images=num_images
        )
        
        return {
            "article": article,
            "images": images["images"],
        }
        
    def save_content(
        self,
        content: Dict,
        base_path: Optional[str] = None
    ) -> Dict[str, str]:
        """Save the generated content to disk."""
        base_path = base_path or os.getcwd()
        articles_path = os.path.join(base_path, "generated_articles")
        images_path = os.path.join(base_path, "generated_images")
        
        # Create directories if they don't exist
        os.makedirs(articles_path, exist_ok=True)
        os.makedirs(images_path, exist_ok=True)
        
        # Save article
        article_filename = f"{content['article']['title'].lower().replace(' ', '_')}.md"
        article_path = os.path.join(articles_path, article_filename)
        with open(article_path, "w") as f:
            f.write(content["article"]["content"])
            
        # Save image URLs (actual image download could be implemented if needed)
        images_filename = f"{content['article']['title'].lower().replace(' ', '_')}_images.txt"
        images_path = os.path.join(images_path, images_filename)
        with open(images_path, "w") as f:
            f.write("\n".join(content["images"]))
            
        return {
            "article_path": article_path,
            "images_path": images_path,
        } 