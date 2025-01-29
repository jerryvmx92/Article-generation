"""Image generation using Flux Pro API."""

import os
from typing import Dict, Optional

import fal_client
from dotenv import load_dotenv

load_dotenv()

class ImageGenerator:
    def __init__(self):
        self.api_key = os.getenv("FAL_KEY")
        if not self.api_key:
            raise ValueError("FAL_KEY environment variable is not set")
        
    async def generate_image(
        self,
        prompt: str,
        aspect_ratio: str = "16:9",
        num_images: int = 3,
        seed: Optional[int] = None,
    ) -> Dict[str, str]:
        """Generate an image using Flux Pro API."""
        arguments = {
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "num_images": num_images,
            "output_format": "jpeg",
            "enable_safety_checker": True,
            "safety_tolerance": "2"
        }
        if seed is not None:
            arguments["seed"] = seed
            
        # Create a handler for the request
        handler = fal_client.submit(
            "fal-ai/flux-pro/v1.1-ultra",
            arguments=arguments
        )
        
        # Wait for the result
        result = fal_client.result("fal-ai/flux-pro/v1.1-ultra", handler.request_id)
        
        return {
            "prompt": prompt,
            "images": [img["url"] for img in result["images"]],
        }
            
    def _create_image_prompt(self, article_content: str) -> str:
        """Create an optimized prompt for image generation based on article content."""
        return f"""Create a professional, high-quality image that represents the following article content, set in Coatzacoalcos, Mexico:
{article_content[:500]}...

Visual Style Requirements:
- Professional and modern business look
- High contrast and clear composition
- Suitable for a multiservice company website
- Avoid text overlays or watermarks
- Natural lighting with realistic Gulf Coast atmosphere

Regional Elements to Include:
- Coastal industrial environment elements
- Gulf of Mexico influence
- Local architectural styles
- Relevant service equipment or tools
- Professional service personnel if applicable

Technical Requirements:
- Sharp focus on main subject
- High-resolution details
- Balanced color palette
- Professional lighting that shows texture and depth
- Clear foreground and background separation

Composition Guidelines:
- Rule of thirds for main elements
- Clear focal point
- Professional depth of field
- Balanced negative space
- Strong leading lines

Avoid:
- Generic stock photo look
- Unrealistic or staged scenes
- Oversaturated colors
- Text in the image
- Obvious AI artifacts

The image should convey:
- Professional expertise
- Local relevance
- Trust and reliability
- Quality service
- Understanding of regional challenges""" 