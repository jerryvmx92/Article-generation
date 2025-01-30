"""Script to run an A/B test experiment for article generation."""

import os
import asyncio
from article_generation.llm.generator import ArticleGenerator
from article_generation.evaluation.evaluator import ArticleEvaluator
from article_generation.experimentation.experiment import Experiment
from article_generation.experimentation.feedback import FeedbackManager

# Test cases with topics and keywords
TEST_CASES = [
    {
        "title": "Benefits of Regular AC Maintenance",
        "keywords": ["AC maintenance", "air conditioning", "HVAC service", "energy efficiency"]
    },
    {
        "title": "Industrial Safety Best Practices",
        "keywords": ["workplace safety", "industrial safety", "safety protocols", "risk management"]
    },
    {
        "title": "Construction Project Management",
        "keywords": ["project management", "construction", "timeline planning", "resource allocation"]
    }
]

# Prompt templates for different variants
PROMPT_VARIANTS = {
    "baseline": """Generate an article about {title} that includes these keywords: {keywords}.
The article should be between {min_length} and {max_length} words.

Required sections:
# {title}
## Introduction
## Body Content
## Conclusion""",

    "structured": """Create a highly structured article about {title} with these keywords: {keywords}.
Length: {min_length}-{max_length} words.

Required Structure:
# {title}

## Introduction
- Problem statement
- Current challenges
- Article overview

## Body Content
Must include:
1. Current Industry Status
2. Key Challenges
3. Best Practices
4. Implementation Guide
5. Case Studies
6. Future Trends

## Conclusion
- Summary of key points
- Actionable recommendations
- Next steps""",

    "seo_focused": """Write an SEO-optimized article about {title} targeting these keywords: {keywords}.
Word count: {min_length}-{max_length}

Structure:
# {title}

## Introduction
- Hook with industry statistics
- Clear value proposition
- Topic relevance
- Reader benefits

## Body Content
Must include:
- H3 subheadings for each major point
- Bullet points for key takeaways
- Numbered lists for processes
- Expert quotes or statistics
- Real-world examples
- Industry best practices
- Common challenges and solutions
- Implementation tips

## Conclusion
- Summary of benefits
- Call to action
- Next steps"""
}

async def run_experiment():
    """Run the A/B test experiment."""
    try:
        # Initialize components
        generator = ArticleGenerator()
        evaluator = ArticleEvaluator()
        feedback_manager = FeedbackManager()
        
        # Create experiment
        experiment = Experiment(
            name="prompt_optimization",
            description="Testing different prompt structures for article generation",
            metrics=["structure_score", "content_score", "seo_score"]
        )
        
        # Add variants
        for variant_name, prompt_template in PROMPT_VARIANTS.items():
            experiment.add_variant(
                name=variant_name,
                prompt_template=prompt_template
            )
        
        # Run trials for each test case
        for test_case in TEST_CASES:
            print(f"\nGenerating articles for: {test_case['title']}")
            
            for variant_name, prompt_template in PROMPT_VARIANTS.items():
                print(f"\nTrying variant: {variant_name}")
                try:
                    # Generate article using variant's prompt template
                    result = await generator.generate_article(
                        title=test_case["title"],
                        keywords=test_case["keywords"],
                        min_length=1200,
                        max_length=2000,
                        prompt_template=prompt_template
                    )
                    
                    # Record trial
                    experiment.record_trial(
                        variant_name=variant_name,
                        metrics={
                            "structure_score": result["evaluation"]["structure_score"],
                            "content_score": result["evaluation"]["content_score"],
                            "seo_score": result["evaluation"]["seo_score"]
                        },
                        metadata={
                            "title": test_case["title"],
                            "keywords": test_case["keywords"],
                            "content_length": len(result["content"].split())
                        }
                    )
                    
                    print(f"Successfully generated and recorded trial for {variant_name}")
                    
                except Exception as e:
                    print(f"Error generating article for variant {variant_name}: {str(e)}")
                    continue
        
        # Analyze results
        analysis = experiment.analyze_results()
        print("\n=== Experiment Results ===")
        print(f"Total trials: {analysis['total_trials']}")
        print("\nPerformance vs Baseline:")
        for variant, metrics in analysis["variant_performance"].items():
            if variant != "baseline":
                print(f"\n{variant}:")
                for metric, value in metrics.items():
                    print(f"  {metric}: {value:+.2%}")
        
        return analysis
        
    except Exception as e:
        print(f"Error running experiment: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(run_experiment()) 