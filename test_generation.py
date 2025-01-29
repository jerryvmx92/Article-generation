import asyncio
import json
import os
from datetime import datetime
from article_generation.llm.generator import ArticleGenerator

def get_latest_trace(success_dir):
    """Get the most recent trace file."""
    files = os.listdir(success_dir)
    if not files:
        return None
    files.sort(reverse=True)  # Latest file will be first
    return os.path.join(success_dir, files[0])

async def main():
    # Initialize generator
    generator = ArticleGenerator()
    
    # Test article parameters
    title = "Benefits of Regular AC Maintenance in Coatzacoalcos"
    keywords = ["AC maintenance", "air conditioning", "Coatzacoalcos", "humidity control"]
    
    try:
        # Generate and evaluate article
        print("\nGenerating article...")
        result = await generator.generate_article(
            title=title,
            keywords=keywords,
            min_length=800,
            max_length=1500
        )
        
        print("\nArticle generated successfully!")
        print(f"\nTitle: {result['title']}")
        print(f"Keywords: {', '.join(result['keywords'])}")
        print("\nContent Preview (first 200 chars):")
        print(result['content'][:200] + "...")
        
        print("\nEvaluation Results:")
        evaluation = result.get('evaluation', {})
        if not evaluation:
            print("No evaluation results found!")
        elif 'error' in evaluation:
            print(f"Evaluation Error: {evaluation['error']}")
        else:
            print("\nScores:")
            print(f"Structure: {evaluation.get('structure_score', 'N/A')}/10")
            print(f"Content: {evaluation.get('content_score', 'N/A')}/10")
            print(f"SEO: {evaluation.get('seo_score', 'N/A')}/10")
            print(f"Regional: {evaluation.get('regional_score', 'N/A')}/10")
            print(f"Overall: {evaluation.get('overall_score', 'N/A')}/10")
            
            print("\nTop Recommendations:")
            for rec in evaluation.get('top_recommendations', []):
                print(f"- {rec}")
        
        # Check latest trace
        print("\nChecking latest trace...")
        success_dir = os.path.join(os.getcwd(), "traces", "success")
        latest_trace = get_latest_trace(success_dir)
        
        if latest_trace:
            with open(latest_trace, 'r') as f:
                trace = json.load(f)
                if 'evaluation' in trace['response']:
                    print("Evaluation results found in trace!")
                    print(json.dumps(trace['response']['evaluation'], indent=2))
                else:
                    print("No evaluation results in trace!")
        else:
            print("No trace files found!")
        
    except Exception as e:
        print(f"\nError: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main()) 