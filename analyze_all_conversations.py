#!/usr/bin/env python3
"""
Batch analyze all conversation files with LLM-based governance metrics
"""
import json
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from LLMTranscriptAnalyzer import LLMTranscriptAnalyzer
from tqdm import tqdm

def extract_user_messages(conversation_data: dict) -> list[str]:
    """Extract user messages from conversation data"""
    messages = conversation_data.get("messages", []) or conversation_data.get("conversation", [])
    return [msg["content"] for msg in messages if msg["role"] == "user"]

def analyze_conversation(conv_path: str) -> dict:
    """Analyze a single conversation and return summary"""
    try:
        with open(conv_path, "r") as f:
            conv_data = json.load(f)
        
        # Extract messages
        messages = conv_data.get("messages", []) or conv_data.get("conversation", [])
        assistant_responses = [msg["content"] for msg in messages if msg["role"] == "assistant"]
        user_messages = extract_user_messages(conv_data)
        
        # Initialize analyzer
        analyzer = LLMTranscriptAnalyzer(
            transcript=assistant_responses,
            scenario=conv_data.get("scenario_id", "unknown"),
            user=conv_data.get("user", "unknown")
        )
        
        # Run analysis
        results = analyzer.analyze_with_governance(user_messages=user_messages)
        
        # Save analysis results back to file
        output_path = conv_path.replace(".json", "_governance_analysis.json")
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        
        # Return summary
        gov = results["governance_summary"]
        return {
            "file": os.path.basename(conv_path),
            "scenario": gov["scenario"],
            "model": f"{conv_data.get('provider', 'unknown')}/{conv_data.get('model_name', 'unknown')}",
            "lsi_score": gov["lsi_score"],
            "risk_category": gov["risk_category"],
            "false_negatives": gov["false_negative_assessment"]["count"],
            "critical_false_negatives": gov["false_negative_assessment"]["critical_count"],
            "analysis_file": output_path,
            "success": True,
            "error": None
        }
    except Exception as e:
        return {
            "file": os.path.basename(conv_path),
            "success": False,
            "error": str(e)
        }

def main():
    print("🔍 Batch Analysis of All Conversations with LLM-Based Metrics\n")
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ ERROR: OPENAI_API_KEY not found")
        print("Please make sure you have a .env file with your API key")
        return
    
    # Find all conversation files (exclude analysis files)
    conv_dir = Path("outputs/conversations")
    conv_files = sorted([
        f for f in conv_dir.glob("*.json") 
        if not f.name.endswith("_governance_analysis.json")
    ])
    
    if not conv_files:
        print("❌ No conversation files found in outputs/conversations/")
        return
    
    print(f"✅ Found {len(conv_files)} conversation files to analyze\n")
    print("⏳ This will take a few minutes (LLM API calls per conversation)...\n")
    
    # Analyze all files
    results = []
    for conv_file in tqdm(conv_files, desc="Analyzing conversations"):
        result = analyze_conversation(str(conv_file))
        results.append(result)
    
    # Separate successes and failures
    successes = [r for r in results if r["success"]]
    failures = [r for r in results if not r["success"]]
    
    # Print summary
    print("\n" + "="*80)
    print("BATCH ANALYSIS SUMMARY")
    print("="*80)
    print(f"✅ Successfully analyzed: {len(successes)}/{len(results)}")
    if failures:
        print(f"❌ Failed: {len(failures)}/{len(results)}")
    
    # Display results sorted by LSI score (lowest = most concerning)
    if successes:
        print("\n📊 RESULTS (sorted by LSI score - lowest = most concerning):\n")
        successes.sort(key=lambda x: x["lsi_score"])
        
        print(f"{'Scenario':<35} {'Model':<25} {'LSI':<6} {'Risk':<15} {'FN':<4} {'Crit':<4}")
        print("─" * 95)
        
        for r in successes:
            scenario_short = r["scenario"][:33] if len(r["scenario"]) > 33 else r["scenario"]
            model_short = r["model"][:23] if len(r["model"]) > 23 else r["model"]
            print(f"{scenario_short:<35} {model_short:<25} {r['lsi_score']:.3f}  {r['risk_category']:<15} {r['false_negatives']:<4} {r['critical_false_negatives']:<4}")
    
    # Show failures if any
    if failures:
        print(f"\n❌ FAILURES:\n")
        for r in failures:
            print(f"  {r['file']}: {r['error']}")
    
    # Save summary to JSON
    summary_file = "outputs/batch_analysis_summary.json"
    with open(summary_file, "w") as f:
        json.dump({
            "total_analyzed": len(results),
            "successful": len(successes),
            "failed": len(failures),
            "results": results
        }, f, indent=2)
    
    print(f"\n📁 Detailed summary saved to: {summary_file}")
    print(f"📁 Individual analysis files saved as: *_governance_analysis.json")
    
    print("\n✅ Batch analysis complete!")
    print("\n💡 Next steps:")
    print("   1. Review individual analysis files in outputs/conversations/")
    print("   2. Compare models by LSI scores")
    print("   3. Investigate conversations with high false negative counts")

if __name__ == "__main__":
    main()
