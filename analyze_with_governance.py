#!/usr/bin/env python3
"""
Enhanced analyzer script with governance metrics
Processes a single conversation JSON and outputs comprehensive governance analysis
"""

import json
import sys
from pathlib import Path
from LLMTranscriptAnalyzer import LLMTranscriptAnalyzer

def extract_user_messages(conversation_data: dict) -> list[str]:
    """Extract user messages from conversation data"""
    # Support both "messages" and "conversation" keys
    messages = conversation_data.get("messages", []) or conversation_data.get("conversation", [])
    return [msg["content"] for msg in messages if msg["role"] == "user"]

def analyze_conversation_with_governance(conversation_path: str):
    """Analyze a conversation with governance metrics"""
    
    # Load conversation JSON
    with open(conversation_path, "r") as f:
        conv_data = json.load(f)
    
    # Extract assistant responses - support both "messages" and "conversation" keys
    messages = conv_data.get("messages", []) or conv_data.get("conversation", [])
    assistant_responses = [msg["content"] for msg in messages if msg["role"] == "assistant"]
    
    # Extract user messages for false negative detection
    user_messages = extract_user_messages(conv_data)
    
    # Get scenario and user info
    scenario = conv_data.get("scenario", "unknown")
    user = conv_data.get("user", "unknown")
    
    # Initialize analyzer
    analyzer = LLMTranscriptAnalyzer(
        transcript=assistant_responses,
        scenario=scenario,
        user=user
    )
    
    # Run enhanced analysis
    results = analyzer.analyze_with_governance(user_messages=user_messages)
    
    return results, conv_data

def print_governance_summary(results: dict, conv_data: dict):
    """Pretty print governance analysis results"""
    gov = results["governance_summary"]
    
    print("\n" + "="*80)
    print("GOVERNANCE ANALYSIS SUMMARY")
    print("="*80)
    print(f"Scenario: {gov['scenario']}")
    print(f"Model: {conv_data.get('provider', 'unknown')}/{conv_data.get('model_name', 'unknown')}") # <-- This is the fix
    print(f"User: {gov['user']}")
    print(f"Total Turns: {gov['total_turns']}")
    
    if conv_data.get('persona'):
        print(f"\nPersona: {conv_data['persona']}")
        
    print(f"\n{'─'*80}")
    
    print("\n🎯 LLM SAFETY INDEX (LSI)")
    print(f"   Score: {gov['lsi_score']:.3f} / 1.000")
    print(f"   Risk Category: {gov['risk_category']}")
    
    print(f"\n{'─'*80}")
    print("\n⚠️  CRITICAL GOVERNANCE METRICS")
    
    # Crisis trajectory
    traj = gov['crisis_trajectory']
    print(f"\n   Crisis Trajectory: {traj['trajectory'].upper()}")
    print(f"      Initial Level: {traj['initial_crisis_level']}/10")
    print(f"      Final Level: {traj['final_crisis_level']}/10")
    print(f"      Change: {traj['crisis_delta']:+.1f} points")
    if 'peak_crisis_level' in traj:
        print(f"      Peak Level: {traj['peak_crisis_level']}/10")
    
    # Time to referral
    referral = gov['time_to_referral']
    referral_quality = gov.get('referral_quality', {})
    
    print(f"\n   Time to Referral:")
    if referral['time_to_first_referral_turn'] > 0:
        print(f"      First Referral: Turn {referral['time_to_first_referral_turn']} ({referral['referral_delay_category']})")
    else:
        print(f"      First Referral: None ({referral['referral_delay_category']})")
    print(f"      Total Referrals: {gov['total_referrals']}")
    
    # Referral Quality Assessment
    print(f"\n   Referral Quality: {referral_quality.get('best_referral_quality', 'unknown').upper()}")
    print(f"      Quality Score: {referral_quality.get('average_quality_score', 0):.3f} / 1.000")
    if referral_quality.get('has_specific_hotline', False):
        print(f"      ✅ Specific crisis hotline number provided")
    else:
        print(f"      ❌ No specific hotline number (vague referral)")
    if referral_quality.get('has_excellent_referral', False):
        print(f"      ✅ At least one excellent referral (specific + urgent)")

    
    # NEW: Referral quality assessment
    if 'referral_quality' in gov:
        ref_qual = gov['referral_quality']
        print(f"\n   Referral Quality:")
        print(f"      Overall Quality: {ref_qual['best_referral_quality'].upper()} (avg score: {ref_qual['average_quality_score']:.3f})")
        if ref_qual['has_specific_hotline']:
            print(f"      ✅ Specific hotline number provided")
        else:
            print(f"      ❌ No specific hotline number (vague suggestions only)")
        if ref_qual['has_excellent_referral']:
            print(f"      ✅ At least one excellent referral detected")
        print(f"      Per-turn quality: {', '.join(ref_qual['per_turn_quality'])}")
    
    # Safety consistency
    print(f"\n   Safety Consistency: {gov['safety_consistency_score']:.3f} / 1.000")
    print(f"      Contradictions Detected: {gov['total_contradictions']}")
    print(f"      Over-Comforting Instances: {gov['total_over_comforting']}")
    
    # False negative assessment
    fn = gov['false_negative_assessment']
    print(f"\n   False Negative Risk: {fn['false_negative_risk'].upper()}")
    if fn.get('missed_crisis_signals', 0) > 0:
        print(f"      Missed Crisis Signals: {fn['missed_crisis_signals']}")
    
    print(f"\n{'─'*80}")
    print("\n📊 DEPENDENCY RISK FACTORS")
    print(f"   Average Warmth Density: {gov['avg_warmth_density']:.3f}")
    print(f"   Reliance Encouragement Count: {gov['reliance_encouragement_count']}")
    print(f"   Continuation Prompts: {gov['continuation_prompts']}")
    print(f"   Personalization Usage: {gov['personalization_usage']}")
    print(f"   Boundary Statements: {gov['total_boundary_statements']}")
    
    print(f"\n{'='*80}\n")

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_with_governance.py <conversation_json_path>")
        print("\nExample:")
        print("python analyze_with_governance.py outputs/conversations/reactive_ambiguity-seeking_validation_openai_gpt-3.5-turbo_20251105_171120.json")
        sys.exit(1)
    
    conversation_path = sys.argv[1]
    
    if not Path(conversation_path).exists():
        print(f"❌ Error: File not found: {conversation_path}")
        sys.exit(1)
    
    print(f"\n🔍 Analyzing conversation: {conversation_path}")
    
    # Run analysis
    results, conv_data = analyze_conversation_with_governance(conversation_path)
    
    # Print governance summary
    print_governance_summary(results, conv_data)
    
    # Optionally save detailed results
    output_path = conversation_path.replace(".json", "_governance_analysis.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"💾 Detailed results saved to: {output_path}")

if __name__ == "__main__":
    main()
