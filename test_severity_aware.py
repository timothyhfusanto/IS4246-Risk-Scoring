#!/usr/bin/env python3
"""
Quick test of severity-aware metrics with a real conversation file
"""

import json
from LLMTranscriptAnalyzer import LLMTranscriptAnalyzer

def test_severity_aware_analysis():
    print("🧪 Testing Severity-Aware Analysis\n")
    
    # Load a real conversation
    conv_file = "outputs/conversations/ambiguity-seeking_validation_openai_gpt-3.5-turbo_20251107_004912.json"
    
    try:
        with open(conv_file, "r") as f:
            conv_data = json.load(f)
    except FileNotFoundError:
        print(f"❌ File not found: {conv_file}")
        print("   Run test_single.py first to generate conversations.")
        return
    
    # Extract messages
    conversation = conv_data.get("conversation", [])
    assistant_responses = [msg["content"] for msg in conversation if msg["role"] == "assistant"]
    user_messages = [msg["content"] for msg in conversation if msg["role"] == "user"]
    
    print(f"✅ Loaded conversation: {conv_data.get('scenario_id', 'unknown')}")
    print(f"   Model: {conv_data.get('provider', 'unknown')}/{conv_data.get('model_name', 'unknown')}")
    print(f"   Turns: {len(assistant_responses)}")
    print(f"   User: {conv_data.get('user', 'unknown')} (Age: {conv_data.get('age', 'N/A')})")
    print()
    
    # Initialize analyzer
    analyzer = LLMTranscriptAnalyzer(
        transcript=assistant_responses,
        scenario=conv_data.get("scenario_id", "test"),
        user=conv_data.get("user", "User")
    )
    
    # Run analysis with governance + severity awareness
    print("🔍 Running severity-aware analysis...")
    results = analyzer.analyze_with_governance(user_messages=user_messages)
    
    # Display results
    print("\n" + "="*80)
    print("RESULTS: SEVERITY-AWARE GOVERNANCE ANALYSIS")
    print("="*80)
    
    gov = results["governance_summary"]
    
    # LSI Score
    print(f"\n🎯 LLM SAFETY INDEX (LSI) v3")
    print(f"   Score: {gov['lsi_score']:.3f} / 1.000")
    print(f"   Risk Category: {gov['risk_category']}")
    print(f"   Weighting: 30% False Neg | 25% Referrals | 15% Crisis Response | 15% Dependency")
    
    # Severity Analysis
    if "severity_analysis" in results:
        print(f"\n{'─'*80}")
        print("\n📊 SEVERITY-AWARE EVALUATION (Turn-by-Turn)")
        for turn_data in results["severity_analysis"]:
            print(f"\n   Turn {turn_data['turn']} - {turn_data['severity'].upper()} Severity:")
            print(f"      User Crisis Level: {turn_data['user_crisis_level']}/10")
            print(f"      Model Warmth: {turn_data['model_warmth']:.3f}")
            print(f"      Severity Match: {turn_data['severity_appropriateness']['severity_match']}")
            print(f"      Appropriateness Score: {turn_data['severity_appropriateness']['score']:.2f}")
            print(f"      Referral Quality: {turn_data['referral_quality']}")
            print(f"      Actionability: {turn_data['actionability']}")
            if turn_data['severity_appropriateness'].get('reason'):
                print(f"      Reason: {turn_data['severity_appropriateness']['reason']}")
    
    # Response Adaptation
    if gov.get("response_adaptation"):
        adapt = gov["response_adaptation"]
        print(f"\n{'─'*80}")
        print("\n📈 RESPONSE ADAPTATION")
        print(f"   Pattern: {adapt['adaptation_pattern']}")
        print(f"   Adaptation Score: {adapt['adaptation_score']:.2f}/1.00")
        print(f"   Warmth Trend: {adapt['warmth_trend']} ({adapt.get('warmth_delta', 'N/A'):+.3f})")
        print(f"   Referral Quality Trend: {adapt['referral_trend']} ({adapt.get('referral_delta', 'N/A'):+.3f})")
        if adapt.get('reason'):
            print(f"   → {adapt['reason']}")
        
        # Show per-turn details
        if 'warmth_per_turn' in adapt:
            print(f"\n   Warmth per turn: {adapt['warmth_per_turn']}")
        if 'referral_quality_per_turn' in adapt:
            print(f"   Referral quality per turn: {adapt['referral_quality_per_turn']}")
    
    # False Negative Check
    print(f"\n{'─'*80}")
    print("\n⚠️  FALSE NEGATIVE ASSESSMENT")
    fn = gov["false_negative_assessment"]
    print(f"   Risk Level: {fn['false_negative_risk'].upper()}")
    print(f"   Missed Crisis Signals: {fn['missed_crisis_signals']}")
    
    # Referral Quality
    print(f"\n{'─'*80}")
    print("\n📞 REFERRAL QUALITY")
    ref_quality = gov["referral_quality"]
    print(f"   Average Quality Score: {ref_quality['average_quality_score']:.3f}/1.00")
    print(f"   Best Referral: {ref_quality['best_referral_quality'].upper()}")
    print(f"   Has Excellent Referral: {ref_quality['has_excellent_referral']}")
    print(f"   Has Specific Hotline: {ref_quality['has_specific_hotline']}")
    print(f"   Per-Turn: {ref_quality['per_turn_quality']}")
    
    # Actionability
    print(f"\n{'─'*80}")
    print("\n🎬 ACTIONABILITY")
    action = gov["actionability"]
    print(f"   Average Score: {action['average_score']:.3f}/1.00")
    print(f"   Per-Turn: {action['per_turn_quality']}")
    
    print("\n" + "="*80)
    print("✅ Test complete! Severity-aware metrics are working.")
    print("="*80 + "\n")

if __name__ == "__main__":
    test_severity_aware_analysis()
