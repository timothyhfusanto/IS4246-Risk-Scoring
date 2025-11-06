#!/usr/bin/env python3
"""
Run all scenarios with 3-severity testing (low → medium → high)
"""

from LLMConversationSimulator import ConversationSimulator
import os
from datetime import datetime

def main():
    print("🚀 Running All Scenarios - 3-Severity Testing (low → medium → high)\n")
    
    # Initialize simulator
    print("🤖 Initializing simulator...")
    simulator = ConversationSimulator()
    
    # Load all scenarios
    print("📚 Loading scenarios...")
    scenarios = simulator.scenario_loader.load_all_scenarios()
    print(f"✅ Found {len(scenarios)} scenarios\n")
    
    # Define models to test
    models_to_test = [
        ('openai', 'gpt-3.5-turbo'),
        # ('openai', 'gpt-4'),  # Uncomment if you want to use GPT-4 (more expensive)
        # ('google', 'gemini-pro'),  # Uncomment if you add Google API key
    ]
    
    print(f"🎯 Testing with {len(models_to_test)} model(s):")
    for provider, model in models_to_test:
        print(f"   - {provider}/{model}")
    print()
    
    # Calculate total simulations (each scenario = 3 turns)
    total_sims = len(scenarios) * len(models_to_test)
    print(f"📊 Total simulations to run: {total_sims}")
    print(f"   ({len(scenarios)} scenarios × {len(models_to_test)} models)")
    print(f"   Each conversation = 3 turns (low → medium → high severity)\n")
    
    # Confirm before running
    response = input("▶️  Start running? (y/n): ")
    if response.lower() != 'y':
        print("❌ Cancelled")
        return
    
    print("\n" + "="*60)
    print("Starting batch simulation...")
    print("="*60 + "\n")
    
    # Run all scenarios
    results = []
    success_count = 0
    error_count = 0
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n[{i}/{len(scenarios)}] Scenario: {scenario.scenario_id}")
        print(f"   User: {scenario.user}, Age: {scenario.age}, Crisis: {scenario.crisis}")
        print(f"   Messages: {len(scenario.messages)} (expecting 3)")
        
        for provider, model in models_to_test:
            try:
                print(f"   → Running with {provider}/{model}...", end=" ")
                result = simulator.simulate_conversation(scenario, provider, model)
                results.append(result)
                success_count += 1
                print("✅")
            except Exception as e:
                print(f"❌ Error: {e}")
                error_count += 1
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"✅ Successful: {success_count}/{total_sims}")
    print(f"❌ Failed: {error_count}/{total_sims}")
    print(f"📁 Outputs saved to: ./outputs/conversations/")
    print()
    
    # Show sample of generated files
    print("📄 Sample conversation files:")
    conv_files = sorted([f for f in os.listdir('./outputs/conversations') if f.endswith('.json')])
    for f in conv_files[-5:]:
        print(f"   - {f}")
    if len(conv_files) > 5:
        print(f"   ... and {len(conv_files) - 5} more")
    
    print("\n🎉 Data collection complete!")
    print("\n💡 Next steps:")
    print("   1. Analyze conversations with: python analyze_with_governance.py <file>")
    print("   2. Check individual files in ./outputs/conversations/")
    print(f"   3. Total conversations collected: {len(conv_files)}")

if __name__ == "__main__":
    main()