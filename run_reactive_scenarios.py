#!/usr/bin/env python3
"""
Run all scenarios using the REACTIVE ADVERSARIAL mode
"""

from LLMConversationSimulator import ConversationSimulator
import sys

def main():
    print("🚀 Running All Scenarios - REACTIVE ADVERSARIAL MODE\n")
    
    # Initialize simulator
    print("🤖 Initializing simulator...")
    simulator = ConversationSimulator()
    
    # Check if reactive mode is configured
    if not simulator.config.get('reactive_simulation.enabled', False):
        print("❌ ERROR: Reactive simulation is not enabled in config.yaml.")
        print("   Please add the 'reactive_simulation' block to your config.")
        sys.exit(1)
        
    red_team_model = simulator.config.get('reactive_simulation.red_team_model')
    print(f"🔥 Using '{red_team_model}' as the Red Team persona bot.\n")

    # Load all scenarios
    print("📚 Loading scenarios...")
    scenarios = simulator.scenario_loader.load_all_scenarios()
    print(f"✅ Found {len(scenarios)} scenarios\n")
    
    # Define models to test (the "test subjects")
    models_to_test = [
        ('openai', 'gpt-3.5-turbo'),
        # ('google', 'gemini-pro'),
        # Add other models you want to test *against* the red team
    ]
    
    print(f"🎯 Testing {len(models_to_test)} model(s) against reactive personas:")
    for provider, model in models_to_test:
        print(f"   - {provider}/{model}")
    print()
    
    total_sims = len(scenarios) * len(models_to_test)
    print(f"📊 Total simulations to run: {total_sims}\n")
    
    # Confirm before running
    response = input("▶️  Start running? (y/n): ")
    if response.lower() != 'y':
        print("❌ Cancelled")
        return
    
    print("\n" + "="*60)
    print("Starting REACTIVE batch simulation...")
    print("="*60 + "\n")
    
    # Run all scenarios
    results = []
    success_count = 0
    error_count = 0
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n[{i}/{len(scenarios)}] Scenario: {scenario.scenario_id}")
        print(f"   Persona: {scenario.persona[:70]}...") # Show persona
        
        for provider, model in models_to_test:
            try:
                print(f"   → Running with {provider}/{model}...", end=" ")
                # --- THIS IS THE KEY CHANGE ---
                result = simulator.simulate_reactive_conversation(scenario, provider, model)
                # ------------------------------
                results.append(result)
                success_count += 1
                print("✅")
            except Exception as e:
                print(f"❌ Error: {str(e)}")
                error_count += 1
    
    # Summary
    print("\n" + "="*60)
    print("REACTIVE SUMMARY")
    print("="*60)
    print(f"✅ Successful: {success_count}/{total_sims}")
    print(f"❌ Failed: {error_count}/{total_sims}")
    print(f"📁 Outputs saved to: ./outputs/conversations/")
    print("\n🎉 Reactive data collection complete!")

if __name__ == "__main__":
    main()