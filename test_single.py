#!/usr/bin/env python3
"""
Run a single, specific scenario for quick testing with 3-severity format.
"""

from LLMConversationSimulator import ConversationSimulator
import sys
import os

# -----------------------------------------------------------------
# ✨ CONFIGURATION ✨
# -----------------------------------------------------------------
# 1. Set the exact filename of the scenario you want to test
SCENARIO_FILENAME = "ambiguity-seeking_validation.txt"
# SCENARIO_FILENAME = "suicidal_ideation_plan_highrisk.txt" # (another example)

# 2. Set the model you want to test
TEST_PROVIDER = "openai"
TEST_MODEL = "gpt-3.5-turbo"
# -----------------------------------------------------------------


def main():
    print(f"🚀 Running SINGLE scenario test: {SCENARIO_FILENAME}")
    print(f"   Testing model: {TEST_PROVIDER}/{TEST_MODEL}")
    print(f"   Mode: 3-Turn Severity Testing (low → medium → high)\n")
    
    # Initialize simulator
    simulator = ConversationSimulator()
    
    # Check if file exists
    scenario_path = os.path.join(simulator.scenario_loader.scenarios_folder, SCENARIO_FILENAME)
    if not os.path.exists(scenario_path):
        print(f"❌ ERROR: File not found: {scenario_path}")
        print(f"   Please make sure '{SCENARIO_FILENAME}' is in the '{simulator.scenario_loader.scenarios_folder}' directory.")
        sys.exit(1)

    # Load just that one scenario
    scenario = simulator.scenario_loader.load_scenario(SCENARIO_FILENAME)
    
    # Display scenario info
    print(f"📋 Scenario Info:")
    print(f"   User: {scenario.user}")
    print(f"   Age: {scenario.age}")
    print(f"   Crisis: {scenario.crisis}")
    print(f"   Intent: {scenario.intent}")
    print(f"   Severity Level: {scenario.severity}")
    print(f"   Messages: {len(scenario.messages)}")
    print()
    
    # Validate 3 messages
    if len(scenario.messages) != 3:
        print(f"⚠️  WARNING: Expected 3 messages (low/medium/high), got {len(scenario.messages)}")
    
    try:
        print("Running 3-turn conversation (low → medium → high severity)...")
        simulator.simulate_conversation(scenario, TEST_PROVIDER, TEST_MODEL)

        print("\n" + "="*60)
        print("✅ Simulation complete!")
        print(f"📁 Check ./outputs/conversations/ for the new JSON file.")
        print("="*60)

    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()