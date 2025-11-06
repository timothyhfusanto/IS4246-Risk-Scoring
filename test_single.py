#!/usr/bin/env python3
"""
Run a single, specific scenario for quick testing.
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

# 3. Set which simulator mode to use
#    True = use simulate_reactive_conversation() (Red Team mode)
#    False = use simulate_conversation() (Static script mode)
USE_REACTIVE_MODE = True
# -----------------------------------------------------------------


def main():
    print(f"🚀 Running SINGLE scenario test: {SCENARIO_FILENAME}")
    print(f"   Testing model: {TEST_PROVIDER}/{TEST_MODEL}")
    print(f"   Mode: {'REACTIVE' if USE_REACTIVE_MODE else 'STATIC'}\n")
    
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
    
    try:
        if USE_REACTIVE_MODE:
            # Check if reactive mode is configured
            if not simulator.config.get('reactive_simulation.enabled', False):
                print("❌ ERROR: Reactive simulation is not enabled in config.yaml.")
                sys.exit(1)
            print("Running in REACTIVE mode...")
            simulator.simulate_reactive_conversation(scenario, TEST_PROVIDER, TEST_MODEL)
        
        else:
            print("Running in STATIC mode...")
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