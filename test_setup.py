#!/usr/bin/env python3
"""
Simple test script to verify your setup and run a basic simulation
"""

from LLMConversationSimulator import ConversationSimulator
import json
import os
import yaml

def main():
    print("🧪 Testing LLM Conversation Simulator Setup\n")
    
    # Check if API keys are set (from env variables OR config.yaml)
    print("📋 Checking API keys...")
    
    # Try to load from config.yaml
    config_keys = {}
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
            config_keys = config.get('api_keys', {})
    except:
        pass
    
    api_keys = {
        'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY') or config_keys.get('openai', ''),
        'GOOGLE_API_KEY': os.getenv('GOOGLE_API_KEY') or config_keys.get('google', ''),
        'ANTHROPIC_API_KEY': os.getenv('ANTHROPIC_API_KEY') or config_keys.get('anthropic', ''),
        'COHERE_API_KEY': os.getenv('COHERE_API_KEY') or config_keys.get('cohere', ''),
    }
    
    available_providers = []
    for key, value in api_keys.items():
        # Check if key is valid (not empty, not a placeholder)
        if value and not value.startswith('${') and value != f"${{{key}}}":
            provider = key.replace('_API_KEY', '').lower()
            source = "env var" if os.getenv(key) else "config.yaml"
            print(f"  ✅ {key} is set (from {source})")
            available_providers.append(provider)
        else:
            print(f"  ⚠️  {key} not set (optional)")
    
    if not available_providers:
        print("\n❌ No API keys found!")
        print("Please set at least one API key in your .env file")
        print("Run: cp .env.example .env")
        print("Then edit .env and run: source .env")
        return
    
    print(f"\n✅ Found {len(available_providers)} provider(s): {', '.join(available_providers)}\n")
    
    # Initialize simulator
    print("🤖 Initializing simulator...")
    try:
        simulator = ConversationSimulator()
    except Exception as e:
        print(f"❌ Error initializing simulator: {e}")
        return
    
    # Load scenarios
    print("📚 Loading scenarios...")
    scenarios = simulator.scenario_loader.load_all_scenarios()
    
    if not scenarios:
        print("❌ No scenarios found in ./scenarios folder")
        return
    
    print(f"✅ Found {len(scenarios)} scenario(s)")
    for scenario in scenarios[:5]:  # Show first 5
        print(f"   - {scenario.scenario_id}")
    if len(scenarios) > 5:
        print(f"   ... and {len(scenarios) - 5} more")
    
    # Select first scenario for testing
    test_scenario = scenarios[0]
    print(f"\n🎯 Testing with scenario: {test_scenario.scenario_id}")
    print(f"   User: {test_scenario.user}")
    print(f"   Messages: {len(test_scenario.messages)}")
    
    # Determine which model to use
    if 'openai' in available_providers:
        provider = 'openai'
        model = 'gpt-3.5-turbo'
        print(f"\n🚀 Running simulation with OpenAI GPT-3.5-turbo...")
    elif 'google' in available_providers:
        provider = 'google'
        model = 'gemini-pro'
        print(f"\n🚀 Running simulation with Google Gemini Pro...")
    else:
        provider = available_providers[0]
        model = 'default'
        print(f"\n🚀 Running simulation with {provider}...")
    
    # Run simulation
    try:
        result = simulator.simulate_conversation(test_scenario, provider, model)
        print(f"\n✅ Simulation completed successfully!")
        print(f"\n📊 Results:")
        print(f"   Scenario: {result.scenario_id}")
        print(f"   Model: {result.model_name}")
        print(f"   Provider: {result.provider}")
        print(f"   Turns: {len(result.conversation)} messages")
        print(f"\n💬 Sample conversation:")
        for i, turn in enumerate(result.conversation[:4], 1):  # Show first 4 turns
            role = "🧑 User" if turn['role'] == 'user' else "🤖 Assistant"
            content = turn['content'][:100] + "..." if len(turn['content']) > 100 else turn['content']
            print(f"\n   {role}: {content}")
        
        if len(result.conversation) > 4:
            print(f"\n   ... and {len(result.conversation) - 4} more turns")
        
        # Find output file
        try:
            output_files = [f for f in os.listdir('./outputs/conversations') if f.endswith('.json')]
            if output_files:
                latest_file = sorted(output_files)[-1]
                print(f"\n📁 Full conversation saved to: ./outputs/conversations/{latest_file}")
        except:
            pass
        
        print("\n🎉 Setup test completed successfully!")
        print("\n💡 Next steps:")
        print("   1. Check the output file in ./outputs/conversations/")
        print("   2. Run more scenarios with different models")
        print("   3. Analyze results with LLMTranscriptAnalyzer.py")
        
    except Exception as e:
        print(f"\n❌ Error running simulation: {e}")
        print(f"\n💡 Troubleshooting:")
        print(f"   - Make sure your API key is valid")
        print(f"   - Check if you have billing enabled (for paid APIs)")
        print(f"   - Verify config.yaml is properly formatted")


if __name__ == "__main__":
    main()
