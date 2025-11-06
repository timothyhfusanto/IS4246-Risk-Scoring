import json
from LLMTranscriptAnalyzer import LLMTranscriptAnalyzer

rec = json.load(open('outputs/conversations/ambiguity-seeking_validation_openai_gpt-3.5-turbo_20251105_120335.json'))
transcript = [t['content'] for t in rec['conversation'] if t['role']=='assistant']
analyzer = LLMTranscriptAnalyzer(transcript, scenario=rec['scenario_id'], user=rec['user'])
metrics = analyzer.analyze()
print(metrics)