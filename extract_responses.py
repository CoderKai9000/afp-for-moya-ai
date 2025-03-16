import json

# Load the benchmark results
with open('neurosynth_benchmark_results.json', 'r', encoding='utf-8') as f:
    results = json.load(f)

# Extract final responses
standard_response = results['standard_result']['final_response']
kg_response = results['kg_result']['final_response']
direct_response = results['direct_result']['response']

# Print responses with headers
print("=" * 80)
print("STANDARD AFP FINAL RESPONSE")
print("=" * 80)
print(standard_response)
print("\n\n")

print("=" * 80)
print("KG-ENHANCED AFP FINAL RESPONSE")
print("=" * 80)
print(kg_response)
print("\n\n")

print("=" * 80)
print("DIRECT (GROUND TRUTH) RESPONSE")
print("=" * 80)
print(direct_response)

# Also print key analysis metrics
print("\n\n")
print("=" * 80)
print("KEY METRICS")
print("=" * 80)
print(f"Topic Coverage (Standard): {results['analysis']['topic_mentions']['standard']} topics")
print(f"Topic Coverage (KG): {results['analysis']['topic_mentions']['kg']} topics")
print(f"Topic Coverage (Direct): {results['analysis']['topic_mentions']['direct']} topics")
print(f"Average entities per turn (Standard): {results['analysis']['entity_analysis']['standard_avg']}")
print(f"Average entities per turn (KG): {results['analysis']['entity_analysis']['kg_avg']}")
print(f"Response Length (Standard): {results['analysis']['response_lengths']['standard']} chars")
print(f"Response Length (KG): {results['analysis']['response_lengths']['kg']} chars")
print(f"Response Length (Direct): {results['analysis']['response_lengths']['direct']} chars") 