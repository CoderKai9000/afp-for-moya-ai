# Integration Tests

This directory contains integration tests that demonstrate the Agent Flow Protocol (AFP) in real-world scenarios, including integration with external APIs and example applications.

## Tests Included

- **afp_ai_integration_test.py** - Tests the integration of AFP with Azure OpenAI API, comparing performance metrics between direct API calls and AFP-mediated calls.
- **afp_example_app.py** - A complete example application that showcases AFP in a practical multi-agent communication scenario.

## Running the Tests

```bash
# Run the Azure OpenAI integration test
# Note: Requires Azure OpenAI API credentials in your environment
python tests/integration/afp_ai_integration_test.py

# Run the example application
python tests/integration/afp_example_app.py
```

## Output Files

When executed, these tests generate JSON result files:
- `afp_ai_integration_results.json` - Performance metrics from the Azure OpenAI integration test

## Key Features Demonstrated

These integration tests demonstrate several important aspects of AFP:
- **External API Integration** - How AFP handles communication with external services
- **Multi-Agent Communication** - Real-world communication patterns between multiple agents
- **Error Handling and Resilience** - How AFP handles failures and recovers in production scenarios
- **Performance in Production** - Realistic performance metrics in integrated environments 