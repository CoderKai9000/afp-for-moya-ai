import os
from openai import AzureOpenAI

# Print the values of the environment variables
print(f"API Key: {os.environ.get('AZURE_OPENAI_API_KEY')}")
print(f"Endpoint: {os.environ.get('AZURE_OPENAI_ENDPOINT')}")
print(f"API Version: {os.environ.get('AZURE_OPENAI_API_VERSION')}")

# Try to create a client
try:
    client = AzureOpenAI(
        api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION"),
    )
    print("Successfully created client")
    
    # List available models
    try:
        models = client.models.list()
        print("\nAvailable models:")
        for model in models:
            print(f"- {model.id}")
    except Exception as e:
        print(f"Error listing models: {str(e)}")
        
    # Test chat completion with a deployment name
    deployment_name = "gpt-4o"  # Replace with your actual deployment name
    try:
        print(f"\nTesting chat completion with deployment: {deployment_name}")
        response = client.chat.completions.create(
            model=deployment_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello, can you hear me?"}
            ],
        )
        print("Response:", response.choices[0].message.content)
    except Exception as e:
        print(f"Error with chat completion: {str(e)}")
        
except Exception as e:
    print(f"Error creating client: {str(e)}") 