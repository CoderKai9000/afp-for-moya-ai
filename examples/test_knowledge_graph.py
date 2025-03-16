"""
Test script for Knowledge Graph and Entity Extraction components.
This script tests the basic functionality of the knowledge graph
and entity extraction independently.
"""

import os
import time
from knowledge_graph import KnowledgeGraphService
from entity_extraction import EntityExtractor
from moya.agents.azure_openai_agent import AzureOpenAIAgent, AzureOpenAIAgentConfig
from moya.tools.tool_registry import ToolRegistry

# Read .env file for the environment variables
from dotenv import load_dotenv
from pathlib import Path
env_path = Path.joinpath(Path(__file__).parent, ".env")
load_dotenv(env_path)

# Make sure we're using a valid deployment name
DEPLOYMENT_NAME = "gpt-4o"

# Print environment variables for debugging
print(f"API Key: {os.environ.get('AZURE_OPENAI_API_KEY')[:5]}...")
print(f"Endpoint: {os.environ.get('AZURE_OPENAI_ENDPOINT')}")
print(f"API Version: {os.environ.get('AZURE_OPENAI_API_VERSION')}")
print(f"Using deployment: {DEPLOYMENT_NAME}")


DEPLOYMENT_NAME = "gpt-4o"

def create_test_agent():
    """Create a test agent for entity extraction."""
    agent_config = AzureOpenAIAgentConfig(
        agent_name="test_agent",
        agent_type="ChatAgent",
        description="Test agent for entity extraction",
        system_prompt="You are a helpful assistant.",
        llm_config={
            'temperature': 0.7,
        },
        model_name=DEPLOYMENT_NAME,
        deployment_name=DEPLOYMENT_NAME,
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_base=os.getenv("AZURE_OPENAI_ENDPOINT").rstrip('/'),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        tool_registry=ToolRegistry()
    )

    return AzureOpenAIAgent(config=agent_config)

def test_entity_extraction():
    """Test the entity extraction component."""
    print("\n--- Testing Entity Extraction ---")
    
    # Create a test agent
    agent = create_test_agent()
    extractor = EntityExtractor(agent)
    
    # Test texts
    test_texts = [
        "John visited Paris last summer and enjoyed the Eiffel Tower.",
        "OpenAI developed GPT-4, which is a powerful language model.",
        "The book was written by Mark Twain, who lived in Missouri."
    ]
    
    for i, text in enumerate(test_texts):
        print(f"\nText {i+1}: {text}")
        
        # Test main entity extraction
        print("\nMain Entities:")
        main_entities = extractor.extract_main_entities(text)
        for entity in main_entities:
            print(f"- {entity}")
        
        # Test relationship extraction
        print("\nRelationship Triplets:")
        triplets = extractor.extract_entities(text)
        if triplets:
            for entity1, relation, entity2 in triplets:
                print(f"({entity1}, {relation}, {entity2})")
        else:
            print("No triplets extracted.")
        
        print("-" * 50)

def test_knowledge_graph():
    """Test the knowledge graph component."""
    print("\n--- Testing Knowledge Graph ---")
    
    # Create a knowledge graph service with a test directory
    test_dir = "./test_graph_storage"
    kg = KnowledgeGraphService(storage_dir=test_dir)
    
    # Test thread ID
    thread_id = "test_thread"
    
    # Add some test triplets
    test_triplets = [
        ("john", "visited", "paris"),
        ("paris", "has", "eiffel tower"),
        ("john", "liked", "eiffel tower"),
        ("openai", "developed", "gpt-4"),
        ("gpt-4", "is a", "language model")
    ]
    
    print("\nAdding triplets to the graph:")
    for entity1, relation, entity2 in test_triplets:
        print(f"({entity1}, {relation}, {entity2})")
        kg.add_triplet(thread_id, entity1, relation, entity2)
    
    # Test getting related information
    test_entities = ["john", "paris", "gpt-4"]
    for entity in test_entities:
        print(f"\nRelated information for '{entity}':")
        related_info = kg.get_related_information([entity], thread_id)
        if related_info:
            print(related_info)
        else:
            print("No related information found.")
    
    # Test pruning
    print("\nPruning graph:")
    kg.prune_graph(thread_id, max_nodes=3, age_threshold=0)
    
    # Check what's left after pruning
    print("\nGraph after pruning:")
    for entity in test_entities:
        related_info = kg.get_related_information([entity], thread_id)
        if related_info:
            print(f"Related to '{entity}':")
            print(related_info)
    
    # Clean up test directory
    print("\nCleaning up test directory...")
    import shutil
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    print("Done.")

def test_conversation_flow():
    """Test a simulated conversation flow with knowledge graph integration."""
    print("\n--- Testing Conversation Flow with Knowledge Graph ---")
    
    # Create knowledge graph and entity extractor
    kg = KnowledgeGraphService(storage_dir="./test_flow_storage")
    agent = create_test_agent()
    extractor = EntityExtractor(agent)
    
    # Simulate a conversation
    thread_id = "test_conversation"
    conversation = [
        ("user", "My name is Alex and I work at Microsoft."),
        ("assistant", "Nice to meet you, Alex! It's great to hear that you work at Microsoft. What do you do there?"),
        ("user", "I'm a software engineer working on Azure services."),
        ("assistant", "That's fascinating! Azure is Microsoft's cloud computing service, right? What specific Azure services are you working on?"),
        ("user", "I mainly work on Azure AI and machine learning services."),
        ("assistant", "Working on Azure AI and machine learning services sounds exciting! Those are cutting-edge technologies that are transforming many industries. Do you focus on any particular aspects of AI or machine learning within Azure?")
    ]
    
    print("\nProcessing conversation and building knowledge graph:")
    for speaker, message in conversation:
        print(f"\n{speaker.capitalize()}: {message}")
        
        # For each exchange, extract entities and add to graph
        if speaker == "user" and len(message) > 0:
            # Extract main entities
            main_entities = extractor.extract_main_entities(message)
            print(f"Main entities: {main_entities}")
            
            # Get any relevant context from the graph
            graph_context = kg.get_related_information(main_entities, thread_id)
            if graph_context:
                print(f"Knowledge graph context: {graph_context}")
        
        # After each exchange, extract relationship triplets
        if len(message) > 0:
            triplets = extractor.extract_entities(message)
            
            if triplets:
                print(f"Extracted {len(triplets)} relationship triplets:")
                for entity1, relation, entity2 in triplets:
                    print(f"- ({entity1}, {relation}, {entity2})")
                
                # Add to knowledge graph
                kg.add_triplets(thread_id, triplets)
    
    # After the conversation, query for information about key entities
    key_entities = ["alex", "microsoft", "azure", "ai"]
    print("\nQuerying knowledge graph after the conversation:")
    for entity in key_entities:
        related_info = kg.get_related_information([entity], thread_id)
        if related_info:
            print(f"\nInformation related to '{entity}':")
            print(related_info)
        else:
            print(f"\nNo information found related to '{entity}'")
    
    # Clean up test directory
    print("\nCleaning up test directory...")
    import shutil
    if os.path.exists("./test_flow_storage"):
        shutil.rmtree("./test_flow_storage")
    print("Done.")

if __name__ == "__main__":
    # Test entity extraction
    test_entity_extraction()
    
    # Test knowledge graph
    test_knowledge_graph()
    
    # Test conversation flow
    test_conversation_flow() 