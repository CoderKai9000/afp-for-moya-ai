#!/usr/bin/env python
"""
Knowledge Graph with AFP Multiagent Example

This example demonstrates how to integrate knowledge graph functionality
with the AFP multiagent orchestrator for context-aware conversations.
"""

import os
import sys
import time
import logging
from typing import Dict, Any, List
import json

# Set up path to include moya modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import Moya modules
from moya.agents.azure_openai_agent import AzureOpenAIAgent
from moya.multiagent.afp_multiagent import AFPOrchestrator
from moya.knowledge_graph import KnowledgeGraphService, EntityExtractor, KnowledgeGraphOrchestrator

# Set up Azure OpenAI API configuration from environment variables
API_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2023-07-01-preview")
DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")

if not API_KEY or not ENDPOINT:
    # Use placeholder values for demonstration if not provided
    logger.warning("Using placeholder API credentials. Set environment variables for actual API access.")
    API_KEY = "dummy_key"
    ENDPOINT = "https://dummy-endpoint.openai.azure.com/"


def create_agents() -> List[AzureOpenAIAgent]:
    """Create a set of agents for the multiagent system."""
    
    # English language agent
    english_agent = AzureOpenAIAgent(
        name="english_agent",
        system_message=(
            "You are a helpful assistant that provides information in English. "
            "You can answer questions on a wide range of topics clearly and accurately."
        ),
        api_key=API_KEY,
        endpoint=ENDPOINT,
        api_version=API_VERSION,
        deployment=DEPLOYMENT
    )
    
    # Spanish language agent
    spanish_agent = AzureOpenAIAgent(
        name="spanish_agent",
        system_message=(
            "Eres un asistente útil que proporciona información en español. "
            "Puedes responder preguntas sobre una amplia gama de temas de manera clara y precisa."
        ),
        api_key=API_KEY,
        endpoint=ENDPOINT,
        api_version=API_VERSION,
        deployment=DEPLOYMENT
    )
    
    # Technical expert agent
    tech_agent = AzureOpenAIAgent(
        name="tech_agent",
        system_message=(
            "You are a technical expert specializing in computer science, programming, "
            "and software development. Provide detailed, accurate technical information."
        ),
        api_key=API_KEY,
        endpoint=ENDPOINT,
        api_version=API_VERSION,
        deployment=DEPLOYMENT
    )
    
    # Classifier agent that determines which other agent to use
    classifier_agent = AzureOpenAIAgent(
        name="classifier",
        system_message=(
            "You are a classifier that determines which agent should handle a query. "
            "For English queries about technology, programming, or computer science, respond with 'tech_agent'. "
            "For queries in Spanish on any topic, respond with 'spanish_agent'. "
            "For all other English queries, respond with 'english_agent'. "
            "Respond ONLY with the agent name, nothing else."
        ),
        api_key=API_KEY,
        endpoint=ENDPOINT,
        api_version=API_VERSION,
        deployment=DEPLOYMENT
    )
    
    return [english_agent, spanish_agent, tech_agent, classifier_agent]


def setup_knowledge_graph() -> tuple:
    """Set up the knowledge graph components."""
    
    # Create knowledge graph service
    kg_service = KnowledgeGraphService(
        storage_dir="./graph_storage",
        max_nodes_per_graph=1000
    )
    
    # Create entity extractor
    entity_extractor = EntityExtractor(
        use_spacy=False,  # Use only rule-based extraction for demo
        extraction_methods=["rule_based"]
    )
    
    # Seed the knowledge graph with some initial knowledge
    thread_id = "demo_session"
    seed_knowledge(kg_service, thread_id)
    
    return kg_service, entity_extractor


def seed_knowledge(kg_service: KnowledgeGraphService, thread_id: str) -> None:
    """Seed the knowledge graph with initial triplets."""
    
    # Technology-related triplets
    tech_triplets = [
        ("Python", "is", "programming language"),
        ("Python", "created by", "Guido van Rossum"),
        ("JavaScript", "runs in", "browser"),
        ("HTML", "used for", "web development"),
        ("CSS", "styles", "web pages"),
        ("Azure", "developed by", "Microsoft"),
        ("AWS", "competitor of", "Azure"),
        ("GPT-4", "developed by", "OpenAI"),
        ("OpenAI", "partnered with", "Microsoft"),
        ("Transformer", "is", "neural network architecture"),
        ("LLM", "stands for", "Large Language Model"),
        ("Python", "has library", "TensorFlow"),
        ("TensorFlow", "used for", "machine learning")
    ]
    
    # General knowledge triplets
    general_triplets = [
        ("Earth", "is", "planet"),
        ("Sun", "is", "star"),
        ("Earth", "orbits", "Sun"),
        ("Spain", "located in", "Europe"),
        ("Spanish", "spoken in", "Spain"),
        ("Spanish", "spoken in", "Mexico"),
        ("English", "official language of", "United States"),
        ("English", "official language of", "United Kingdom")
    ]
    
    # Add all triplets to the graph
    kg_service.add_triplets(thread_id, tech_triplets)
    kg_service.add_triplets(thread_id, general_triplets)
    
    logger.info(f"Added {len(tech_triplets) + len(general_triplets)} seed triplets to the knowledge graph")


def run_conversation_demo() -> None:
    """Run a demonstration of knowledge graph-enhanced conversations."""
    
    logger.info("Setting up agents and knowledge graph...")
    
    # Create agents
    agents = create_agents()
    
    # Create AFP orchestrator
    afp_orchestrator = AFPOrchestrator(
        agents=agents,
        classifier_agent=agents[3],  # classifier agent
        auto_select_agent=True
    )
    
    # Set up knowledge graph
    kg_service, entity_extractor = setup_knowledge_graph()
    
    # Create knowledge graph orchestrator
    kg_orchestrator = KnowledgeGraphOrchestrator(
        orchestrator=afp_orchestrator,
        knowledge_graph_service=kg_service,
        entity_extractor=entity_extractor,
        auto_extract=True,
        context_enrichment=True
    )
    
    # Thread ID for this conversation
    thread_id = "demo_session"
    
    logger.info("Starting knowledge graph-enhanced AFP multiagent chat...")
    print("\n" + "="*80)
    print("Knowledge Graph-Enhanced AFP Multiagent Chat")
    print("Type 'exit' to quit, 'kg info' to see graph statistics.")
    print("="*80 + "\n")
    
    # Run conversation loop
    while True:
        # Get user input
        user_input = input("\nYou: ")
        
        # Check for special commands
        if user_input.lower() == 'exit':
            break
            
        if user_input.lower() == 'kg info':
            display_kg_info(kg_service, thread_id)
            continue
            
        if user_input.lower() == 'export graph':
            graph_data = kg_service.export_graph(thread_id)
            with open("knowledge_graph_export.json", "w") as f:
                json.dump(graph_data, f, indent=2)
            print("\nKnowledge graph exported to knowledge_graph_export.json")
            continue
        
        # Process query
        start_time = time.time()
        response = kg_orchestrator.query(user_input, thread_id=thread_id)
        processing_time = time.time() - start_time
        
        # Display response
        print(f"\nAssistant ({response.get('agent_name', 'unknown')}): {response.get('response', 'No response')}")
        
        # Display knowledge graph metadata if available
        kg_meta = response.get("knowledge_graph", {})
        if kg_meta:
            entities = kg_meta.get("extracted_entities", [])
            triplets = kg_meta.get("extracted_triplets", [])
            
            if entities or triplets:
                print("\n" + "-"*40)
                print("Knowledge Graph Updates:")
                
                if entities:
                    print(f"Entities: {', '.join(entities[:5])}" + (f" + {len(entities) - 5} more" if len(entities) > 5 else ""))
                    
                if triplets:
                    print("New relationships:")
                    for i, (subj, rel, obj) in enumerate(triplets[:3]):
                        print(f"  - {subj} {rel} {obj}")
                    if len(triplets) > 3:
                        print(f"  + {len(triplets) - 3} more...")
                        
                if kg_meta.get("context_enriched", False):
                    print("Context was enriched with knowledge graph information")
                    
                print(f"Processing time: {processing_time:.2f} seconds")
                print("-"*40)


def display_kg_info(kg_service: KnowledgeGraphService, thread_id: str) -> None:
    """Display information about the current state of the knowledge graph."""
    
    # Get graph export for statistics
    graph_data = kg_service.export_graph(thread_id)
    
    # Extract statistics
    node_count = graph_data.get("metadata", {}).get("node_count", 0)
    edge_count = graph_data.get("metadata", {}).get("edge_count", 0)
    
    # Calculate additional statistics
    relation_count = 0
    relation_types = set()
    
    for edge in graph_data.get("edges", []):
        edge_data = edge.get("data", {})
        relations = edge_data.get("relations", {})
        relation_count += sum(rel.get("count", 1) for rel in relations.values())
        relation_types.update(relations.keys())
    
    # Display statistics
    print("\n" + "-"*40)
    print("Knowledge Graph Statistics:")
    print(f"Entities: {node_count}")
    print(f"Relationships: {edge_count}")
    print(f"Total assertions: {relation_count}")
    print(f"Relation types: {len(relation_types)}")
    
    # Show sample of entities if available
    if graph_data.get("nodes"):
        sample_entities = [node.get("id") for node in graph_data.get("nodes", [])[:5]]
        print(f"Sample entities: {', '.join(sample_entities)}" + 
              (f" + {node_count - 5} more" if node_count > 5 else ""))
    
    # Show sample of relation types if available
    if relation_types:
        sample_relations = list(relation_types)[:5]
        print(f"Sample relation types: {', '.join(sample_relations)}" + 
              (f" + {len(relation_types) - 5} more" if len(relation_types) > 5 else ""))
    
    print("-"*40)


if __name__ == "__main__":
    try:
        run_conversation_demo()
    except KeyboardInterrupt:
        print("\nExiting knowledge graph demo...")
    except Exception as e:
        logger.exception(f"Error in knowledge graph demo: {e}")
    finally:
        print("\nThank you for trying the knowledge graph demo!") 