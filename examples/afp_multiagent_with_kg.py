import os
import time
import json
from typing import Dict, List, Any, Optional, Callable

# Import the standard AFP components
from moya.agents.azure_openai_agent import AzureOpenAIAgent, AzureOpenAIAgentConfig
from moya.classifiers.llm_classifier import LLMClassifier
from moya.tools.ephemeral_memory import EphemeralMemory
from moya.memory.in_memory_repository import InMemoryRepository
from moya.tools.tool_registry import ToolRegistry

from moya.communication.afp.message import AFPMessage, ContentType
from moya.communication.afp.bus import AFPCommunicationBus

# Import our knowledge graph components
from knowledge_graph import KnowledgeGraphService
from entity_extraction import EntityExtractor

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


# Make sure we're using a valid deployment name
DEPLOYMENT_NAME = "gpt-4o"

# Print environment variables for debugging
print(f"API Key: {os.environ.get('AZURE_OPENAI_API_KEY')[:5]}...")
print(f"Endpoint: {os.environ.get('AZURE_OPENAI_ENDPOINT')}")
print(f"API Version: {os.environ.get('AZURE_OPENAI_API_VERSION')}")
print(f"Using deployment: {DEPLOYMENT_NAME}")


class AFPOrchestratorWithKG:
    """Orchestrator using AFP for multi-agent communication with knowledge graph integration."""
    
    def __init__(self, bus: AFPCommunicationBus, classifier: Any, 
                 knowledge_graph: KnowledgeGraphService, 
                 entity_extractor: EntityExtractor, 
                 default_agent: str = None):
        """
        Initialize an AFP orchestrator with knowledge graph support.
        
        Args:
            bus: The AFP communication bus
            classifier: The classifier wrapper
            knowledge_graph: The knowledge graph service
            entity_extractor: The entity extractor
            default_agent: The default agent to use if classification fails
        """
        self.bus = bus
        self.classifier = classifier
        self.default_agent = default_agent
        self.agents = {}  # Dictionary to store agent wrappers
        self.knowledge_graph = knowledge_graph
        self.entity_extractor = entity_extractor
        
        # Register with the bus
        self.bus.register_agent("orchestrator")
        
        # Subscribe to messages addressed to the orchestrator
        self.subscription_id = self.bus.subscribe(
            subscriber="orchestrator",
            callback=self._handle_message,
            pattern={"recipients": ["orchestrator"]}
        )
    
    def register_agent(self, agent_name: str, agent_wrapper: Any):
        """
        Register an agent with the orchestrator.
        
        Args:
            agent_name: The name of the agent
            agent_wrapper: The agent wrapper instance
        """
        self.agents[agent_name] = agent_wrapper
    
    def _handle_message(self, message: AFPMessage):
        """
        Handle messages sent to the orchestrator.
        
        Args:
            message: The AFP message to handle
        """
        # If it's a user message that needs routing
        if message.metadata.get("needs_routing", False):
            thread_id = message.metadata.get("thread_id", "default_thread")
            user_message = message.content
            
            # Get enriched thread context
            session_summary = EphemeralMemory.get_thread_summary(thread_id)
            
            # Extract main entities from user message
            main_entities = self.entity_extractor.extract_main_entities(user_message, thread_id)
            
            # Get related information from knowledge graph
            graph_context = self.knowledge_graph.get_related_information(main_entities, thread_id)
            
            # Create enriched input with both conversation history and knowledge graph context
            enriched_input = f"{session_summary}\n\n{graph_context}\n\nCurrent user message: {user_message}"
            
            # Classify the message to determine which agent should handle it
            agent_name = self.classifier.classify(enriched_input, thread_id)
            
            if agent_name in self.agents:
                # Route to the appropriate agent
                agent_message = AFPMessage(
                    sender="orchestrator",
                    recipients=[agent_name],
                    content_type=ContentType.TEXT,
                    content=enriched_input,
                    metadata={
                        "thread_id": thread_id,
                        "context": session_summary,
                        "graph_context": graph_context,
                        "original_message_id": message.message_id
                    },
                    parent_message_id=message.message_id
                )
                
                # Send the message to the agent and wait for response
                self.bus.send_message(agent_message)
        
        # Handle other types of messages as needed
        else:
            # For now, just acknowledge receipt
            pass
    
    def orchestrate(self, thread_id: str, user_message: str, stream_callback: Callable = None) -> str:
        """
        Orchestrate a user message to the appropriate agent with knowledge graph integration.
        
        Args:
            thread_id: The conversation thread ID
            user_message: The user message
            stream_callback: Optional callback for streaming responses
            
        Returns:
            The agent's response
        """
        # Start performance timing
        start_time = time.time()
        
        # Store user message for future reference
        EphemeralMemory.store_message(thread_id=thread_id, sender="user", content=user_message)
        
        # Extract main entities from user message
        main_entities = self.entity_extractor.extract_main_entities(user_message, thread_id)
        print(f"Extracted entities: {main_entities}")
        
        # Get related information from knowledge graph
        graph_context = self.knowledge_graph.get_related_information(main_entities, thread_id)
        if graph_context:
            print(f"Knowledge graph context: {graph_context}")
        
        # Get enriched thread context
        session_summary = EphemeralMemory.get_thread_summary(thread_id)
        
        # Create enriched input with both conversation history and knowledge graph context
        if graph_context:
            enriched_input = f"{session_summary}\n\n{graph_context}\n\nCurrent user message: {user_message}"
        else:
            enriched_input = f"{session_summary}\nCurrent user message: {user_message}"
        
        # Classify the message
        agent_name = self.classifier.classify(enriched_input, thread_id)
        
        # Check if we have the agent
        if agent_name not in self.agents:
            if self.default_agent and self.default_agent in self.agents:
                agent_name = self.default_agent
            else:
                # No suitable agent found
                return "I'm sorry, I don't have an agent available to handle your request."
        
        # Send directly to the agent
        agent = self.agents[agent_name]
        
        # Format for agent response 
        agent_prefix = f"[{agent_name}] "
        
        # Use the underlying agent to handle the message
        try:
            # If streaming is requested, use streaming method
            if stream_callback:
                # Send agent prefix first
                stream_callback(agent_prefix)
                
                # Store the full response for later
                full_response = agent_prefix
                
                # For streaming, use the handle_message_stream method
                for chunk in agent.agent.handle_message_stream(
                    enriched_input,  # Using enriched input with graph context
                    thread_id=thread_id
                ):
                    # Send the chunk to the callback
                    stream_callback(chunk)
                    full_response += chunk
            else:
                # For regular responses, use handle_message
                agent_response = agent.agent.handle_message(
                    enriched_input,  # Using enriched input with graph context
                    thread_id=thread_id
                )
                full_response = agent_prefix + agent_response
                
                # Print the response since we're not streaming
                print(full_response, end="", flush=True)
        except Exception as e:
            print(f"Error during agent processing: {str(e)}")
            full_response = f"Error processing your request: {str(e)}"
        
        # Store the assistant's response
        EphemeralMemory.store_message(thread_id=thread_id, sender="assistant", content=full_response)
        
        # Extract entity relationships from the conversation and update knowledge graph
        combined_text = f"User: {user_message}\nAssistant: {full_response}"
        triplets = self.entity_extractor.extract_entities(combined_text, thread_id)
        
        if triplets:
            print(f"Extracted {len(triplets)} relationship triplets")
            self.knowledge_graph.add_triplets(thread_id, triplets)
        
        # End performance timing
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Print performance metrics on a new line after the response
        print(f"\n[Performance] Message processed in {processing_time:.4f} seconds by {agent_name}")
        
        return full_response


def setup_memory_components():
    """Set up memory components for the agents."""
    tool_registry = ToolRegistry()
    EphemeralMemory.configure_memory_tools(tool_registry)
    return tool_registry


def create_english_agent(tool_registry):
    """Create an English-speaking OpenAI agent."""
    agent_config = AzureOpenAIAgentConfig(
        agent_name="english_agent",
        agent_type="ChatAgent",
        description="English language specialist",
        system_prompt="""You are a helpful AI assistant that always responds in English.
        You should be polite, informative, and maintain a professional tone.
        When provided with context from a knowledge graph, use it to enhance your responses
        but only if it's relevant to the current query.""",
        llm_config={
            'temperature': 0.7,
        },
        model_name=DEPLOYMENT_NAME,
        deployment_name=DEPLOYMENT_NAME,
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_base=os.getenv("AZURE_OPENAI_ENDPOINT").rstrip('/'),  # Remove trailing slash if present
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        tool_registry=tool_registry
    )

    return AzureOpenAIAgent(config=agent_config)


def create_spanish_agent(tool_registry) -> AzureOpenAIAgent:
    """Create a Spanish-speaking OpenAI agent."""
    agent_config = AzureOpenAIAgentConfig(
        agent_name="spanish_agent",
        agent_type="ChatAgent",
        description="Spanish language specialist that provides responses only in Spanish",
        system_prompt="""Eres un asistente de IA servicial que siempre responde en español.
        Debes ser educado, informativo y mantener un tono profesional.
        Si te piden hablar en otro idioma, declina cortésmente y continúa en español.
        Cuando se te proporcione contexto de un grafo de conocimiento, utilízalo para mejorar tus respuestas
        pero solo si es relevante para la consulta actual.""",
        llm_config={
            'temperature': 0.7
        },
        model_name=DEPLOYMENT_NAME,
        deployment_name=DEPLOYMENT_NAME,
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_base=os.getenv("AZURE_OPENAI_ENDPOINT").rstrip('/'),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        tool_registry=tool_registry
    )

    return AzureOpenAIAgent(config=agent_config)


def create_joke_agent(tool_registry) -> AzureOpenAIAgent:
    """Create a joke agent for joke-related queries."""
    agent_config = AzureOpenAIAgentConfig(
        agent_name="joke_agent",
        agent_type="ChatAgent",
        description="Agent specialized in telling jokes",
        system_prompt="""You are specialized in telling jokes. Your responses should always 
        include a joke relevant to the user's query. Keep it clean and appropriate.
        When provided with context from a knowledge graph, you can use it to create more
        personalized jokes based on previous interactions.""",
        llm_config={
            'temperature': 0.8
        },
        model_name=DEPLOYMENT_NAME,
        deployment_name=DEPLOYMENT_NAME,
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_base=os.getenv("AZURE_OPENAI_ENDPOINT").rstrip('/'),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        tool_registry=tool_registry
    )

    return AzureOpenAIAgent(config=agent_config)


def create_classifier_agent() -> AzureOpenAIAgent:
    """Create a classifier agent for language and task detection."""
    agent_config = AzureOpenAIAgentConfig(
        agent_name="classifier",
        agent_type="AgentClassifier",
        description="Language and task classifier for routing messages",
        tool_registry=None,
        model_name=DEPLOYMENT_NAME,   
        deployment_name=DEPLOYMENT_NAME,
        system_prompt="""You are a classifier. Your job is to determine the best agent based on the user's message:
        1. If the message requests or implies a need for a joke, return 'joke_agent'
        2. If the message is in English or requests English response, return 'english_agent'
        3. If the message is in Spanish or requests Spanish response, return 'spanish_agent'
        4. For any other language requests, return null
        
        Analyze both the language and intent of the message.
        Return only the agent name as specified above.""",
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_base=os.getenv("AZURE_OPENAI_ENDPOINT").rstrip('/'),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION")
    )

    return AzureOpenAIAgent(config=agent_config)


def setup_afp_orchestrator_with_kg():
    """Set up the AFP-based multi-agent orchestrator with all components including knowledge graph."""
    # Set up shared components
    tool_registry = setup_memory_components()
    afp_bus = AFPCommunicationBus()
    
    # Create agents
    english_agent = create_english_agent(tool_registry)
    spanish_agent = create_spanish_agent(tool_registry)
    joke_agent = create_joke_agent(tool_registry)
    classifier_agent = create_classifier_agent()
    
    # Wrap agents with AFP communication
    from afp_multiagent import AFPAgentWrapper, AFPClassifierWrapper
    english_wrapper = AFPAgentWrapper("english_agent", english_agent, afp_bus)
    spanish_wrapper = AFPAgentWrapper("spanish_agent", spanish_agent, afp_bus)
    joke_wrapper = AFPAgentWrapper("joke_agent", joke_agent, afp_bus)
    
    # Create the classifier wrapper
    classifier_wrapper = AFPClassifierWrapper(classifier_agent, afp_bus, default_agent="english_agent")
    
    # Set up knowledge graph components
    knowledge_graph = KnowledgeGraphService(storage_dir="./graph_storage")
    entity_extractor = EntityExtractor(english_agent)  # Use English agent for entity extraction
    
    # Create the orchestrator with knowledge graph support
    orchestrator = AFPOrchestratorWithKG(
        bus=afp_bus,
        classifier=classifier_wrapper,
        knowledge_graph=knowledge_graph,
        entity_extractor=entity_extractor,
        default_agent="english_agent"
    )
    
    # Register agents with the orchestrator
    orchestrator.register_agent("english_agent", english_wrapper)
    orchestrator.register_agent("spanish_agent", spanish_wrapper)
    orchestrator.register_agent("joke_agent", joke_wrapper)
    
    # Create direct routes for efficiency
    afp_bus.create_direct_route("orchestrator", "english_agent")
    afp_bus.create_direct_route("orchestrator", "spanish_agent")
    afp_bus.create_direct_route("orchestrator", "joke_agent")
    afp_bus.create_direct_route("orchestrator", "classifier")
    
    return orchestrator


def main():
    """Main function to run the AFP-based multi-agent system with knowledge graph."""
    # Set up the orchestrator and all components
    orchestrator = setup_afp_orchestrator_with_kg()
    thread_id = "afp_kg_test_conversation"

    print("\nStarting AFP-based multi-agent chat with Knowledge Graph support (type 'exit' to quit)")
    print("You can chat in English or Spanish, or request responses in either language.")
    print("The system will build a knowledge graph from your conversation to provide better context.")
    print("-" * 60)

    # Initialize conversation history
    EphemeralMemory.store_message(thread_id=thread_id, sender="system", content=f"thread ID: {thread_id}")

    # Start timing for performance metrics
    total_messages = 0
    total_response_time = 0
    
    while True:
        # Get user input
        user_message = input("\nYou: ").strip()

        # Check for exit condition
        if user_message.lower() == 'exit':
            # Print performance summary
            if total_messages > 0:
                avg_response_time = total_response_time / total_messages
                print(f"\nPerformance Summary:")
                print(f"Total messages processed: {total_messages}")
                print(f"Average response time: {avg_response_time:.4f} seconds")
            
            print("\nGoodbye!")
            break

        # Print Assistant prompt
        print("\nAssistant: ", end="", flush=True)
        
        # Start timing for this message
        start_time = time.time()
        
        # Use try-except to catch and display any errors during orchestration
        try:
            # Get response from the orchestrator - not using streaming for simplicity
            response = orchestrator.orchestrate(
                thread_id=thread_id,
                user_message=user_message,
                stream_callback=None  # Not using streaming
            )
            
            # End timing for this message
            end_time = time.time()
            message_time = end_time - start_time
            
            # Update metrics
            total_messages += 1
            total_response_time += message_time
            
        except Exception as e:
            print(f"\nError during orchestration: {str(e)}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main() 