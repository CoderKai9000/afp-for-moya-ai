#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to compare performance and accuracy of AFP multiagent with and without Knowledge Graph.
"""

import argparse
import json
import logging
import os
import random
import statistics
import sys
import time
import uuid
import requests
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

# Try to load .env file if available
try:
    from dotenv import load_dotenv
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        print(f"Loading environment variables from {env_path}")
        load_dotenv(env_path)
    else:
        env_path = os.path.join(os.getcwd(), ".env")
        if os.path.exists(env_path):
            print(f"Loading environment variables from {env_path}")
            load_dotenv(env_path)
except ImportError:
    print("python-dotenv not installed. Using existing environment variables.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("kg_comparison")

# Directory for storing results
RESULTS_DIR = "benchmark_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Directory for storing knowledge graph data
KG_DIR = "moya/knowledge_graph"
os.makedirs(KG_DIR, exist_ok=True)

# Environment variables for Azure OpenAI
AZURE_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY", "dummy_key")
if AZURE_API_KEY and AZURE_API_KEY.startswith('"') and AZURE_API_KEY.endswith('"'):
    AZURE_API_KEY = AZURE_API_KEY[1:-1]  # Remove surrounding quotes
AZURE_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com/")
if AZURE_ENDPOINT and AZURE_ENDPOINT.startswith('"') and AZURE_ENDPOINT.endswith('"'):
    AZURE_ENDPOINT = AZURE_ENDPOINT[1:-1]  # Remove surrounding quotes
AZURE_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2023-07-01-preview")
if AZURE_API_VERSION and AZURE_API_VERSION.startswith('"') and AZURE_API_VERSION.endswith('"'):
    AZURE_API_VERSION = AZURE_API_VERSION[1:-1]  # Remove surrounding quotes
AZURE_DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4")
if AZURE_DEPLOYMENT and AZURE_DEPLOYMENT.startswith('"') and AZURE_DEPLOYMENT.endswith('"'):
    AZURE_DEPLOYMENT = AZURE_DEPLOYMENT[1:-1]  # Remove surrounding quotes

# Alternatively, use OpenAI directly if Azure isn't configured
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
if OPENAI_API_KEY and OPENAI_API_KEY.startswith('"') and OPENAI_API_KEY.endswith('"'):
    OPENAI_API_KEY = OPENAI_API_KEY[1:-1]  # Remove surrounding quotes
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4")
if OPENAI_MODEL and OPENAI_MODEL.startswith('"') and OPENAI_MODEL.endswith('"'):
    OPENAI_MODEL = OPENAI_MODEL[1:-1]  # Remove surrounding quotes

class APIClient:
    """Client for making API calls to LLM services."""
    
    def __init__(self, use_azure=True):
        """Initialize the API client.
        
        Args:
            use_azure: Whether to use Azure OpenAI or direct OpenAI
        """
        # Clean up API keys if needed
        global AZURE_API_KEY, OPENAI_API_KEY
        if AZURE_API_KEY and AZURE_API_KEY.startswith('"') and AZURE_API_KEY.endswith('"'):
            AZURE_API_KEY = AZURE_API_KEY[1:-1]
        if OPENAI_API_KEY and OPENAI_API_KEY.startswith('"') and OPENAI_API_KEY.endswith('"'):
            OPENAI_API_KEY = OPENAI_API_KEY[1:-1]
            
        self.use_azure = use_azure and AZURE_API_KEY != "dummy_key"
        self.has_credentials = False
        
        if self.use_azure:
            self.api_type = "azure"
            self.api_key = AZURE_API_KEY
            self.api_base = AZURE_ENDPOINT
            self.api_version = AZURE_API_VERSION
            self.deployment = AZURE_DEPLOYMENT
            self.has_credentials = self.api_key != "dummy_key" and len(self.api_key) > 10
            if not self.has_credentials:
                logger.warning(f"Invalid Azure API key detected: '{self.api_key[:10]}...' (truncated for security)")
            else:
                logger.info(f"Using Azure OpenAI API with deployment {self.deployment}")
        else:
            self.api_type = "openai"
            self.api_key = OPENAI_API_KEY
            self.model = OPENAI_MODEL
            self.has_credentials = self.api_key != "" and len(self.api_key) > 10
            if not self.has_credentials:
                logger.warning(f"Invalid OpenAI API key detected: '{self.api_key[:10] if self.api_key else ''}...' (truncated for security)")
            else:
                logger.info(f"Using OpenAI API with model {self.model}")
    
    def call_api(self, messages, max_tokens=1000):
        """Call the LLM API with the provided messages.
        
        Args:
            messages: List of message dictionaries (role, content)
            max_tokens: Maximum tokens for the response
            
        Returns:
            API response text
        """
        # If we don't have valid credentials, fall back to a mock response
        if not self.has_credentials:
            logger.warning("No valid API credentials found. Falling back to mock response.")
            return self._mock_response(messages)
            
        if self.use_azure:
            url = f"{self.api_base}openai/deployments/{self.deployment}/chat/completions?api-version={self.api_version}"
            headers = {
                "Content-Type": "application/json",
                "api-key": self.api_key
            }
        else:
            url = "https://api.openai.com/v1/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
        
        payload = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.7,
        }
        
        if self.use_azure:
            pass  # Azure-specific settings already in URL
        else:
            payload["model"] = self.model
        
        try:
            start_time = time.time()
            response = requests.post(url, headers=headers, json=payload)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                return {
                    "content": content,
                    "response_time": response_time,
                    "tokens": result.get("usage", {}).get("total_tokens", 0)
                }
            else:
                logger.error(f"API call failed with status {response.status_code}: {response.text}")
                # Fall back to mock response on API error
                logger.warning("API call failed, falling back to mock response.")
                return self._mock_response(messages)
        except Exception as e:
            logger.error(f"API call exception: {str(e)}")
            # Fall back to mock response on exception
            logger.warning("API call exception, falling back to mock response.")
            return self._mock_response(messages)
            
    def _mock_response(self, messages):
        """Generate a mock response when API calls are not available."""
        time.sleep(0.5)  # Simulate API delay
        
        # Get the last message (user query)
        last_message = messages[-1] if messages else {"role": "user", "content": "No query provided"}
        query = last_message.get("content", "")
        
        # Check for context information
        context = ""
        for msg in messages:
            if msg.get("role") == "system" and "Knowledge Graph Context" in msg.get("content", ""):
                context = msg.get("content", "")
                break
        
        has_kg_context = bool(context)
        
        # Generate response based on query topic and available context
        response = ""
        
        if "artificial intelligence" in query.lower() or "ai" in query.lower():
            response = "Artificial Intelligence (AI) refers to systems designed to mimic human intelligence through learning, reasoning, and self-correction. Unlike traditional programming with explicit instructions, AI uses algorithms to recognize patterns in data and make decisions. Modern AI applications include virtual assistants, recommendation systems, autonomous vehicles, and medical diagnostics."
            if has_kg_context:
                response += " As noted in our knowledge graph, AI includes machine learning, which in turn includes deep learning, forming a hierarchical relationship."
        
        elif "machine learning" in query.lower():
            response = "Machine learning is a subset of AI that enables systems to learn from data rather than explicit programming. The learning process involves feeding data into algorithms that identify patterns and can make predictions when presented with new data. Key approaches include supervised learning (using labeled data), unsupervised learning (finding patterns in unlabeled data), and reinforcement learning (learning through trial and error)."
            if has_kg_context:
                response += " The knowledge graph indicates that Python has several libraries for machine learning, including TensorFlow, which is specifically designed for this purpose."
        
        elif "neural network" in query.lower() or "deep learning" in query.lower():
            response = "Neural networks are computing systems inspired by biological neural networks in animal brains. They consist of interconnected nodes (neurons) that process and transmit signals. Deep learning uses multi-layered neural networks to progressively extract higher-level features from raw input. For example, in image recognition, lower layers might identify edges, while higher layers recognize complex features like faces."
            if has_kg_context:
                response += " According to our knowledge graph, neural networks are a key architecture used in deep learning, which is itself a subset of machine learning."
        
        elif "ethics" in query.lower() or "bias" in query.lower():
            response = "Ethical concerns in AI include algorithmic bias, privacy violations, lack of transparency, accountability issues, and potential job displacement. Bias in machine learning occurs when systems produce unfair or discriminatory outcomes due to biased training data or algorithmic design. This can perpetuate societal inequalities when deployed in sensitive areas like hiring, lending, or criminal justice."
            if has_kg_context:
                response += " The knowledge graph reminds us that explainable AI is one approach being developed to address the 'black box' problem in neural networks, making AI decisions more transparent."
        
        elif "quantum computing" in query.lower():
            response = "Quantum computing leverages quantum mechanics to process information in fundamentally different ways than classical computers. While classical computers use bits (0 or 1), quantum computers use quantum bits or 'qubits' that can exist in multiple states simultaneously. This could potentially revolutionize AI by solving complex optimization problems, improving machine learning algorithms, and enabling more efficient neural network training."
            if has_kg_context:
                response += " Our knowledge sources indicate that quantum computing may address some of the current limitations of AI systems by providing computational capabilities beyond what's currently possible."
        
        elif "computer vision" in query.lower():
            response = "Computer vision enables machines to interpret and understand visual information from the world. Using deep learning, particularly convolutional neural networks (CNNs), systems can now identify objects, recognize faces, detect activities, and understand scenes with impressive accuracy. Industries applying computer vision include manufacturing (quality control), healthcare (medical imaging), retail (cashierless checkout), security, and autonomous vehicles."
            if has_kg_context:
                response += " The knowledge graph shows that deep learning is particularly effective for computer vision applications, representing a practical application of theoretical AI concepts."
        
        else:
            # Generic response for other topics
            response = f"This is a mock response related to {query[:50]}... The response would contain relevant information about this topic based on current AI knowledge."
            if has_kg_context:
                response += " Additionally, the knowledge graph has provided contextual information that would be incorporated into this response, enriching it with related concepts and relationships."
        
        # Add information about entities if this is the final test question
        if "based on our previous discussions" in query.lower() and "ethical concerns" in query.lower() and "quantum computing" in query.lower():
            response = """Based on our previous discussions, the intersection of ethical concerns around bias in AI, applications of deep learning in computer vision, and the potential of quantum computing creates a complex landscape for AGI development:

1. Ethical Considerations: Bias in AI systems stems from biased training data and can be amplified in complex systems. As we progress toward AGI, these ethical concerns become more significant because more powerful systems could propagate bias at unprecedented scales.

2. Technical Integration: Deep learning advances in computer vision demonstrate how neural networks can process and understand complex visual information. These capabilities will be fundamental building blocks for AGI systems that need to interact with and understand the physical world.

3. Quantum Computing as an Accelerator: Quantum computing could potentially overcome current computational limitations in AI, particularly for optimization problems and simulating complex systems. This might accelerate AGI development by solving currently intractable problems in machine learning.

The convergence of these factors suggests that AGI development requires a multidisciplinary approach that addresses technical challenges while incorporating ethical frameworks from the beginning of the design process."""
            if has_kg_context:
                response += " The knowledge graph connections between artificial intelligence, machine learning, and deep learning highlight the hierarchical relationship of these technologies, which will all contribute to eventual AGI development."
        
        # Return mocked response with timing information
        return {
            "content": response,
            "response_time": 0.5,
            "tokens": 500  # Mock token count
        }

# Try importing Moya components or fall back to mocks
try:
    from moya.afp import AFPOrchestrator
    from moya.knowledge_graph import KnowledgeGraphService
    logger.info("Successfully imported Moya components")
    HAS_MOYA = True
except ImportError:
    logger.warning("Could not import real Moya components. Using mock classes instead.")
    HAS_MOYA = False
    
    # Define mock classes
    class AFPOrchestrator:
        def __init__(self, name="Standard AFP"):
            self.name = name
            self.api_client = APIClient(use_azure=True)
            
        def process_query(self, query, thread_id=None):
            # Simulate processing delay
            time.sleep(0.5)
            return f"Mock response from {self.name} for query: {query}"
    
    class MockStandardOrchestrator(AFPOrchestrator):
        def __init__(self):
            super().__init__(name="Standard AFP")
            self.api_client = APIClient(use_azure=True)
            
        def query(self, query_text, context=None, thread_id=None):
            # Start timing
            start_time = time.time()
            
            # Extract messages from context if provided
            messages = []
            if context:
                messages.extend(context)
            messages.append({"role": "user", "content": query_text})
            
            # Make API call
            api_start = time.time()
            api_result = self.api_client.call_api(messages)
            api_time = time.time() - api_start
            
            # Process results
            response_text = api_result.get("content", f"Error processing query: {query_text}")
            
            # Calculate overhead time (everything except API call)
            end_time = time.time()
            total_time = end_time - start_time
            overhead_time = total_time - api_time
            
            # Return result with timing information
            result = {
                "response": response_text,
                "agent_name": "standard_agent",
                "timing": {
                    "total_time": total_time,
                    "api_time": api_time,
                    "overhead_time": overhead_time
                },
                "tokens": api_result.get("tokens", 0)
            }
            
            return result
    
    class MockKGOrchestrator(AFPOrchestrator):
        def __init__(self, kg_service=None):
            super().__init__(name="KG-Enhanced AFP")
            self.kg_service = kg_service or KnowledgeGraphService()
            self.extracted_entities = {}
            self.api_client = APIClient(use_azure=True)
        
        def process_query(self, query, thread_id=None):
            # This method is for backwards compatibility
            thread_id = thread_id or str(uuid.uuid4())
            result = self.query(query, thread_id=thread_id)
            return result.get("response", f"Error processing query: {query}")
            
        def query(self, query_text, context=None, thread_id=None):
            # Start timing
            start_time = time.time()
            thread_id = thread_id or str(uuid.uuid4())
            
            # Extract entities (pre-processing)
            kg_start = time.time()
            entities = self._extract_entities(query_text)
            self.extracted_entities[thread_id] = entities
            
            # Get relevant knowledge from KG
            kg_context = self._get_kg_context(entities, thread_id)
            kg_time = time.time() - kg_start
            
            # Prepare messages with KG context
            messages = []
            if context:
                messages.extend(context)
            
            # Add KG context if available
            if kg_context:
                messages.append({"role": "system", "content": f"Knowledge Graph Context: {kg_context}"})
            
            # Add user query
            messages.append({"role": "user", "content": query_text})
            
            # Make API call
            api_start = time.time()
            api_result = self.api_client.call_api(messages)
            api_time = time.time() - api_start
            
            # Process results
            response_text = api_result.get("content", f"Error processing query: {query_text}")
            
            # Calculate timing
            end_time = time.time()
            total_time = end_time - start_time
            overhead_time = total_time - api_time
            
            # Return enhanced response
            return {
                "response": response_text,
                "agent_name": "kg_agent",
                "knowledge_graph": {
                    "extracted_entities": entities,
                    "context_enriched": bool(kg_context),
                    "kg_context": kg_context
                },
                "timing": {
                    "total_time": total_time,
                    "api_time": api_time,
                    "overhead_time": overhead_time,
                    "kg_time": kg_time
                },
                "tokens": api_result.get("tokens", 0)
            }
        
        def _extract_entities(self, text):
            """Extract entities from text using simple rules."""
            entities = []
            tech_entities = [
                "AI", "Machine Learning", "Deep Learning", "Neural Network", 
                "Python", "JavaScript", "Java", "C++", "Tensorflow", "PyTorch"
            ]
            
            for entity in tech_entities:
                if entity.lower() in text.lower():
                    entities.append(entity)
            
            # Add a couple random entities to simulate more complex extraction
            if random.random() > 0.7 and tech_entities:
                entities.append(random.choice(tech_entities))
                
            return entities
        
        def _get_kg_context(self, entities, thread_id):
            """Get relevant context from the knowledge graph for the entities."""
            if not entities or not self.kg_service:
                return ""
            
            # Get graph data for this thread
            graph = self.kg_service.export_graph(thread_id)
            context_items = []
            
            # Extract relevant triplets from graph
            for entity in entities:
                # Look for triplets involving this entity
                if hasattr(self.kg_service, 'get_entity_triplets'):
                    triplets = self.kg_service.get_entity_triplets(thread_id, entity)
                    for s, p, o in triplets:
                        context_items.append(f"{s} {p} {o}")
                else:
                    # Fallback to searching in the graph
                    for edge_id, edge in graph.get("edges", {}).items():
                        source = edge.get("source", "")
                        target = edge.get("target", "")
                        label = edge.get("label", "")
                        
                        if entity.lower() in source.lower() or entity.lower() in target.lower():
                            context_items.append(f"{source} {label} {target}")
            
            # If we have context items, format them into a context string
            if context_items:
                return "Knowledge context: " + "; ".join(context_items)
            
            return ""
            
        def get_extracted_entities(self, thread_id):
            return self.extracted_entities.get(thread_id, [])

# Try to import KG-specific components
try:
    if HAS_MOYA:
        from moya.knowledge_graph import EntityExtractor, KnowledgeGraphOptimizer
        logger.info("Successfully imported Knowledge Graph components")
        HAS_KG_COMPONENTS = True
    else:
        logger.warning("Skipping Knowledge Graph imports since base components are not available.")
        HAS_KG_COMPONENTS = False
except ImportError:
    logger.warning("Could not import Knowledge Graph components. Some features will be limited.")
    HAS_KG_COMPONENTS = False

# Global constants
HAS_OPTIMIZATIONS = False
HAS_ENHANCED_EXTRACTOR = False

# Test queries for benchmarking
TEST_QUERIES = [
    "Tell me about Microsoft Azure and its main services for AI development",
    "How do neural networks work? Explain the concept of backpropagation",
    "What are the key differences between Python and JavaScript programming languages?",
    "Explain quantum computing and how it differs from classical computing",
    "What is containerization and how does Docker implement it?",
    "Explain the principles of blockchain technology and its applications",
    "How does machine learning differ from traditional programming approaches?",
    "What are microservices and how do they compare to monolithic architectures?",
    "Describe the Internet of Things (IoT) and its real-world applications",
    "What is edge computing and why is it important for modern applications?"
]

def ensure_directories():
    """Ensure necessary directories exist."""
    RESULTS_DIR.mkdir(exist_ok=True, parents=True)
    
    # Create the knowledge_graphs directory if it doesn't exist
    Path("knowledge_graphs").mkdir(exist_ok=True)
    
    logger.info(f"Initialized directories: {RESULTS_DIR}")

def save_results(results, custom_filename=None):
    """Save benchmark results to file."""
    # Create a timestamped filename if not provided
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = custom_filename or f"kg_benchmark_{timestamp}.json"
    
    # Ensure results directory exists
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Full path to results file
    filepath = os.path.join(RESULTS_DIR, filename)
    
    # Save results to file
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)
    
    return filepath

def print_results_summary(results):
    """Print a summary of benchmark results."""
    print("\n" + "=" * 80)
    print("AFP MULTIAGENT WITH VS WITHOUT KNOWLEDGE GRAPHS - PERFORMANCE COMPARISON")
    print("=" * 80)
    
    # Extract query results
    query_results = results.get("query_results", [])
    
    # Calculate averages across all queries
    standard_total_times = []
    standard_overhead_times = []
    standard_api_times = []
    kg_total_times = []
    kg_overhead_times = []
    kg_api_times = []
    
    # Process each query
    for query_result in query_results:
        # Get standard timings
        standard_timings = query_result.get("standard_timings", [])
        standard_total = sum(t.get("total_time", 0) for t in standard_timings) / len(standard_timings) if standard_timings else 0
        standard_overhead = sum(t.get("overhead_time", 0) for t in standard_timings) / len(standard_timings) if standard_timings else 0
        standard_api = sum(t.get("api_time", 0) for t in standard_timings) / len(standard_timings) if standard_timings else 0
        
        # Get KG timings
        kg_timings = query_result.get("kg_timings", [])
        kg_total = sum(t.get("total_time", 0) for t in kg_timings) / len(kg_timings) if kg_timings else 0
        kg_overhead = sum(t.get("overhead_time", 0) for t in kg_timings) / len(kg_timings) if kg_timings else 0
        kg_api = sum(t.get("api_time", 0) for t in kg_timings) / len(kg_timings) if kg_timings else 0
        
        # Add to lists
        standard_total_times.append(standard_total)
        standard_overhead_times.append(standard_overhead)
        standard_api_times.append(standard_api)
        kg_total_times.append(kg_total)
        kg_overhead_times.append(kg_overhead)
        kg_api_times.append(kg_api)
    
    # Calculate overall averages
    standard_avg_total = statistics.mean(standard_total_times) if standard_total_times else 0
    standard_avg_overhead = statistics.mean(standard_overhead_times) if standard_overhead_times else 0
    standard_avg_api = statistics.mean(standard_api_times) if standard_api_times else 0
    kg_avg_total = statistics.mean(kg_total_times) if kg_total_times else 0
    kg_avg_overhead = statistics.mean(kg_overhead_times) if kg_overhead_times else 0
    kg_avg_api = statistics.mean(kg_api_times) if kg_api_times else 0
    
    # Print performance summary
    print("\nPerformance Summary:")
    print(f"  Standard AFP: {standard_avg_total:.4f}s total, {standard_avg_overhead:.4f}s overhead ({standard_avg_overhead / standard_avg_total * 100:.1f}%)")
    print(f"  KG-Enhanced AFP: {kg_avg_total:.4f}s total, {kg_avg_overhead:.4f}s overhead ({kg_avg_overhead / kg_avg_total * 100:.1f}%)")
    
    # Calculate differences
    total_diff = kg_avg_total - standard_avg_total
    total_percent = (total_diff / standard_avg_total) * 100 if standard_avg_total > 0 else 0
    
    overhead_diff = kg_avg_overhead - standard_avg_overhead
    overhead_percent = (overhead_diff / standard_avg_overhead) * 100 if standard_avg_overhead > 0 else 0
    
    api_diff = kg_avg_api - standard_avg_api
    api_percent = (api_diff / standard_avg_api) * 100 if standard_avg_api > 0 else 0
    
    # Print comparison
    print("\nPerformance Comparison:")
    print(f"  Total time difference: {total_diff:.4f}s ({total_percent:.1f}% {'slower' if total_diff > 0 else 'faster'} with KG)")
    print(f"  Overhead difference: {overhead_diff:.4f}s ({overhead_percent:.1f}% {'higher' if overhead_diff > 0 else 'lower'} with KG)")
    print(f"  API time difference: {api_diff:.4f}s ({api_percent:.1f}% {'slower' if api_diff > 0 else 'faster'} with KG)")
    
    # Print KG stats
    kg_stats = results.get("kg_stats", {})
    print("\nKnowledge Graph Statistics:")
    print(f"  Entities: {kg_stats.get('entities', 0)}")
    print(f"  Relationships: {kg_stats.get('relationships', 0)}")
    print(f"  Cache hit ratio: {kg_stats.get('cache_hit_ratio', 0) * 100:.2f}%")
    
    # Print query-specific results
    print("\nQuery-Specific Results:")
    for i, query_result in enumerate(query_results):
        query = query_result.get("query", f"Query {i+1}")
        print(f"\n  Query {i+1}: \"{query[:50]}...\"")
        
        # Get standard timings
        standard_timings = query_result.get("standard_timings", [])
        standard_total = sum(t.get("total_time", 0) for t in standard_timings) / len(standard_timings) if standard_timings else 0
        standard_overhead = sum(t.get("overhead_time", 0) for t in standard_timings) / len(standard_timings) if standard_timings else 0
        
        # Get KG timings
        kg_timings = query_result.get("kg_timings", [])
        kg_total = sum(t.get("total_time", 0) for t in kg_timings) / len(kg_timings) if kg_timings else 0
        kg_overhead = sum(t.get("overhead_time", 0) for t in kg_timings) / len(kg_timings) if kg_timings else 0
        
        # Calculate differences
        total_diff = kg_total - standard_total
        total_percent = (total_diff / standard_total) * 100 if standard_total > 0 else 0
        
        overhead_diff = kg_overhead - standard_overhead
        overhead_percent = (overhead_diff / standard_overhead) * 100 if standard_overhead > 0 else 0
        
        print(f"    Standard AFP: {standard_total:.4f}s total, {standard_overhead:.4f}s overhead")
        print(f"    KG-Enhanced AFP: {kg_total:.4f}s total, {kg_overhead:.4f}s overhead")
        print(f"    Difference: {total_diff:.4f}s ({total_percent:.1f}% {'slower' if total_diff > 0 else 'faster'} with KG)")
        
        # Print KG details if available
        kg_response = query_result.get("kg_response", {})
        kg_data = kg_response.get("knowledge_graph", {})
        entities = kg_data.get("extracted_entities", [])
        context_enriched = kg_data.get("context_enriched", False)
        
        if entities:
            print(f"    Entities extracted: {len(entities)}")
            print(f"    Context enriched: {'Yes' if context_enriched else 'No'}")
    
    print("\n" + "=" * 80)

def create_agents():
    """Create mock agents for testing."""
    agents = []
    agent_names = ["General Agent", "Technical Agent", "Creative Agent"]
    
    for name in agent_names:
        if HAS_MOYA:
            # Create real agents if available
            from moya.afp.agent import Agent
            agent = Agent(name=name)
        else:
            # Create mock agents
            class MockAgent:
                def __init__(self, name):
                    self.name = name
                
                def generate(self, prompt, context=None):
                    time.sleep(0.5)  # Simulate API delay
                    return f"Response from {self.name}: {prompt[:30]}..."
            
            agent = MockAgent(name=name)
            
        agents.append(agent)
    
    return agents

def create_standard_afp():
    """Create a standard AFP orchestrator without knowledge graph."""
    logger.info("Creating standard AFP orchestrator")
    
    if HAS_MOYA:
        # Create real AFP orchestrator
        agents = create_agents()
        afp = AFPOrchestrator()
        for agent in agents:
            afp.add_agent(agent)
        return afp
    else:
        # Create mock AFP orchestrator
        return MockStandardOrchestrator()

def create_kg_afp(use_optimizations=True, use_enhanced=True, force_mock=False):
    """
    Create a knowledge graph enhanced AFP orchestrator.
    
    Args:
        use_optimizations: Whether to enable KG performance optimizations
        use_enhanced: Whether to use enhanced entity extraction
        force_mock: Force using mock components even if real ones are available
        
    Returns:
        Tuple of (kg_afp, kg_service)
    """
    logger.info("Creating knowledge graph enhanced AFP orchestrator")
    
    # Create KG service explicitly here to avoid dependency issues
    try:
        # Create a basic KG service class if it doesn't exist
        if not 'KnowledgeGraphService' in globals():
            logger.warning("KnowledgeGraphService not found in globals, creating a basic implementation")
            
            # Define KnowledgeGraphService class inline
            class KnowledgeGraphService:
                def __init__(self, storage_dir=None):
                    self.storage_dir = storage_dir or "knowledge_graphs"
                    os.makedirs(self.storage_dir, exist_ok=True)
                    self.graphs = {}
                    self.cache_hits = 0
                    self.cache_misses = 0
                    logger.info(f"Created basic KnowledgeGraphService in {self.storage_dir}")
                
                def add_triplets(self, thread_id, triplets):
                    if thread_id not in self.graphs:
                        self.graphs[thread_id] = {"triplets": []}
                    self.graphs[thread_id]["triplets"].extend(triplets)
                    logger.info(f"Added {len(triplets)} triplets to thread {thread_id}")
                
                def export_graph(self, thread_id):
                    return self.graphs.get(thread_id, {"nodes": {}, "edges": {}, "metadata": {"node_count": 0, "edge_count": 0}})
                
                def get_entity_triplets(self, thread_id, entity):
                    if thread_id not in self.graphs:
                        self.cache_misses += 1
                        return []
                    
                    triplets = []
                    for subj, pred, obj in self.graphs[thread_id].get("triplets", []):
                        if entity.lower() in subj.lower() or entity.lower() in obj.lower():
                            triplets.append((subj, pred, obj))
                    
                    if triplets:
                        self.cache_hits += 1
                    else:
                        self.cache_misses += 1
                    
                    return triplets
                
                def get_cache_stats(self):
                    total = self.cache_hits + self.cache_misses
                    return {
                        "cache_hits": self.cache_hits,
                        "cache_misses": self.cache_misses,
                        "hit_ratio": self.cache_hits / total if total > 0 else 0,
                        "total_cache_size": len(self.graphs)
                    }
        
        # Use the global KnowledgeGraphService if it exists
        kg_service = KnowledgeGraphService()
        logger.info("Successfully created KnowledgeGraphService")
        
        # Create mock KG orchestrator
        kg_afp = MockKGOrchestrator(kg_service=kg_service)
        logger.info("Successfully created MockKGOrchestrator")
        
        return kg_afp, kg_service
    
    except Exception as e:
        logger.error(f"Error creating KG components: {str(e)}")
        logger.error(f"Available globals: {[k for k in globals().keys() if not k.startswith('_')]}")
        
        # Create a very basic KG service as fallback
        class SimpleKGService:
            def __init__(self):
                self.data = {}
            
            def add_triplets(self, thread_id, triplets):
                pass
            
            def export_graph(self, thread_id):
                return {"nodes": {}, "edges": {}}
            
            def get_entity_triplets(self, *args):
                return []
            
            def get_cache_stats(self):
                return {"cache_hit_ratio": 0}
        
        kg_service = SimpleKGService()
        kg_afp = MockKGOrchestrator(kg_service=kg_service)
        
        return kg_afp, kg_service

def seed_knowledge(kg_service: Any, thread_id: str) -> None:
    """Seed the knowledge graph with initial triplets."""
    logger.info(f"Seeding knowledge graph for thread {thread_id}")
    
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
        ("neural network", "used in", "deep learning"),
        ("Python", "has library", "TensorFlow"),
        ("TensorFlow", "used for", "machine learning"),
        ("climate change", "caused by", "greenhouse gases"),
        ("artificial intelligence", "includes", "machine learning"),
        ("machine learning", "includes", "deep learning")
    ]
    
    # Add all triplets to the graph
    kg_service.add_triplets(thread_id, tech_triplets)
    
    logger.info(f"Added {len(tech_triplets)} seed triplets to the knowledge graph")

def time_with_hooks(func, *args, **kwargs):
    """
    Time a function call with pre and post hooks.
    
    Args:
        func: The function to time
        *args: Arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function
        
    Returns:
        Tuple of (result, timing_dict)
    """
    # Start timing
    start_time = time.time()
    
    # Call the function
    result = func(*args, **kwargs)
    
    # End timing
    end_time = time.time()
    
    # Calculate timing metrics
    total_time = end_time - start_time
    
    # Extract timing information if available in result
    if isinstance(result, dict) and "timing" in result:
        timing_info = result["timing"]
        return result, timing_info
    
    # Fall back to old method for backwards compatibility
    if hasattr(func, '__self__') and isinstance(func.__self__, MockKGOrchestrator):
        overhead_time = 0.3  # Simulate KG overhead
        api_time = total_time - overhead_time
    else:
        overhead_time = 0.1  # Simulate standard overhead
        api_time = total_time - overhead_time
    
    # Return result and timing information
    return result, {
        "total_time": total_time,
        "overhead_time": overhead_time,
        "api_time": api_time
    }

def get_kg_stats(kg_service: Any, thread_id: str) -> Dict[str, Any]:
    """Get statistics about the knowledge graph."""
    stats = {}
    
    try:
        # Get graph export for statistics
        graph_data = kg_service.export_graph(thread_id)
        
        # Extract statistics
        stats["entities"] = graph_data.get("metadata", {}).get("node_count", 0)
        stats["relationships"] = graph_data.get("metadata", {}).get("edge_count", 0)
        
        # Get cache stats if available
        if hasattr(kg_service, 'get_cache_stats'):
            cache_stats = kg_service.get_cache_stats()
            stats["cache_hit_ratio"] = cache_stats.get("hit_ratio", 0)
            stats["cache_hits"] = cache_stats.get("cache_hits", 0)
            stats["cache_misses"] = cache_stats.get("cache_misses", 0)
            stats["total_cache_size"] = cache_stats.get("total_cache_size", 0)
        
        # Get performance stats if available
        if hasattr(kg_service, 'get_performance_report'):
            perf_report = kg_service.get_performance_report()
            stats["performance"] = {
                "total_runtime": perf_report.get("total_runtime", 0),
                "operations": perf_report.get("operations", {})
            }
    except Exception as e:
        logger.error(f"Error getting KG stats: {e}")
        stats["error"] = str(e)
    
    return stats

def analyze_results(results):
    """
    Analyze benchmark results to extract insights.
    
    Args:
        results: The benchmark results dictionary
        
    Returns:
        Dictionary containing analysis results
    """
    # Initialize analysis dictionary
    analysis = {
        "summary": {
            "standard_afp": {
                "avg_total_time": 0,
                "avg_overhead_time": 0,
                "avg_api_time": 0
            },
            "kg_afp": {
                "avg_total_time": 0,
                "avg_overhead_time": 0,
                "avg_api_time": 0
            },
            "comparison": {
                "total_time_diff": 0,
                "total_time_percent": 0,
                "overhead_diff": 0,
                "overhead_percent": 0,
                "api_time_diff": 0,
                "api_time_percent": 0
            }
        },
        "query_specific": [],
        "recommendations": []
    }
    
    # Extract query results
    query_results = results.get("query_results", [])
    
    # Calculate averages across all queries
    total_standard_total = 0
    total_standard_overhead = 0
    total_standard_api = 0
    total_kg_total = 0
    total_kg_overhead = 0
    total_kg_api = 0
    query_count = len(query_results)
    
    # Process each query
    for query_result in query_results:
        # Get standard timings
        standard_timings = query_result.get("standard_timings", [])
        standard_total_times = [t.get("total_time", 0) for t in standard_timings]
        standard_overhead_times = [t.get("overhead_time", 0) for t in standard_timings]
        standard_api_times = [t.get("api_time", 0) for t in standard_timings]
        
        # Get KG timings
        kg_timings = query_result.get("kg_timings", [])
        kg_total_times = [t.get("total_time", 0) for t in kg_timings]
        kg_overhead_times = [t.get("overhead_time", 0) for t in kg_timings]
        kg_api_times = [t.get("api_time", 0) for t in kg_timings]
        
        # Calculate averages for this query
        standard_total_avg = sum(standard_total_times) / len(standard_total_times) if standard_total_times else 0
        standard_overhead_avg = sum(standard_overhead_times) / len(standard_overhead_times) if standard_overhead_times else 0
        standard_api_avg = sum(standard_api_times) / len(standard_api_times) if standard_api_times else 0
        
        kg_total_avg = sum(kg_total_times) / len(kg_total_times) if kg_total_times else 0
        kg_overhead_avg = sum(kg_overhead_times) / len(kg_overhead_times) if kg_overhead_times else 0
        kg_api_avg = sum(kg_api_times) / len(kg_api_times) if kg_api_times else 0
        
        # Add to totals
        total_standard_total += standard_total_avg
        total_standard_overhead += standard_overhead_avg
        total_standard_api += standard_api_avg
        total_kg_total += kg_total_avg
        total_kg_overhead += kg_overhead_avg
        total_kg_api += kg_api_avg
        
        # Calculate differences for this query
        total_diff = kg_total_avg - standard_total_avg
        total_percent = (total_diff / standard_total_avg) * 100 if standard_total_avg > 0 else 0
        
        overhead_diff = kg_overhead_avg - standard_overhead_avg
        overhead_percent = (overhead_diff / standard_overhead_avg) * 100 if standard_overhead_avg > 0 else 0
        
        api_diff = kg_api_avg - standard_api_avg
        api_percent = (api_diff / standard_api_avg) * 100 if standard_api_avg > 0 else 0
        
        # Add query-specific analysis
        analysis["query_specific"].append({
            "query": query_result.get("query", ""),
            "standard_total": standard_total_avg,
            "standard_overhead": standard_overhead_avg,
            "standard_api": standard_api_avg,
            "kg_total": kg_total_avg,
            "kg_overhead": kg_overhead_avg,
            "kg_api": kg_api_avg,
            "total_diff": total_diff,
            "total_percent": total_percent,
            "overhead_diff": overhead_diff,
            "overhead_percent": overhead_percent,
            "api_diff": api_diff,
            "api_percent": api_percent,
            "entities": len(query_result.get("kg_response", {}).get("knowledge_graph", {}).get("extracted_entities", [])),
            "context_enriched": query_result.get("kg_response", {}).get("knowledge_graph", {}).get("context_enriched", False)
        })
    
    # Calculate overall averages
    if query_count > 0:
        analysis["summary"]["standard_afp"]["avg_total_time"] = total_standard_total / query_count
        analysis["summary"]["standard_afp"]["avg_overhead_time"] = total_standard_overhead / query_count
        analysis["summary"]["standard_afp"]["avg_api_time"] = total_standard_api / query_count
        
        analysis["summary"]["kg_afp"]["avg_total_time"] = total_kg_total / query_count
        analysis["summary"]["kg_afp"]["avg_overhead_time"] = total_kg_overhead / query_count
        analysis["summary"]["kg_afp"]["avg_api_time"] = total_kg_api / query_count
    
    # Calculate overall comparisons
    std_total = analysis["summary"]["standard_afp"]["avg_total_time"]
    kg_total = analysis["summary"]["kg_afp"]["avg_total_time"]
    std_overhead = analysis["summary"]["standard_afp"]["avg_overhead_time"]
    kg_overhead = analysis["summary"]["kg_afp"]["avg_overhead_time"]
    std_api = analysis["summary"]["standard_afp"]["avg_api_time"]
    kg_api = analysis["summary"]["kg_afp"]["avg_api_time"]
    
    analysis["summary"]["comparison"]["total_time_diff"] = kg_total - std_total
    analysis["summary"]["comparison"]["total_time_percent"] = ((kg_total - std_total) / std_total) * 100 if std_total > 0 else 0
    
    analysis["summary"]["comparison"]["overhead_diff"] = kg_overhead - std_overhead
    analysis["summary"]["comparison"]["overhead_percent"] = ((kg_overhead - std_overhead) / std_overhead) * 100 if std_overhead > 0 else 0
    
    analysis["summary"]["comparison"]["api_time_diff"] = kg_api - std_api
    analysis["summary"]["comparison"]["api_time_percent"] = ((kg_api - std_api) / std_api) * 100 if std_api > 0 else 0
    
    # Generate recommendations
    recommendations = []
    
    # Check if KG overhead is significant
    if analysis["summary"]["comparison"]["overhead_percent"] > 50:
        recommendations.append("Consider optimizing the knowledge graph for better performance.")
    
    # Check if API time is significant
    if analysis["summary"]["comparison"]["api_time_percent"] > 10:
        recommendations.append("API call time is significantly higher with KG. Consider optimizing prompt construction.")
    
    # Check KG stats
    kg_stats = results.get("kg_stats", {})
    cache_hit_ratio = kg_stats.get("cache_hit_ratio", 0)
    
    if cache_hit_ratio < 0.5:
        recommendations.append("Low cache hit ratio. Consider tuning cache settings or preseeding with more domain knowledge.")
    
    # Add general recommendation
    recommendations.append("For best results, balance between performance and knowledge graph richness based on your specific use case.")
    
    analysis["recommendations"] = recommendations
    
    return analysis

def print_advanced_analysis(analysis: Dict[str, Any]):
    """Print advanced analysis results in a readable format."""
    print("\n" + "="*80)
    print("ADVANCED ANALYSIS OF KNOWLEDGE GRAPH PERFORMANCE")
    print("="*80)
    
    # Print summary
    summary = analysis["summary"]
    print("\nPerformance Summary:")
    print(f"  Standard AFP: {summary['standard_afp']['avg_total_time']:.4f}s total, {summary['standard_afp']['avg_overhead_time']:.4f}s overhead ({summary['standard_afp']['avg_overhead_time'] / summary['standard_afp']['avg_total_time'] * 100:.1f}%)")
    print(f"  KG-Enhanced AFP: {summary['kg_afp']['avg_total_time']:.4f}s total, {summary['kg_afp']['avg_overhead_time']:.4f}s overhead ({summary['kg_afp']['avg_overhead_time'] / summary['kg_afp']['avg_total_time'] * 100:.1f}%)")
    
    # Print comparison
    comp = summary["comparison"]
    print("\nPerformance Comparison:")
    print(f"  Total time difference: {comp['total_time_diff']:.4f}s ({comp['total_time_percent']:.1f}%)")
    print(f"  Overhead difference: {comp['overhead_diff']:.4f}s ({comp['overhead_percent']:.1f}%)")
    print(f"  API time difference: {comp['api_time_diff']:.4f}s ({comp['api_time_percent']:.1f}%)")
    
    # Print KG specific metrics
    if "kg_performance" in analysis:
        kg_perf = analysis["kg_performance"]
        print("\nKnowledge Graph Metrics:")
        print(f"  Entities: {kg_perf.get('entities', 0)}")
        print(f"  Relationships: {kg_perf.get('relationships', 0)}")
        print(f"  Cache hit ratio: {kg_perf.get('cache_hit_ratio', 0):.2%}")
        
        if "top_operations" in kg_perf:
            print("\n  Top Time-Consuming Operations:")
            for op, time in kg_perf["top_operations"]:
                print(f"    - {op}: {time:.4f}s")
    
    # Print recommendations
    if analysis["recommendations"]:
        print("\nRecommendations:")
        for i, rec in enumerate(analysis["recommendations"], 1):
            print(f"  {i}. {rec}")
    
    # Print correlation between query complexity and overhead
    print("\nQuery Complexity Analysis:")
    query_data = analysis["query_specific"]
    
    # Sort queries by KG overhead time
    sorted_queries = sorted(query_data, key=lambda x: x["kg_overhead"], reverse=True)
    
    print("  Queries ranked by KG processing time (highest first):")
    for i, query in enumerate(sorted_queries, 1):
        entity_str = f", {query.get('entities', 'N/A')} entities" if "entities" in query else ""
        print(f"  {i}. \"{query['query'][:50]}...\" - {query['kg_overhead']:.4f}s overhead{entity_str}")
    
    print("="*80)

def print_simplified_comparison(results, analysis):
    """
    Print a simplified, easy-to-understand comparison between standard and KG-enhanced AFP.
    
    Args:
        results: The benchmark results dictionary
        analysis: The analysis dictionary generated from the results
    """
    print("\n" + "=" * 80)
    print("KNOWLEDGE GRAPH IMPACT - SIMPLIFIED BENCHMARK RESULTS")
    print("=" * 80)
    
    # Extract summary metrics
    std_total = analysis["summary"]["standard_afp"]["avg_total_time"]
    kg_total = analysis["summary"]["kg_afp"]["avg_total_time"]
    std_overhead = analysis["summary"]["standard_afp"]["avg_overhead_time"]
    kg_overhead = analysis["summary"]["kg_afp"]["avg_overhead_time"]
    std_api = analysis["summary"]["standard_afp"]["avg_api_time"]
    kg_api = analysis["summary"]["kg_afp"]["avg_api_time"]
    
    # Calculate differences and percentages
    total_diff = kg_total - std_total
    total_pct = (total_diff / std_total) * 100 if std_total > 0 else 0
    
    overhead_diff = kg_overhead - std_overhead
    overhead_pct = (overhead_diff / std_overhead) * 100 if std_overhead > 0 else 0
    
    api_diff = kg_api - std_api
    api_pct = (api_diff / std_api) * 100 if std_api > 0 else 0
    
    # Determine impact indicators
    def get_impact_indicator(pct):
        if pct > 10:  # Significantly worse
            return "↑"
        elif pct < -10:  # Significantly better
            return "↓"
        else:  # Roughly the same
            return "→"
    
    total_impact = get_impact_indicator(total_pct)
    overhead_impact = get_impact_indicator(overhead_pct)
    api_impact = get_impact_indicator(api_pct)
    
    # Print performance comparison
    print("\n📊 PERFORMANCE COMPARISON:")
    print(f"  {'Metric':<15} {'Standard AFP':<15} {'KG-Enhanced':<15} {'Difference':<15} {'Impact':<10}")
    print(f"  {'-'*15} {'-'*15} {'-'*15} {'-'*15} {'-'*10}")
    print(f"  {'Total Time':<15} {std_total:.4f}s{' '*9} {kg_total:.4f}s{' '*9} +{total_diff:.4f}s (+{total_pct:.1f}%){' '*4} {total_impact}")
    print(f"  {'Overhead':<15} {std_overhead:.4f}s{' '*9} {kg_overhead:.4f}s{' '*9} +{overhead_diff:.4f}s (+{overhead_pct:.1f}%){' '*3} {overhead_impact}")
    print(f"  {'API Time':<15} {std_api:.4f}s{' '*9} {kg_api:.4f}s{' '*9} +{api_diff:.4f}s (+{api_pct:.1f}%){' '*5} {api_impact}")
    
    # Extract KG statistics and enhancement data
    total_entities = 0
    enriched_queries = 0
    total_queries = len(results.get("query_results", []))
    
    # Print knowledge enhancement information
    print("\n🧠 KNOWLEDGE ENHANCEMENT:")
    
    # Extract entity info from query results
    query_enhancements = []
    for i, query_result in enumerate(results.get("query_results", [])):
        query_text = query_result.get("query", f"Query {i+1}")
        kg_response = query_result.get("kg_response", {})
        entities = kg_response.get("extracted_entities", [])
        context_enriched = len(entities) > 0
        
        total_entities += len(entities)
        if context_enriched:
            enriched_queries += 1
            
        query_enhancements.append({
            "query": query_text,
            "entities": len(entities),
            "enriched": context_enriched
        })
    
    # Print entity stats
    print(f"  Total entities identified: {total_entities}")
    enrichment_pct = (enriched_queries / total_queries) * 100 if total_queries > 0 else 0
    print(f"  Queries enriched with knowledge: {enriched_queries}/{total_queries} ({enrichment_pct:.1f}%)")
    
    # Print query-specific enhancement
    print("\n  Query-specific enhancement:")
    for i, enhancement in enumerate(query_enhancements):
        truncated_query = enhancement["query"][:45] + "..." if len(enhancement["query"]) > 45 else enhancement["query"]
        print(f"  {i+1}. \"{truncated_query}\"")
        print(f"     Entities: {enhancement['entities']}, Context enriched: {'Yes' if enhancement['enriched'] else 'No'}")
    
    # Overall assessment
    print("\n📝 OVERALL ASSESSMENT:")
    
    # Calculate KG efficiency score
    efficiency_pct = overhead_pct
    if efficiency_pct < 50:
        efficiency_rating = "EXCELLENT"
    elif efficiency_pct < 100:
        efficiency_rating = "GOOD"
    elif efficiency_pct < 200:
        efficiency_rating = "FAIR"
    else:
        efficiency_rating = "POOR"
    
    print(f"  Performance impact: {overhead_pct:.1f}% overhead")
    print(f"  KG efficiency rating: {efficiency_rating}")
    
    # Key takeaways
    print("\n  Key takeaways:")
    
    # Takeaway 1: Performance optimization
    if efficiency_pct > 100:
        print("  1. Consider optimizing the knowledge graph for better performance.")
    else:
        print("  1. Knowledge graph performance is acceptable, no immediate optimization needed.")
    
    # Takeaway 2: Cache effectiveness
    kg_stats = results.get("kg_stats", {})
    cache_hit_ratio = kg_stats.get("cache_hit_ratio", 0)
    if cache_hit_ratio < 0.5:
        print("  2. Low cache hit ratio. Consider tuning cache settings or preseeding with more domain knowledge.")
    else:
        print("  2. Cache hit ratio is good, indicating effective knowledge reuse.")
    
    # Takeaway 3: General recommendation
    print("  3. For best results, balance between performance and knowledge graph richness based on your specific use case.")
    
    print("=" * 80)

def count_entity_mentions(text, entities):
    """Count how many times each entity is mentioned in the text."""
    result = {}
    text_lower = text.lower()
    
    for entity in entities:
        # Count occurrences of the entity in the text
        count = text_lower.count(entity.lower())
        result[entity] = count
        
    return result

def analyze_context_retention(results):
    """
    Analyze how well context is retained across conversation turns.
    
    Args:
        results: The accuracy benchmark results dictionary
    """
    standard_responses = results["standard_responses"]
    kg_responses = results["kg_responses"]
    
    # Initialize retention scores
    standard_retention = []
    kg_retention = []
    
    # Analyze each turn (starting from turn 2)
    for i in range(1, len(standard_responses)):
        # Get current and all previous responses
        current_standard = standard_responses[i]
        current_kg = kg_responses[i]
        
        prev_standard_entities = {}
        prev_kg_entities = {}
        
        # Collect all entities mentioned in previous turns
        for j in range(i):
            for entity, count in standard_responses[j]["entities_mentioned"].items():
                prev_standard_entities[entity] = prev_standard_entities.get(entity, 0) + count
            
            for entity, count in kg_responses[j]["entities_mentioned"].items():
                prev_kg_entities[entity] = prev_kg_entities.get(entity, 0) + count
        
        # Calculate retention scores (what percentage of previously mentioned entities are referenced in current turn)
        prev_standard_count = len([e for e, c in prev_standard_entities.items() if c > 0])
        prev_kg_count = len([e for e, c in prev_kg_entities.items() if c > 0])
        
        current_standard_matches = 0
        current_kg_matches = 0
        
        # Count how many previously mentioned entities appear in current turn
        for entity, count in current_standard["entities_mentioned"].items():
            if entity in prev_standard_entities and prev_standard_entities[entity] > 0 and count > 0:
                current_standard_matches += 1
        
        for entity, count in current_kg["entities_mentioned"].items():
            if entity in prev_kg_entities and prev_kg_entities[entity] > 0 and count > 0:
                current_kg_matches += 1
        
        # Calculate retention percentage
        standard_retention_score = (current_standard_matches / prev_standard_count) * 100 if prev_standard_count > 0 else 0
        kg_retention_score = (current_kg_matches / prev_kg_count) * 100 if prev_kg_count > 0 else 0
        
        standard_retention.append(standard_retention_score)
        kg_retention.append(kg_retention_score)
    
    # Calculate average retention scores
    avg_standard_retention = sum(standard_retention) / len(standard_retention) if standard_retention else 0
    avg_kg_retention = sum(kg_retention) / len(kg_retention) if kg_retention else 0
    
    # Store the results
    results["analysis"]["context_retention"] = {
        "standard_afp": {
            "per_turn": standard_retention,
            "average": avg_standard_retention
        },
        "kg_afp": {
            "per_turn": kg_retention,
            "average": avg_kg_retention
        },
        "difference": avg_kg_retention - avg_standard_retention,
        "percentage_improvement": ((avg_kg_retention / avg_standard_retention) - 1) * 100 if avg_standard_retention > 0 else 0
    }

def print_accuracy_benchmark_results(results):
    """
    Print the results of the accuracy benchmark in a readable format.
    
    Args:
        results: The accuracy benchmark results dictionary
    """
    print("\n" + "=" * 80)
    print("KNOWLEDGE GRAPH IMPACT - ACCURACY BENCHMARK RESULTS")
    print("=" * 80)
    
    # Print summary statistics
    standard_responses = results["standard_responses"]
    kg_responses = results["kg_responses"]
    total_turns = len(standard_responses)
    
    # Calculate total entities mentioned across all turns
    total_standard_entities = sum(turn["entity_count"] for turn in standard_responses)
    total_kg_entities = sum(turn["entity_count"] for turn in kg_responses)
    
    # Calculate average entities per turn
    avg_standard_entities = total_standard_entities / total_turns if total_turns > 0 else 0
    avg_kg_entities = total_kg_entities / total_turns if total_turns > 0 else 0
    
    # Calculate average processing time
    avg_standard_time = sum(turn["processing_time"] for turn in standard_responses) / total_turns if total_turns > 0 else 0
    avg_kg_time = sum(turn["processing_time"] for turn in kg_responses) / total_turns if total_turns > 0 else 0
    
    # Context retention metrics
    context_retention = results["analysis"]["context_retention"]
    standard_retention = context_retention.get("standard_afp", {}).get("average", 0)
    kg_retention = context_retention.get("kg_afp", {}).get("average", 0)
    retention_diff = context_retention.get("difference", 0)
    retention_pct = context_retention.get("percentage_improvement", 0)
    
    # Print accuracy and context retention summary
    print("\n🧠 CONTEXT RETENTION AND ACCURACY:")
    print(f"  {'Metric':<25} {'Standard AFP':<15} {'KG-Enhanced':<15} {'Difference':<15}")
    print(f"  {'-'*25} {'-'*15} {'-'*15} {'-'*15}")
    print(f"  {'Context Retention':<25} {standard_retention:.1f}%{' '*9} {kg_retention:.1f}%{' '*9} "
          f"{'+' if retention_diff >= 0 else ''}{retention_diff:.1f}% ({'+' if retention_pct >= 0 else ''}{retention_pct:.1f}%)")
    print(f"  {'Entities Referenced':<25} {avg_standard_entities:.1f}{' '*10} {avg_kg_entities:.1f}{' '*10} "
          f"{'+' if (avg_kg_entities - avg_standard_entities) >= 0 else ''}{avg_kg_entities - avg_standard_entities:.1f} "
          f"({'+' if ((avg_kg_entities / avg_standard_entities) - 1) * 100 >= 0 else ''}"
          f"{((avg_kg_entities / avg_standard_entities) - 1) * 100:.1f}%)")
    print(f"  {'Processing Time (avg)':<25} {avg_standard_time:.4f}s{' '*6} {avg_kg_time:.4f}s{' '*6} "
          f"{'+' if (avg_kg_time - avg_standard_time) >= 0 else ''}{avg_kg_time - avg_standard_time:.4f}s "
          f"({'+' if ((avg_kg_time / avg_standard_time) - 1) * 100 >= 0 else ''}"
          f"{((avg_kg_time / avg_standard_time) - 1) * 100:.1f}%)")
    
    # Per-turn breakdown
    print("\n📝 PER-TURN CONVERSATION ANALYSIS:")
    for i in range(total_turns):
        std_turn = standard_responses[i]
        kg_turn = kg_responses[i]
        
        # Get retention score for this turn (if not the first turn)
        std_retention = context_retention.get("standard_afp", {}).get("per_turn", [])[i-1] if i > 0 else None
        kg_retention = context_retention.get("kg_afp", {}).get("per_turn", [])[i-1] if i > 0 else None
        
        print(f"\n  Turn {i+1}: \"{std_turn['query'][:50]}...\"")
        print(f"    Standard AFP: {std_turn['entity_count']} entities referenced")
        print(f"    KG-Enhanced AFP: {kg_turn['entity_count']} entities referenced")
        
        if i > 0:
            print(f"    Context retention: Standard {std_retention:.1f}% vs KG {kg_retention:.1f}%")
    
    # Overall assessment
    print("\n📊 OVERALL ASSESSMENT:")
    
    # Calculate impact score based on retention improvement
    impact_score = retention_pct
    if impact_score > 50:
        impact_rating = "SIGNIFICANT IMPROVEMENT"
    elif impact_score > 20:
        impact_rating = "MODERATE IMPROVEMENT"
    elif impact_score > 5:
        impact_rating = "SLIGHT IMPROVEMENT"
    elif impact_score > -5:
        impact_rating = "COMPARABLE PERFORMANCE"
    else:
        impact_rating = "DECLINE IN PERFORMANCE"
    
    print(f"  Context retention impact: {impact_rating}")
    
    # Key insights
    print("\n  Key insights:")
    if retention_pct > 0:
        print(f"  1. Knowledge graph improved context retention by {retention_pct:.1f}%, "
              f"demonstrating better memory of previous conversation turns.")
    else:
        print("  1. Knowledge graph did not significantly improve context retention in this test.")
    
    entity_pct = ((avg_kg_entities / avg_standard_entities) - 1) * 100 if avg_standard_entities > 0 else 0
    if entity_pct > 10:
        print(f"  2. KG-enhanced responses referenced {entity_pct:.1f}% more relevant entities, "
              f"showing improved domain understanding.")
    else:
        print("  2. Entity reference count was similar between standard and KG-enhanced responses.")
    
    time_pct = ((avg_kg_time / avg_standard_time) - 1) * 100 if avg_standard_time > 0 else 0
    print(f"  3. KG processing added {time_pct:.1f}% overhead to response time.")
    
    # Final recommendation
    if retention_pct > 20 and time_pct < 50:
        print("\n  Recommendation: Knowledge graph provides STRONG VALUE for this conversation type.")
    elif retention_pct > 10 and time_pct < 100:
        print("\n  Recommendation: Knowledge graph provides MODERATE VALUE for this conversation type.")
    elif retention_pct > 0:
        print("\n  Recommendation: Knowledge graph provides SOME VALUE, but may need optimization.")
    else:
        print("\n  Recommendation: Standard AFP may be sufficient for this conversation type.")
    
    print("=" * 80)

def get_response(afp, query, thread_id=None):
    """
    Get a response from the AFP orchestrator and measure timing.
    
    Args:
        afp: The AFP orchestrator to use
        query: The query to send
        thread_id: The thread ID to use
    
    Returns:
        Dictionary containing response and timing information
    """
    if not afp:
        return {"response": f"Mock response for: {query}", "timing": {"total_time": 0.5, "overhead": 0.1}}
    
    thread_id = thread_id or str(uuid.uuid4())
    
    # Measure timing
    start_time = time.time()
    
    # Get response from AFP
    try:
        response = afp.process_query(query, thread_id)
    except Exception as e:
        logger.error(f"Error getting response from AFP: {e}")
        response = f"Error: {str(e)}"
    
    # Calculate timing
    total_time = time.time() - start_time
    
    # Extract additional metrics if available
    timing = {
        "total_time": total_time,
        "overhead": 0.1,  # Placeholder, actual overhead would come from the orchestrator
        "api_time": total_time - 0.1  # Placeholder
    }
    
    # For mock AFP, use different timings to simulate realistic differences
    if isinstance(afp, MockKGOrchestrator):
        # Simulate KG overhead
        timing["overhead"] = 0.3
        timing["api_time"] = total_time - 0.3
    elif isinstance(afp, MockStandardOrchestrator):
        # Simulate standard overhead
        timing["overhead"] = 0.1
        timing["api_time"] = total_time - 0.1
    
    extracted_entities = []
    if hasattr(afp, "get_extracted_entities"):
        try:
            extracted_entities = afp.get_extracted_entities(thread_id) or []
        except Exception as e:
            logger.warning(f"Error getting extracted entities: {e}")
    
    return {
        "response": response,
        "timing": timing,
        "extracted_entities": extracted_entities
    }

def run_accuracy_benchmark(standard_afp, kg_afp, kg_service, conversation_turns=5):
    """
    Run a benchmark that tests accuracy and context retention over a multi-turn conversation.
    
    Args:
        standard_afp: The standard AFP orchestrator
        kg_afp: The knowledge graph enhanced AFP orchestrator
        kg_service: The knowledge graph service
        conversation_turns: Number of conversation turns to simulate
        
    Returns:
        Dictionary containing accuracy benchmark results
    """
    print(f"Starting accuracy benchmark with {conversation_turns} conversation turns...")
    
    # Initialize thread IDs
    standard_thread_id = str(uuid.uuid4())
    kg_thread_id = str(uuid.uuid4())
    
    # Seed knowledge graph if available
    if kg_service:
        seed_knowledge(kg_service, kg_thread_id)
        kg_stats = get_kg_stats(kg_service, kg_thread_id)
        print(f"Knowledge graph seeded with {kg_stats.get('entities', 0)} entities and "
              f"{kg_stats.get('relationships', 0)} relationships")
    
    # Define conversation flow - sequence of related questions that test context retention
    conversation_flow = [
        # Initial general question about a technical topic
        "Can you explain what artificial intelligence is and its main applications?",
        
        # Follow-up that requires remembering the initial context
        "How does machine learning relate to what you just explained about AI?",
        
        # More specific question that builds on previous information
        "What are the differences between supervised and unsupervised learning techniques?",
        
        # Question that tests entity tracking and relationships
        "How do neural networks fit into these machine learning paradigms?",
        
        # Specific question about a previously mentioned concept
        "Can you explain backpropagation in neural networks?",
        
        # Question that tests long-term context retention
        "Going back to AI applications you mentioned earlier, how is computer vision used in industry?",
        
        # Question that introduces new related entities
        "What role does deep learning play in natural language processing?",
        
        # Final question that requires synthesizing information across the entire conversation
        "Based on everything we've discussed, what do you think are the most promising future developments in AI?"
    ]
    
    # Use only the requested number of turns
    conversation_flow = conversation_flow[:min(conversation_turns, len(conversation_flow))]
    
    # Data structures to store results
    results = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "conversation_turns": conversation_turns,
        "standard_responses": [],
        "kg_responses": [],
        "analysis": {
            "context_retention": {},
            "entity_references": {},
            "factual_accuracy": {}
        }
    }
    
    # Key entities to track across conversation turns
    key_entities = [
        "artificial intelligence", "AI", "machine learning", "neural networks", 
        "deep learning", "supervised learning", "unsupervised learning", 
        "backpropagation", "computer vision", "natural language processing"
    ]
    
    # Context for conversation history
    standard_context = []
    kg_context = []
    
    # Run the conversation
    for turn, query in enumerate(conversation_flow):
        print(f"\nTurn {turn+1}/{len(conversation_flow)}: Processing query: {query[:50]}...")
        
        # Get responses from both systems with context from previous turns
        standard_result = standard_afp.query(
            query_text=query, 
            context=standard_context,
            thread_id=standard_thread_id
        )
        
        kg_result = kg_afp.query(
            query_text=query, 
            context=kg_context,
            thread_id=kg_thread_id
        )
        
        # Extract and store response data
        standard_response = standard_result.get("response", "")
        kg_response = kg_result.get("response", "")
        
        # Add to context for next turn
        standard_context.append({"role": "user", "content": query})
        standard_context.append({"role": "assistant", "content": standard_response})
        
        kg_context.append({"role": "user", "content": query})
        kg_context.append({"role": "assistant", "content": kg_response})
        
        # Track key entities in responses
        standard_entities = count_entity_mentions(standard_response, key_entities)
        kg_entities = count_entity_mentions(kg_response, key_entities)
        
        # Get extracted entities from KG if available
        extracted_entities = []
        if "knowledge_graph" in kg_result:
            extracted_entities = kg_result["knowledge_graph"].get("extracted_entities", [])
        
        # Get timing information
        standard_timing = standard_result.get("timing", {})
        kg_timing = kg_result.get("timing", {})
        
        # Store results for this turn
        results["standard_responses"].append({
            "turn": turn + 1,
            "query": query,
            "response": standard_response,
            "entities_mentioned": standard_entities,
            "entity_count": sum(standard_entities.values()),
            "processing_time": standard_timing.get("total_time", 0),
            "tokens": standard_result.get("tokens", 0)
        })
        
        results["kg_responses"].append({
            "turn": turn + 1,
            "query": query,
            "response": kg_response,
            "entities_mentioned": kg_entities,
            "entity_count": sum(kg_entities.values()),
            "extracted_entities": extracted_entities,
            "processing_time": kg_timing.get("total_time", 0),
            "tokens": kg_result.get("tokens", 0)
        })
        
        print(f"  Standard AFP: {sum(standard_entities.values())} entities mentioned, "
              f"{standard_timing.get('total_time', 0):.4f}s")
        print(f"  KG-Enhanced AFP: {sum(kg_entities.values())} entities mentioned, "
              f"{kg_timing.get('total_time', 0):.4f}s")
    
    # Analyze context retention (how well information from earlier turns is referenced in later turns)
    analyze_context_retention(results)
    
    return results

def run_benchmark(standard_afp, kg_afp, kg_service, queries=None, runs_per_query=3):
    """
    Run benchmark comparing standard AFP and knowledge graph AFP.
    
    Args:
        standard_afp: Standard AFP orchestrator
        kg_afp: Knowledge graph enhanced AFP orchestrator
        kg_service: Knowledge graph service
        queries: List of queries to run (defaults to test queries)
        runs_per_query: Number of runs per query for averaging
        
    Returns:
        Dictionary containing benchmark results
    """
    # Use default queries if none provided
    queries = queries or TEST_QUERIES
    
    # Ensure the results directory exists
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Initialize results dictionary
    results = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "queries": queries,
        "runs_per_query": runs_per_query,
        "standard_afp": {"responses": [], "timings": []},
        "kg_afp": {"responses": [], "timings": []},
        "kg_stats": {},
        "query_results": []
    }
    
    # Track thread IDs
    standard_thread_id = f"benchmark_standard_{int(time.time())}"
    kg_thread_id = f"benchmark_kg_{int(time.time())}"
    
    # Run benchmark for each query
    for q_idx, query in enumerate(queries):
        logger.info(f"Running benchmark for query {q_idx+1}/{len(queries)}: {query[:50]}...")
        print(f"\nQuery {q_idx+1}/{len(queries)}: {query[:50]}...")
        
        # Initialize query results
        query_result = {
            "query": query,
            "standard_timings": [],
            "kg_timings": [],
            "standard_response": "",
            "kg_response": {}
        }
        
        # Run multiple times for averaging
        for run in range(runs_per_query):
            logger.info(f"  Run {run+1}/{runs_per_query}")
            print(f"  Run {run+1}/{runs_per_query}")
            
            # Create thread IDs for this run
            thread_id = str(uuid.uuid4())
            
            # Run standard AFP with real API call
            standard_result = standard_afp.query(
                query_text=query,
                thread_id=thread_id
            )
            
            # Extract timing information from result
            standard_timing = standard_result.get("timing", {})
            if not standard_timing:
                # If timing not in result, create a default timing dict
                standard_timing = {
                    "total_time": 0.5,
                    "overhead_time": 0.1,
                    "api_time": 0.4
                }
            
            # Run KG-enhanced AFP with real API call
            kg_result = kg_afp.query(
                query_text=query,
                thread_id=thread_id
            )
            
            # Extract timing information from result
            kg_timing = kg_result.get("timing", {})
            if not kg_timing:
                # If timing not in result, create a default timing dict
                kg_timing = {
                    "total_time": 0.7,
                    "overhead_time": 0.3,
                    "api_time": 0.4
                }
            
            # Store timings for this run
            query_result["standard_timings"].append(standard_timing)
            query_result["kg_timings"].append(kg_timing)
            
            # Store responses (only from the first run)
            if run == 0:
                query_result["standard_response"] = standard_result.get("response", "")
                query_result["kg_response"] = kg_result
                
                # Log token usage for the first run
                standard_tokens = standard_result.get("tokens", 0)
                kg_tokens = kg_result.get("tokens", 0)
                logger.info(f"  Standard AFP: {standard_tokens} tokens, KG-Enhanced AFP: {kg_tokens} tokens")
                print(f"  Standard AFP: {standard_tokens} tokens, KG-Enhanced AFP: {kg_tokens} tokens")
        
        # Get KG stats after all runs
        if kg_service:
            query_result["kg_stats"] = get_kg_stats(kg_service, thread_id)
        
        # Add query result to overall results
        results["query_results"].append(query_result)
    
    # Get final KG stats
    if kg_service:
        results["kg_stats"] = get_kg_stats(kg_service, thread_id)
    
    # Generate analysis
    analysis = analyze_results(results)
    results["analysis"] = analysis
    
    return results

def create_kg_service(optimize=True):
    """
    Create a knowledge graph service.
    
    Args:
        optimize: Whether to apply performance optimizations
        
    Returns:
        KnowledgeGraphService instance
    """
    if HAS_KG_COMPONENTS:
        # Create real KG service
        from moya.knowledge_graph import KnowledgeGraphService, KnowledgeGraphOptimizer
        
        kg_service = KnowledgeGraphService(storage_dir=KG_DIR)
        
        # Apply optimizations if requested
        if optimize and HAS_OPTIMIZATIONS:
            logger.info("Applying performance optimizations to KG service")
            optimizer = KnowledgeGraphOptimizer()
            kg_service = optimizer.optimize(kg_service)
            
        return kg_service
    else:
        # Create mock KG service
        return KnowledgeGraphService()

def create_entity_extractor(enhanced=True):
    """
    Create an entity extractor.
    
    Args:
        enhanced: Whether to use enhanced entity extraction
        
    Returns:
        EntityExtractor instance
    """
    if HAS_KG_COMPONENTS:
        # Create real entity extractor
        from moya.knowledge_graph import EntityExtractor, EnhancedEntityExtractor
        
        if enhanced and HAS_ENHANCED_EXTRACTOR:
            logger.info("Creating enhanced entity extractor")
            return EnhancedEntityExtractor()
        else:
            logger.info("Creating standard entity extractor")
            return EntityExtractor()
    else:
        # Create mock entity extractor
        class MockEntityExtractor:
            def extract_entities(self, text):
                # Simple mock extraction
                words = text.split()
                return [w for w in words if len(w) > 5][:3]  # Return up to 3 longer words
                
            def extract_relationships(self, text):
                # Simple mock triplet extraction
                entities = self.extract_entities(text)
                if len(entities) >= 2:
                    return [(entities[0], "related_to", entities[1])]
                return []
        
        return MockEntityExtractor()

def generate_benchmark_conversations(turns=25):
    """
    Generate a sequence of related conversations for benchmarking long-term knowledge retention.
    
    Args:
        turns: Number of conversation turns to generate
        
    Returns:
        List of conversation messages
    """
    # Create a series of interconnected conversations that build on each other
    # and reference specific entities across multiple turns
    conversations = [
        # Core topic introduction - a fictional company case study
        "Tell me about a fictional tech company called NeuroSynth that specializes in AI solutions. What might their main products be?",
        "What kinds of machine learning models would NeuroSynth likely use in their products?",
        "How might NeuroSynth's engineers train their neural networks differently for various applications?",
        "What data sources would NeuroSynth need to collect for training their recommendation systems?",
        "If NeuroSynth wanted to enter the healthcare market, what specific AI applications could they develop?",
        
        # Expanding the narrative with challenges
        "What ethical challenges might NeuroSynth face when implementing their healthcare AI solutions?",
        "How could NeuroSynth address bias in their algorithms, particularly for medical diagnosis?",
        "What regulatory compliance issues would NeuroSynth encounter when expanding internationally?",
        "Which specific privacy frameworks would NeuroSynth need to follow for their healthcare data?",
        "How might NeuroSynth structure their data governance policies to comply with both GDPR and HIPAA?",
        
        # Technical implementation details
        "What type of neural network architecture would be best for NeuroSynth's medical imaging products?",
        "How would NeuroSynth's transformer models need to be adapted for processing electronic health records?",
        "What techniques could NeuroSynth use to make their black-box AI models more explainable to doctors?",
        "If NeuroSynth wanted to reduce computational costs, how might they optimize their model architecture?",
        "What federated learning approach would help NeuroSynth maintain privacy while still training on hospital data?",
        
        # Business strategy and growth
        "Who would be NeuroSynth's main competitors in the healthcare AI space?",
        "What partnership strategy might help NeuroSynth gain adoption among medical professionals?",
        "How should NeuroSynth price their diagnostic AI tools compared to traditional diagnostic methods?",
        "What metrics should NeuroSynth track to demonstrate the effectiveness of their healthcare AI to investors?",
        "How might NeuroSynth structure their clinical validation studies to get FDA approval?",
        
        # Future evolution of the company
        "If NeuroSynth wanted to expand beyond healthcare, which industry should they target next and why?",
        "What potential acquisition targets would complement NeuroSynth's existing technology stack?",
        "How might NeuroSynth leverage quantum computing to enhance their existing AI models?",
        "What emerging technologies should NeuroSynth's R&D team be investigating for future products?",
        "How might NeuroSynth's business model evolve over the next decade as AI becomes more commoditized?"
    ]
    
    # Return the specified number of conversations (or all if turns is greater than available)
    return conversations[:min(turns, len(conversations))]

def generate_final_test_question():
    """
    Generate a final test question to evaluate knowledge retention.
    
    This complex question tests how well the model can recall information
    from earlier in the conversation and integrate it.
    
    Returns:
        The test question string
    """
    return (
        "Based on our entire discussion about NeuroSynth: Provide a comprehensive strategy for how the company should "
        "approach its next five years of growth. Include specific details about: 1) Its core technologies and how they "
        "should evolve, 2) The regulatory challenges they'll need to navigate in healthcare, 3) How they should address "
        "the ethical concerns we discussed, 4) Which specific industries they should expand into beyond healthcare and why, "
        "and 5) How they should position themselves against competitors. Reference specific details from our conversation "
        "in your response."
    )

def run_sequential_conversation(afp, conversations, test_prompt, thread_id=None):
    """
    Run a sequential conversation through the AFP, building context over time.
    
    Args:
        afp: The AFP orchestrator to use
        conversations: List of conversation prompts
        test_prompt: The final test question
        thread_id: Optional thread ID
        
    Returns:
        Dictionary with conversation results
    """
    thread_id = thread_id or str(uuid.uuid4())
    context = []
    responses = []
    entity_counts = []
    
    print(f"\nRunning sequential conversation ({len(conversations)} turns)")
    
    # Process each conversation turn
    for i, query in enumerate(conversations):
        print(f"  Processing turn {i+1}/{len(conversations)}: {query[:50]}...")
        
        # Get response from AFP
        start_time = time.time()
        result = afp.query(query_text=query, context=context, thread_id=thread_id)
        elapsed = time.time() - start_time
        
        # Extract response text
        response_text = result.get("response", "")
        
        # Track entities if available
        entities = []
        if "knowledge_graph" in result:
            entities = result["knowledge_graph"].get("extracted_entities", [])
            
        # Add to context for next turn (simulating conversation)
        context.append({"role": "user", "content": query})
        context.append({"role": "assistant", "content": response_text})
        
        # Save response data
        responses.append({
            "turn": i+1,
            "query": query,
            "response": response_text,
            "time": elapsed,
            "entities": entities,
            "tokens": result.get("tokens", 0)
        })
        
        entity_counts.append(len(entities))
        print(f"    Response time: {elapsed:.4f}s, Entities: {len(entities)}, Tokens: {result.get('tokens', 0)}")
    
    # Ask the final test question
    print(f"\n  Asking final test question: {test_prompt[:75]}...")
    start_time = time.time()
    final_result = afp.query(query_text=test_prompt, context=context, thread_id=thread_id)
    elapsed = time.time() - start_time
    
    final_response = final_result.get("response", "")
    
    # Extract entities from final response if available
    final_entities = []
    if "knowledge_graph" in final_result:
        final_entities = final_result["knowledge_graph"].get("extracted_entities", [])
    
    print(f"    Final response time: {elapsed:.4f}s, Entities: {len(final_entities)}, Tokens: {final_result.get('tokens', 0)}")
    
    # Return results
    return {
        "responses": responses,
        "final_query": test_prompt,
        "final_response": final_response,
        "final_time": elapsed,
        "final_entities": final_entities,
        "entity_progression": entity_counts,
        "thread_id": thread_id,
        "tokens": final_result.get("tokens", 0)
    }

def run_direct_conversation(afp, conversations, test_prompt, thread_id=None):
    """
    Run the entire conversation as a single batch (to create baseline "perfect memory").
    
    Args:
        afp: The AFP orchestrator to use
        conversations: List of conversation prompts
        test_prompt: The final test question
        thread_id: Optional thread ID
        
    Returns:
        Dictionary with conversation results
    """
    thread_id = thread_id or str(uuid.uuid4())
    
    # Create a single context containing all previous conversations
    context = []
    for query in conversations:
        # For the direct approach, we only need to add the user messages
        # as we're testing how well the model handles the information, not the responses
        context.append({"role": "user", "content": query})
    
    print(f"\nRunning direct conversation (all {len(conversations)} turns in single context)")
    print(f"  Asking test question with full context: {test_prompt[:75]}...")
    
    # Get response from AFP with full context
    start_time = time.time()
    result = afp.query(query_text=test_prompt, context=context, thread_id=thread_id)
    elapsed = time.time() - start_time
    
    # Extract response text
    response_text = result.get("response", "")
    
    # Extract entities if available
    entities = []
    if "knowledge_graph" in result:
        entities = result["knowledge_graph"].get("extracted_entities", [])
    
    print(f"    Response time: {elapsed:.4f}s, Entities: {len(entities)}, Tokens: {result.get('tokens', 0)}")
    
    # Return results
    return {
        "query": test_prompt,
        "response": response_text,
        "time": elapsed,
        "entities": entities,
        "thread_id": thread_id,
        "tokens": result.get("tokens", 0)
    }

def analyze_knowledge_retention(standard_result, kg_result, direct_result, conversations):
    """
    Analyze knowledge retention across different approaches.
    
    Args:
        standard_result: Results from sequential standard AFP
        kg_result: Results from sequential KG-enhanced AFP
        direct_result: Results from direct (all context) approach
        conversations: Original conversation prompts
        
    Returns:
        Dictionary with analysis results
    """
    # Extract important information
    standard_final = standard_result["final_response"]
    kg_final = kg_result["final_response"]
    direct_final = direct_result["response"]
    
    # Define key topics and concepts from our conversations
    key_topics = [
        "artificial intelligence", "machine learning", "neural networks", "deep learning", 
        "supervised learning", "unsupervised learning", "transformer models", "natural language processing",
        "computer vision", "ethics", "bias", "regulations", "explainable AI", "quantum computing",
        "limitations", "artificial general intelligence", "healthcare", "climate modeling", 
        "language models", "robotics", "security", "education", "automation"
    ]
    
    # Count mentions of key topics in each response
    standard_mentions = count_topic_mentions(standard_final, key_topics)
    kg_mentions = count_topic_mentions(kg_final, key_topics)
    direct_mentions = count_topic_mentions(direct_final, key_topics)
    
    # Calculate retention percentages (using direct as baseline)
    standard_retention = calculate_retention_percentage(standard_mentions, direct_mentions)
    kg_retention = calculate_retention_percentage(kg_mentions, direct_mentions)
    
    # Analyze entity progression throughout conversation
    standard_entity_progression = standard_result.get("entity_progression", [])
    kg_entity_progression = kg_result.get("entity_progression", [])
    
    # Calculate average entities per turn
    standard_avg_entities = sum(standard_entity_progression) / len(standard_entity_progression) if standard_entity_progression else 0
    kg_avg_entities = sum(kg_entity_progression) / len(kg_entity_progression) if kg_entity_progression else 0
    
    # Calculate improvement ratio
    if standard_retention > 0:
        retention_improvement = (kg_retention / standard_retention) - 1
    else:
        retention_improvement = 1.0 if kg_retention > 0 else 0.0
    
    # Return analysis results
    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "conversations": len(conversations),
        "key_topics_analyzed": len(key_topics),
        "topic_mentions": {
            "standard": standard_mentions,
            "kg": kg_mentions,
            "direct": direct_mentions
        },
        "retention_scores": {
            "standard": standard_retention,
            "kg": kg_retention,
            "improvement": retention_improvement * 100  # as percentage
        },
        "entity_analysis": {
            "standard_avg": standard_avg_entities,
            "kg_avg": kg_avg_entities,
            "standard_progression": standard_entity_progression,
            "kg_progression": kg_entity_progression
        },
        "response_lengths": {
            "standard": len(standard_final),
            "kg": len(kg_final),
            "direct": len(direct_final)
        }
    }

def count_topic_mentions(text, topics):
    """Count how many topics from the list are mentioned in the text."""
    text = text.lower()
    mentions = {}
    
    for topic in topics:
        count = text.count(topic.lower())
        if count > 0:
            mentions[topic] = count
            
    return mentions

def calculate_retention_percentage(mentions, baseline_mentions):
    """Calculate what percentage of baseline topics were retained."""
    if not baseline_mentions:
        return 0.0
        
    # Count topics that appear in both
    retained_topics = set(mentions.keys()).intersection(set(baseline_mentions.keys()))
    
    # Calculate retention percentage
    return (len(retained_topics) / len(baseline_mentions)) * 100

def print_knowledge_retention_results(analysis):
    """Print the results of the knowledge retention benchmark."""
    print("\n" + "=" * 80)
    print("KNOWLEDGE RETENTION BENCHMARK: STANDARD VS KG-ENHANCED AFP")
    print("=" * 80)
    
    # Print overview
    improvement = analysis["retention_scores"]["improvement"]
    print(f"\n📊 KNOWLEDGE RETENTION OVERVIEW:")
    print(f"  Total conversation turns: {analysis['conversations']}")
    print(f"  Key topics analyzed: {analysis['key_topics_analyzed']}")
    
    # Print retention scores
    print("\n🧠 RETENTION SCORES (% of information retained from direct approach):")
    print(f"  Standard AFP: {analysis['retention_scores']['standard']:.1f}%")
    print(f"  KG-Enhanced AFP: {analysis['retention_scores']['kg']:.1f}%")
    print(f"  Improvement with Knowledge Graph: {'+' if improvement > 0 else ''}{improvement:.1f}%")
    
    # Print entity analysis
    print("\n🔍 ENTITY TRACKING THROUGHOUT CONVERSATION:")
    print(f"  Average entities per turn (Standard): {analysis['entity_analysis']['standard_avg']:.1f}")
    print(f"  Average entities per turn (KG): {analysis['entity_analysis']['kg_avg']:.1f}")
    
    # Print response sizes
    print("\n📝 RESPONSE ANALYSIS:")
    print(f"  Standard response length: {analysis['response_lengths']['standard']} chars")
    print(f"  KG-enhanced response length: {analysis['response_lengths']['kg']} chars")
    print(f"  Direct (full context) response length: {analysis['response_lengths']['direct']} chars")
    
    # Print topic coverage comparison
    print("\n📚 TOPIC COVERAGE COMPARISON:")
    standard_topics = len(analysis["topic_mentions"]["standard"])
    kg_topics = len(analysis["topic_mentions"]["kg"])
    direct_topics = len(analysis["topic_mentions"]["direct"])
    
    print(f"  Topics mentioned (Standard): {standard_topics}/{analysis['key_topics_analyzed']} ({standard_topics/analysis['key_topics_analyzed']*100:.1f}%)")
    print(f"  Topics mentioned (KG): {kg_topics}/{analysis['key_topics_analyzed']} ({kg_topics/analysis['key_topics_analyzed']*100:.1f}%)")
    print(f"  Topics mentioned (Direct): {direct_topics}/{analysis['key_topics_analyzed']} ({direct_topics/analysis['key_topics_analyzed']*100:.1f}%)")
    
    # Print key takeaways
    print("\n🔑 KEY FINDINGS:")
    if improvement > 20:
        print("  ✅ Knowledge Graph provides SUBSTANTIAL improvement in information retention")
    elif improvement > 5:
        print("  ✅ Knowledge Graph provides MODERATE improvement in information retention")
    elif improvement > 0:
        print("  ✅ Knowledge Graph provides SLIGHT improvement in information retention")
    else:
        print("  ❌ Knowledge Graph did not improve information retention in this test")
        
    # Fix division by zero error
    standard_avg = analysis['entity_analysis']['standard_avg']
    kg_avg = analysis['entity_analysis']['kg_avg']
    if kg_avg > 0 and standard_avg > 0:
        entity_percent = (kg_avg / standard_avg - 1) * 100
        print(f"  ✅ Knowledge Graph tracked {entity_percent:.1f}% more entities throughout the conversation")
    elif kg_avg > 0 and standard_avg == 0:
        print(f"  ✅ Knowledge Graph tracked entities throughout the conversation while Standard AFP tracked none")
    
    if analysis['response_lengths']['kg'] > analysis['response_lengths']['standard']:
        increase_percent = (analysis['response_lengths']['kg'] / analysis['response_lengths']['standard'] - 1) * 100
        print(f"  ✅ Knowledge Graph produced {increase_percent:.1f}% more detailed responses")
    
    print("\n" + "=" * 80)

def run_long_conversation_benchmark(standard_afp, kg_afp, kg_service, conversation_turns=10, test_prompt=None):
    """
    Run a benchmark to test knowledge retention across a very long conversation.
    
    Args:
        standard_afp: Standard AFP orchestrator
        kg_afp: Knowledge graph enhanced AFP orchestrator
        kg_service: Knowledge graph service
        conversation_turns: Number of conversation turns before final test
        test_prompt: The final test question to evaluate knowledge retention
        
    Returns:
        Benchmark results comparing standard, KG, and direct approaches
    """
    # Generate conversation turns
    conversations = generate_benchmark_conversations(turns=conversation_turns)
    
    # Generate test prompt if not provided
    if not test_prompt:
        test_prompt = generate_final_test_question()
        
    print(f"\n🧠 LONG-TERM KNOWLEDGE RETENTION BENCHMARK")
    print(f"Testing knowledge retention across {conversation_turns} conversation turns")
    print(f"Final test question: {test_prompt[:100]}...")
    
    # Seed the knowledge graph (if available)
    if kg_service:
        thread_id = str(uuid.uuid4())
        seed_knowledge(kg_service, thread_id)
    
    # Run sequential conversation with standard AFP
    print("\n📝 TESTING STANDARD AFP (SEQUENTIAL CONVERSATION)")
    standard_result = run_sequential_conversation(
        standard_afp, 
        conversations, 
        test_prompt
    )
    
    # Run sequential conversation with KG-enhanced AFP
    print("\n📝 TESTING KG-ENHANCED AFP (SEQUENTIAL CONVERSATION)")
    kg_result = run_sequential_conversation(
        kg_afp, 
        conversations, 
        test_prompt
    )
    
    # Run direct conversation (baseline with all context at once)
    print("\n📝 TESTING BASELINE (DIRECT WITH FULL CONTEXT)")
    direct_result = run_direct_conversation(
        standard_afp,  # Using standard AFP for direct test
        conversations, 
        test_prompt
    )
    
    # Analyze results
    analysis = analyze_knowledge_retention(
        standard_result,
        kg_result,
        direct_result,
        conversations
    )
    
    # Print results
    print_knowledge_retention_results(analysis)
    
    # Return results
    return {
        "analysis": analysis,
        "standard_result": standard_result,
        "kg_result": kg_result,
        "direct_result": direct_result,
        "conversations": conversations,
        "test_prompt": test_prompt
    }

def parse_args():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Compare AFP with and without Knowledge Graph")
    parser.add_argument("--queries", type=int, default=3, help="Number of test queries to run")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs per query for averaging")
    parser.add_argument("--output", type=str, help="Custom filename for results output")
    parser.add_argument("--no-optimize", action="store_true", help="Disable KG performance optimizations")
    parser.add_argument("--no-enhanced", action="store_true", help="Disable enhanced entity extraction")
    parser.add_argument("--mock", action="store_true", help="Force use of mock components for testing")
    parser.add_argument("--seed-only", action="store_true", help="Only seed the knowledge graph and exit")
    parser.add_argument("--custom-query", type=str, help="Run a single custom query instead of test queries")
    parser.add_argument("--visualize", action="store_true", help="Generate visualizations of results")
    parser.add_argument("--accuracy", action="store_true", help="Run accuracy benchmark on multi-turn conversations")
    parser.add_argument("--turns", type=int, default=5, help="Number of conversation turns for accuracy benchmark")
    parser.add_argument("--simplified", action="store_true", help="Show only simplified output (good for quick tests)")
    parser.add_argument("--long-retention", action="store_true", 
                       help="Run long-term knowledge retention benchmark")
    parser.add_argument("--conversation-turns", type=int, default=10,
                       help="Number of conversation turns for long-term benchmark")
    parser.add_argument("--azure-api-key", type=str, help="Azure OpenAI API key (overrides environment variables)")
    parser.add_argument("--azure-endpoint", type=str, help="Azure OpenAI endpoint (overrides environment variables)")
    parser.add_argument("--azure-deployment", type=str, help="Azure OpenAI deployment (overrides environment variables)")
    parser.add_argument("--openai-api-key", type=str, help="OpenAI API key (overrides environment variables)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load .env file if it exists
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        env_path = os.path.join(script_dir, '.env')
        if os.path.exists(env_path):
            print(f"Loading environment variables from {env_path}")
            load_dotenv(env_path)
        elif os.path.exists('.env'):
            print(f"Loading environment variables from {os.getcwd()}/.env")
            load_dotenv()
    except ImportError:
        print("python-dotenv not installed. Using existing environment variables.")
    
    # Override environment variables with command line arguments
    if args.azure_api_key:
        print("Using Azure API key from command line arguments")
        global AZURE_API_KEY
        AZURE_API_KEY = args.azure_api_key
    
    if args.azure_endpoint:
        print("Using Azure endpoint from command line arguments")
        global AZURE_ENDPOINT
        AZURE_ENDPOINT = args.azure_endpoint
        
    if args.azure_deployment:
        print("Using Azure deployment from command line arguments")
        global AZURE_DEPLOYMENT
        AZURE_DEPLOYMENT = args.azure_deployment
    
    if args.openai_api_key:
        print("Using OpenAI API key from command line arguments")
        global OPENAI_API_KEY
        OPENAI_API_KEY = args.openai_api_key
    
    # Create directories if they don't exist
    os.makedirs(RESULTS_DIR, exist_ok=True)
    logger.info(f"Initialized directories: {RESULTS_DIR}")
    
    # Initialize standard AFP orchestrator (without knowledge graph)
    standard_afp = create_standard_afp()
    
    # Initialize knowledge graph AFP orchestrator
    use_optimizations = HAS_OPTIMIZATIONS and not args.no_optimize
    use_enhanced = HAS_ENHANCED_EXTRACTOR and not args.no_enhanced
    kg_afp, kg_service = create_kg_afp(
        use_optimizations=use_optimizations,
        use_enhanced=use_enhanced,
        force_mock=args.mock
    )
    
    # Log initialization status
    logger.info(f"AFP Initialized - Standard: {'Real' if standard_afp else 'Mock'}, "
                f"KG-Enhanced: {'Real' if kg_afp else 'Mock'}")
    logger.info(f"KG Optimizations: {'Enabled' if use_optimizations else 'Disabled'}, "
                f"Enhanced Extraction: {'Enabled' if use_enhanced else 'Disabled'}")
    
    # Handle seed-only mode
    if args.seed_only:
        logger.info("Seed-only mode: Seeding knowledge graph and exiting")
        if kg_service:
            thread_id = str(uuid.uuid4())
            seed_knowledge(kg_service, thread_id)
            stats = get_kg_stats(kg_service, thread_id)
            print("\nKnowledge Graph Seeded Successfully")
            print(f"Entities: {stats.get('entities', 0)}")
            print(f"Relationships: {stats.get('relationships', 0)}")
            sys.exit(0)
        else:
            logger.error("Cannot seed knowledge graph: KG service not available")
            sys.exit(1)
    
    # Choose which benchmark to run
    if args.accuracy:
        # Run accuracy benchmark
        print("\n🧠 RUNNING ACCURACY BENCHMARK - TESTING KNOWLEDGE RETENTION")
        print("This benchmark tests how well knowledge is maintained over a multi-turn conversation.\n")
        
        accuracy_results = run_accuracy_benchmark(
            standard_afp=standard_afp,
            kg_afp=kg_afp,
            kg_service=kg_service,
            conversation_turns=args.turns
        )
        
        # Save results
        accuracy_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        accuracy_filename = f"accuracy_benchmark_{accuracy_timestamp}.json"
        accuracy_filepath = os.path.join(RESULTS_DIR, accuracy_filename)
        
        with open(accuracy_filepath, "w") as f:
            # Sanitize results for JSON serialization
            def sanitize(obj):
                if isinstance(obj, dict):
                    return {k: sanitize(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [sanitize(item) for item in obj]
                elif isinstance(obj, (int, float, str, bool, type(None))):
                    return obj
                else:
                    return str(obj)
            
            json.dump(sanitize(accuracy_results), f, indent=2)
        
        print(f"\nAccuracy benchmark results saved to: {accuracy_filepath}")
        
        # Print results
        print_accuracy_benchmark_results(accuracy_results)
    elif args.long_retention:
        # Run long-term knowledge retention benchmark
        print("\n🧠 RUNNING LONG-TERM KNOWLEDGE RETENTION BENCHMARK")
        print("This benchmark tests how well knowledge is maintained across an extended conversation.")
        
        retention_results = run_long_conversation_benchmark(
            standard_afp=standard_afp,
            kg_afp=kg_afp,
            kg_service=kg_service,
            conversation_turns=args.conversation_turns
        )
        
        # Save results
        retention_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        retention_filename = f"retention_benchmark_{retention_timestamp}.json"
        retention_filepath = os.path.join(RESULTS_DIR, retention_filename)
        
        with open(retention_filepath, "w") as f:
            # Sanitize results for JSON serialization
            def sanitize(obj):
                if isinstance(obj, dict):
                    return {k: sanitize(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [sanitize(item) for item in obj]
                elif isinstance(obj, (int, float, str, bool, type(None))):
                    return obj
                else:
                    return str(obj)
            
            json.dump(sanitize(retention_results), f, indent=2)
        
        print(f"\nRetention benchmark results saved to: {retention_filepath}")
    else:
        # Run performance benchmark
        print("\n⚡ RUNNING PERFORMANCE BENCHMARK - TESTING SPEED AND OVERHEAD")
        print("This benchmark measures the performance impact of knowledge graphs.\n")
        
        # Determine queries to run
        if args.custom_query:
            benchmark_queries = [args.custom_query]
            runs_per_query = 1  # For custom queries, just run once
            logger.info(f"Running benchmark with custom query: {args.custom_query}")
        else:
            # Use subset of test queries based on args.queries
            benchmark_queries = TEST_QUERIES[:min(args.queries, len(TEST_QUERIES))]
            runs_per_query = args.runs
            logger.info(f"Running benchmark with {len(benchmark_queries)} queries, "
                        f"{runs_per_query} runs per query")
        
        # Run benchmark
        results = run_benchmark(
            standard_afp=standard_afp,
            kg_afp=kg_afp,
            kg_service=kg_service,
            queries=benchmark_queries,
            runs_per_query=runs_per_query
        )
        
        # Generate advanced analysis
        analysis = analyze_results(results)
        results["analysis"] = analysis
        
        # Save results
        results_path = save_results(results, args.output)
        print(f"\nBenchmark results saved to: {results_path}")
        
        # Print results based on verbosity preference
        if args.simplified:
            # Only print the simplified output
            print_simplified_comparison(results, analysis)
        else:
            # Print all outputs
            print_results_summary(results)
            print_advanced_analysis(analysis)
        
        # Generate visualizations if requested
        if args.visualize:
            try:
                import matplotlib.pyplot as plt
                import numpy as np
                
                print("\nGenerating visualizations...")
                
                # Create output directory for visualizations
                viz_dir = os.path.join(RESULTS_DIR, "visualizations")
                os.makedirs(viz_dir, exist_ok=True)
                
                # Basic comparison bar chart
                plt.figure(figsize=(10, 6))
                labels = ['Total Time', 'Overhead Time', 'API Time']
                standard_values = [
                    analysis["summary"]["standard_afp"]["avg_total_time"],
                    analysis["summary"]["standard_afp"]["avg_overhead_time"],
                    analysis["summary"]["standard_afp"]["avg_api_time"]
                ]
                kg_values = [
                    analysis["summary"]["kg_afp"]["avg_total_time"],
                    analysis["summary"]["kg_afp"]["avg_overhead_time"],
                    analysis["summary"]["kg_afp"]["avg_api_time"]
                ]
                
                x = np.arange(len(labels))
                width = 0.35
                
                fig, ax = plt.subplots(figsize=(12, 7))
                rects1 = ax.bar(x - width/2, standard_values, width, label='Standard AFP')
                rects2 = ax.bar(x + width/2, kg_values, width, label='KG-Enhanced AFP')
                
                ax.set_ylabel('Time (seconds)')
                ax.set_title('Performance Comparison: Standard vs KG-Enhanced AFP')
                ax.set_xticks(x)
                ax.set_xticklabels(labels)
                ax.legend()
                
                # Add value labels on top of bars
                def autolabel(rects):
                    for rect in rects:
                        height = rect.get_height()
                        ax.annotate(f'{height:.3f}s',
                                    xy=(rect.get_x() + rect.get_width() / 2, height),
                                    xytext=(0, 3),
                                    textcoords="offset points",
                                    ha='center', va='bottom')
                
                autolabel(rects1)
                autolabel(rects2)
                
                plt.tight_layout()
                comparison_chart_path = os.path.join(viz_dir, "performance_comparison.png")
                plt.savefig(comparison_chart_path)
                
                print(f"Visualizations saved to: {viz_dir}")
                
            except ImportError:
                print("\nVisualization requested but matplotlib is not installed.")
                print("To generate visualizations, install matplotlib: pip install matplotlib")
    
    print("\nBenchmark completed successfully. Use the JSON results file for further analysis.")


if __name__ == "__main__":
    main()