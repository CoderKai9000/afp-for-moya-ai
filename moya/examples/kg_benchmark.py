"""
Knowledge Graph Benchmark Script

This script compares the performance and response quality of the AFP orchestrator
with and without the knowledge graph integration.
"""

import os
import sys
import time
import json
import logging
import random
from typing import Dict, List, Any, Optional, Tuple, Set
from pathlib import Path
import argparse
import uuid
import shutil

# Add the parent directory to sys.path to allow importing the moya package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Import required modules from moya
from moya.knowledge_graph.graph_service import KnowledgeGraphService
from moya.knowledge_graph.entity_extraction import EntityExtractor
from moya.knowledge_graph.enhanced_extraction import create_enhanced_extractor
from moya.knowledge_graph.orchestrator import KnowledgeGraphOrchestrator
from moya.knowledge_graph.performance_utils import optimize_kg_service, PerformanceMonitor

# Import AFP components (assuming they exist in the proper locations)
# Replace these imports with the actual imports for your AFP orchestrator
try:
    from moya.afp.orchestrator import Orchestrator as AFPOrchestrator
    from moya.afp.azure_agent import AzureOpenAIAgent
    from moya.afp.agent import Agent
except ImportError:
    print("Warning: Could not import AFP components. Using mock objects for testing.")
    
    # Create mock classes if AFP components are not available
    class Agent:
        def __init__(self, name, model="gpt-35-turbo", temperature=0.7):
            self.name = name
            self.model = model
            self.temperature = temperature
            
        def generate(self, prompt, context=None):
            return f"Response from {self.name} (mock): {prompt[:50]}..."
    
    class AzureOpenAIAgent(Agent):
        pass
    
    class AFPOrchestrator:
        def __init__(self, agents=None, **kwargs):
            self.agents = agents or []
            
        def add_agent(self, agent):
            self.agents.append(agent)
            
        def generate(self, prompt, context=None):
            return f"Orchestrated response (mock): {prompt[:50]}..."

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("kg_benchmark.log")]
)
logger = logging.getLogger("kg_benchmark")

# List of benchmark conversation scenarios for testing
BENCHMARK_SCENARIOS = [
    {
        "name": "New product inquiry",
        "description": "Customer asking about a new product with follow-up questions",
        "conversation": [
            "Hi, I saw your new XPS 15 laptop and wanted to know more about it.",
            "What kind of processor does it use and how much RAM does it come with?",
            "Is the XPS 15 good for video editing? I work with Adobe Premiere a lot.",
            "Great! How does it compare to the MacBook Pro for creative work?",
            "What kind of warranty options do you offer for the XPS 15?"
        ]
    },
    {
        "name": "Technical support",
        "description": "User experiencing technical issues with software",
        "conversation": [
            "I'm having trouble with my Windows installation. It keeps freezing during startup.",
            "Yes, I've tried restarting in safe mode but the same thing happens.",
            "The last thing I installed was a graphics driver update for my NVIDIA card.",
            "I can access the recovery console. What commands should I try?",
            "Should I try to roll back the driver or do a system restore?"
        ]
    },
    {
        "name": "Travel planning",
        "description": "User planning a trip to multiple destinations",
        "conversation": [
            "I'm planning a trip to Europe next summer and want to visit Paris, Rome, and Barcelona.",
            "I'll be traveling for about 2 weeks. What's the best way to get between these cities?",
            "I'm interested in art museums and historic sites. What are the must-see attractions?",
            "What's the best time in June to visit Paris to avoid the largest crowds?",
            "Can you recommend some affordable hotels near the city centers?"
        ]
    },
    {
        "name": "Educational inquiry",
        "description": "Student asking about machine learning concepts",
        "conversation": [
            "I'm studying machine learning and struggling with understanding neural networks.",
            "Specifically, I don't quite get how backpropagation works in a deep neural network.",
            "Can you explain the difference between CNN and RNN architectures?",
            "What kind of neural network would be best for processing text data?",
            "Are there any good online courses you'd recommend for learning more about transformer models?"
        ]
    },
    {
        "name": "Context retention",
        "description": "Conversation with multiple entities and references",
        "conversation": [
            "My name is Alex and I work at Microsoft as a software engineer.",
            "I'm currently working on a project using Azure AI services and Python.",
            "My colleague Sarah recommended using TensorFlow for our machine learning components.",
            "We're building an application that needs to process images and extract text.",
            "Do you think we should use GPT-4 or is there a better model for this specific task?"
        ]
    }
]

class BenchmarkRunner:
    """
    Run benchmarks comparing standard AFP orchestrator with knowledge graph enhanced orchestrator.
    """
    
    def __init__(
        self, 
        storage_dir: str = "knowledge_graphs/benchmark",
        enhanced_extraction: bool = True,
        optimize_performance: bool = True,
        cleanup_after: bool = True
    ):
        """
        Initialize the benchmark runner.
        
        Args:
            storage_dir: Directory for storing knowledge graphs
            enhanced_extraction: Whether to use enhanced entity extraction
            optimize_performance: Whether to apply performance optimizations
            cleanup_after: Whether to clean up benchmark files after running
        """
        self.storage_dir = storage_dir
        self.enhanced_extraction = enhanced_extraction
        self.optimize_performance = optimize_performance
        self.cleanup_after = cleanup_after
        self.results = {}
        
        # Create unique benchmark ID
        self.benchmark_id = f"benchmark_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        
        # Create storage directory if it doesn't exist
        Path(storage_dir).mkdir(parents=True, exist_ok=True)
        
        # Performance monitor for tracking metrics
        self.performance_monitor = PerformanceMonitor(
            report_directory=Path(storage_dir) / "performance"
        )
        
        logger.info(f"Initialized benchmark runner with ID: {self.benchmark_id}")
        logger.info(f"Storage directory: {storage_dir}")
        logger.info(f"Enhanced extraction: {enhanced_extraction}")
        logger.info(f"Performance optimization: {optimize_performance}")
    
    def setup_agents(self) -> Tuple[AFPOrchestrator, KnowledgeGraphOrchestrator]:
        """
        Set up both standard and knowledge graph enhanced orchestrators.
        
        Returns:
            Tuple of (standard_orchestrator, kg_orchestrator)
        """
        # Create agents
        # Typically you'd use real credentials here
        # For this example, we'll use environment variables or mock agents
        api_key = os.environ.get("AZURE_OPENAI_API_KEY", "mock_key")
        api_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com")
        api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2023-07-01-preview")
        
        agents = []
        try:
            # Create agents for the orchestrator
            english_agent = AzureOpenAIAgent(
                name="English",
                api_key=api_key,
                endpoint=api_endpoint,
                api_version=api_version,
                deployment_name=os.environ.get("DEPLOYMENT_NAME", "gpt-4"),
                temperature=0.7
            )
            
            technical_agent = AzureOpenAIAgent(
                name="Technical",
                api_key=api_key,
                endpoint=api_endpoint,
                api_version=api_version,
                deployment_name=os.environ.get("DEPLOYMENT_NAME", "gpt-4"),
                temperature=0.5
            )
            
            agents = [english_agent, technical_agent]
        except Exception as e:
            logger.warning(f"Error creating Azure OpenAI agents: {e}")
            logger.warning("Using mock agents instead")
            agents = [
                Agent(name="English", temperature=0.7),
                Agent(name="Technical", temperature=0.5),
            ]
        
        # Create standard AFP orchestrator
        standard_orchestrator = AFPOrchestrator(agents=agents)
        
        # Create Knowledge Graph service
        kg_service = KnowledgeGraphService(
            storage_dir=self.storage_dir,
            max_nodes_per_graph=500,
            default_prune_age=86400  # 24 hours
        )
        
        # Apply performance optimizations if enabled
        if self.optimize_performance:
            kg_service = optimize_kg_service(kg_service)
        
        # Create entity extractor (enhanced or standard)
        if self.enhanced_extraction:
            entity_extractor = create_enhanced_extractor(
                api_key=api_key,
                endpoint=api_endpoint,
                api_version=api_version,
                deployment_name=os.environ.get("DEPLOYMENT_NAME", "gpt-4")
            )
        else:
            entity_extractor = EntityExtractor(
                api_key=api_key,
                endpoint=api_endpoint,
                api_version=api_version,
                deployment_name=os.environ.get("DEPLOYMENT_NAME", "gpt-4")
            )
        
        # Create Knowledge Graph orchestrator (wrapping the standard orchestrator)
        kg_orchestrator = KnowledgeGraphOrchestrator(
            orchestrator=AFPOrchestrator(agents=agents),
            knowledge_graph_service=kg_service,
            entity_extractor=entity_extractor,
            enrich_context=True,
            log_kg_operations=True
        )
        
        return standard_orchestrator, kg_orchestrator
    
    def run_benchmark(self, scenarios=None):
        """
        Run benchmark on conversation scenarios.
        
        Args:
            scenarios: Optional list of scenarios to run (default: all scenarios)
        """
        scenarios = scenarios or BENCHMARK_SCENARIOS
        total_scenarios = len(scenarios)
        
        logger.info(f"Starting benchmark with {total_scenarios} scenarios")
        start_time = time.time()
        
        standard_orchestrator, kg_orchestrator = self.setup_agents()
        
        results = {
            "benchmark_id": self.benchmark_id,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "scenarios": {},
            "summary": {
                "total_scenarios": total_scenarios,
                "total_messages": 0,
                "avg_response_time_standard": 0,
                "avg_response_time_kg": 0,
                "performance_difference_percent": 0
            }
        }
        
        total_standard_time = 0
        total_kg_time = 0
        total_messages = 0
        
        for i, scenario in enumerate(scenarios):
            scenario_name = scenario["name"]
            logger.info(f"Running scenario {i+1}/{total_scenarios}: {scenario_name}")
            
            # Create result structure for this scenario
            scenario_result = {
                "name": scenario_name,
                "description": scenario.get("description", ""),
                "messages": len(scenario["conversation"]),
                "standard": {
                    "responses": [],
                    "total_time": 0,
                    "avg_time": 0
                },
                "knowledge_graph": {
                    "responses": [],
                    "total_time": 0,
                    "avg_time": 0,
                    "entities_extracted": 0,
                    "triplets_added": 0
                }
            }
            
            # Create unique thread IDs for this scenario
            standard_thread_id = f"{self.benchmark_id}_standard_{i}"
            kg_thread_id = f"{self.benchmark_id}_kg_{i}"
            
            # Track conversation context
            standard_context = []
            kg_context = []
            
            # Run through each message in the conversation
            for j, message in enumerate(scenario["conversation"]):
                logger.info(f"  Message {j+1}/{len(scenario['conversation'])}")
                total_messages += 1
                
                # Run standard orchestrator
                standard_start = time.time()
                standard_response = standard_orchestrator.generate(
                    message, 
                    context=standard_context
                )
                standard_end = time.time()
                standard_time = standard_end - standard_start
                
                # Add to standard context
                standard_context.append({"role": "user", "content": message})
                standard_context.append({"role": "assistant", "content": standard_response})
                
                # Run knowledge graph orchestrator
                kg_start = time.time()
                kg_response = kg_orchestrator.generate(
                    message,
                    context=kg_context,
                    thread_id=kg_thread_id
                )
                kg_end = time.time()
                kg_time = kg_end - kg_start
                
                # Add to KG context
                kg_context.append({"role": "user", "content": message})
                kg_context.append({"role": "assistant", "content": kg_response})
                
                # Update timing stats
                total_standard_time += standard_time
                total_kg_time += kg_time
                scenario_result["standard"]["total_time"] += standard_time
                scenario_result["knowledge_graph"]["total_time"] += kg_time
                
                # Store responses
                scenario_result["standard"]["responses"].append({
                    "message": message,
                    "response": standard_response,
                    "time": standard_time
                })
                
                scenario_result["knowledge_graph"]["responses"].append({
                    "message": message,
                    "response": kg_response,
                    "time": kg_time
                })
                
                logger.info(f"    Standard response time: {standard_time:.2f}s")
                logger.info(f"    KG response time: {kg_time:.2f}s")
            
            # Calculate averages for this scenario
            msg_count = len(scenario["conversation"])
            scenario_result["standard"]["avg_time"] = scenario_result["standard"]["total_time"] / msg_count
            scenario_result["knowledge_graph"]["avg_time"] = scenario_result["knowledge_graph"]["total_time"] / msg_count
            
            # Get KG stats for this thread
            if hasattr(kg_orchestrator.knowledge_graph_service, 'get_or_create_graph'):
                graph = kg_orchestrator.knowledge_graph_service.get_or_create_graph(kg_thread_id)
                scenario_result["knowledge_graph"]["entities_extracted"] = len(graph.nodes())
                scenario_result["knowledge_graph"]["triplets_added"] = len(graph.edges())
            
            # Store scenario results
            results["scenarios"][scenario_name] = scenario_result
            
            logger.info(f"  Completed scenario {i+1}: {scenario_name}")
            logger.info(f"  Standard avg time: {scenario_result['standard']['avg_time']:.2f}s")
            logger.info(f"  KG avg time: {scenario_result['knowledge_graph']['avg_time']:.2f}s")
            logger.info(f"  Entities in graph: {scenario_result['knowledge_graph']['entities_extracted']}")
            logger.info(f"  Triplets in graph: {scenario_result['knowledge_graph']['triplets_added']}")
        
        # Calculate overall summary
        if total_messages > 0:
            avg_standard = total_standard_time / total_messages
            avg_kg = total_kg_time / total_messages
            perf_diff = ((avg_kg - avg_standard) / avg_standard) * 100
            
            results["summary"]["total_messages"] = total_messages
            results["summary"]["avg_response_time_standard"] = avg_standard
            results["summary"]["avg_response_time_kg"] = avg_kg
            results["summary"]["performance_difference_percent"] = perf_diff
        
        # Get performance report if available
        if hasattr(kg_orchestrator.knowledge_graph_service, 'get_performance_report'):
            performance_report = kg_orchestrator.knowledge_graph_service.get_performance_report()
            results["performance_report"] = performance_report
            
            # Log performance summary
            if hasattr(kg_orchestrator.knowledge_graph_service, 'log_performance_summary'):
                kg_orchestrator.knowledge_graph_service.log_performance_summary()
        
        # Save the results
        self.results = results
        self.save_results()
        
        total_time = time.time() - start_time
        logger.info(f"Benchmark completed in {total_time:.2f}s")
        logger.info(f"Standard avg response time: {avg_standard:.2f}s")
        logger.info(f"KG avg response time: {avg_kg:.2f}s")
        logger.info(f"Performance difference: {perf_diff:.2f}%")
        
        return results
    
    def save_results(self, filename=None):
        """
        Save benchmark results to a file.
        
        Args:
            filename: Optional custom filename
        
        Returns:
            Path to saved results file
        """
        if not filename:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"benchmark_results_{timestamp}.json"
        
        results_dir = Path(self.storage_dir) / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        results_path = results_dir / filename
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Results saved to {results_path}")
        return str(results_path)
    
    def generate_summary_report(self):
        """
        Generate a human-readable summary of benchmark results.
        
        Returns:
            String containing the report
        """
        if not self.results:
            return "No benchmark results available."
        
        summary = self.results["summary"]
        
        report = [
            "=" * 50,
            "KNOWLEDGE GRAPH BENCHMARK RESULTS",
            "=" * 50,
            f"Benchmark ID: {self.benchmark_id}",
            f"Date: {self.results['timestamp']}",
            f"Scenarios: {summary['total_scenarios']}",
            f"Total messages: {summary['total_messages']}",
            "-" * 50,
            "RESPONSE TIME COMPARISON:",
            f"Standard orchestrator: {summary['avg_response_time_standard']:.3f}s per message",
            f"KG orchestrator: {summary['avg_response_time_kg']:.3f}s per message",
            f"Difference: {summary['performance_difference_percent']:.1f}%",
            "-" * 50,
            "SCENARIO RESULTS:",
        ]
        
        # Add per-scenario results
        for name, scenario in self.results["scenarios"].items():
            report.extend([
                f"  {name}:",
                f"    Messages: {scenario['messages']}",
                f"    Standard avg time: {scenario['standard']['avg_time']:.3f}s",
                f"    KG avg time: {scenario['knowledge_graph']['avg_time']:.3f}s",
                f"    Entities extracted: {scenario['knowledge_graph']['entities_extracted']}",
                f"    Triplets added: {scenario['knowledge_graph']['triplets_added']}",
            ])
        
        # Add performance data if available
        if "performance_report" in self.results:
            perf = self.results["performance_report"]
            operations = perf.get("operations", {})
            
            report.extend([
                "-" * 50,
                "PERFORMANCE METRICS:",
                f"Total runtime: {perf.get('total_runtime', 0):.2f}s",
            ])
            
            # Add operation stats
            for op_name, op_data in operations.items():
                report.append(f"  {op_name}: {op_data.get('count', 0)} calls, "
                             f"{op_data.get('total_time', 0):.3f}s total, "
                             f"{op_data.get('average_time', 0):.3f}s avg")
        
        report.append("=" * 50)
        return "\n".join(report)
    
    def cleanup(self):
        """Clean up benchmark files if cleanup is enabled."""
        if self.cleanup_after:
            logger.info("Cleaning up benchmark files")
            try:
                # Remove benchmark graphs but keep results
                graph_dir = Path(self.storage_dir) / "graphs"
                if graph_dir.exists():
                    for path in graph_dir.glob(f"{self.benchmark_id}*"):
                        if path.is_file():
                            path.unlink()
                        elif path.is_dir():
                            shutil.rmtree(path)
                
                logger.info("Benchmark cleanup completed")
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")


def main():
    """Run the benchmark script."""
    parser = argparse.ArgumentParser(description="Knowledge Graph Benchmark")
    parser.add_argument("--storage-dir", default="knowledge_graphs/benchmark",
                        help="Directory for storing knowledge graphs")
    parser.add_argument("--enhanced", action="store_true",
                        help="Use enhanced entity extraction")
    parser.add_argument("--optimize", action="store_true",
                        help="Apply performance optimizations")
    parser.add_argument("--no-cleanup", action="store_true",
                        help="Don't clean up benchmark files after running")
    parser.add_argument("--scenarios", type=int, nargs="*",
                        help="Indexes of scenarios to run (default: all)")
    args = parser.parse_args()
    
    # Set up the benchmark runner
    benchmark = BenchmarkRunner(
        storage_dir=args.storage_dir,
        enhanced_extraction=args.enhanced,
        optimize_performance=args.optimize,
        cleanup_after=not args.no_cleanup
    )
    
    # Select scenarios
    if args.scenarios:
        scenarios = [BENCHMARK_SCENARIOS[i] for i in args.scenarios if 0 <= i < len(BENCHMARK_SCENARIOS)]
    else:
        scenarios = BENCHMARK_SCENARIOS
    
    try:
        # Run the benchmark
        benchmark.run_benchmark(scenarios=scenarios)
        
        # Generate and print report
        report = benchmark.generate_summary_report()
        print(report)
        
        # Save report to file
        with open(f"benchmark_report_{benchmark.benchmark_id}.txt", "w") as f:
            f.write(report)
        
    finally:
        # Clean up
        benchmark.cleanup()


if __name__ == "__main__":
    main() 