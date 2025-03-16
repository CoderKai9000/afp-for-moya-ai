"""
Knowledge Graph Integration with AFP Orchestrator

This module provides utilities to integrate the knowledge graph functionality
with the Moya AFP orchestrator for enhanced context-aware multi-agent conversations.
"""

import logging
from typing import List, Dict, Any, Optional, Union, Tuple

# Set up logging
logger = logging.getLogger(__name__)

class KnowledgeGraphOrchestrator:
    """
    Integrates knowledge graph capabilities with the AFP orchestrator.
    
    This class wraps the AFP orchestrator and adds knowledge graph functionality,
    automatically extracting entities from conversations, building a knowledge graph,
    and enriching context for improved agent performance.
    """
    
    def __init__(self, 
                orchestrator, 
                knowledge_graph_service=None, 
                entity_extractor=None,
                auto_extract: bool = True,
                entity_extraction_frequency: str = "every_turn",
                context_enrichment: bool = True):
        """
        Initialize the knowledge graph orchestrator wrapper.
        
        Args:
            orchestrator: Base AFP orchestrator to wrap
            knowledge_graph_service: Knowledge graph service instance
            entity_extractor: Entity extractor instance
            auto_extract: Whether to automatically extract entities and relationships
            entity_extraction_frequency: When to extract entities ("every_turn" or "user_only")
            context_enrichment: Whether to enrich context with knowledge graph information
        """
        self.orchestrator = orchestrator
        self.knowledge_graph_service = knowledge_graph_service
        self.entity_extractor = entity_extractor
        self.auto_extract = auto_extract
        self.entity_extraction_frequency = entity_extraction_frequency
        self.context_enrichment = context_enrichment
        
        # Initialize conversation state
        self.conversation_state = {
            "last_entities": [],
            "last_triplets": []
        }
        
        logger.info(f"Initialized KnowledgeGraphOrchestrator with auto_extract={auto_extract}, context_enrichment={context_enrichment}")
    
    def query(self, query_text: str, thread_id: str = "default", 
             context: Optional[str] = None,
             instruction: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a query with knowledge graph-enhanced orchestration.
        
        Args:
            query_text: The query text to process
            thread_id: Thread identifier for conversation tracking
            context: Optional additional context information
            instruction: Optional instruction for the orchestrator
            
        Returns:
            Response dictionary with enhanced information
        """
        # Extract entities and relationships if enabled
        extracted_info = {}
        if self.auto_extract and self.entity_extractor:
            extracted_info = self._extract_and_update_graph(
                query_text, thread_id, is_user_query=True
            )
        
        # Enrich context with knowledge graph information if enabled
        enriched_context = context or ""
        if self.context_enrichment and self.knowledge_graph_service and extracted_info.get("entities"):
            # Get related information from knowledge graph
            kg_context = self.knowledge_graph_service.get_related_information(
                extracted_info.get("entities", []), 
                thread_id
            )
            
            if kg_context:
                enriched_context = f"{enriched_context}\n\n{kg_context}" if enriched_context else kg_context
        
        # Pass to base orchestrator with enriched context
        response = self.orchestrator.query(
            query_text, 
            thread_id=thread_id,
            context=enriched_context,
            instruction=instruction
        )
        
        # Extract from assistant response if configured to do so
        if (self.auto_extract and self.entity_extractor and 
            self.entity_extraction_frequency == "every_turn"):
            assistant_text = response.get('response', '')
            if assistant_text:
                self._extract_and_update_graph(
                    assistant_text, thread_id, is_user_query=False
                )
        
        # Add knowledge graph info to response metadata
        response["knowledge_graph"] = {
            "extracted_entities": extracted_info.get("entities", []),
            "extracted_triplets": extracted_info.get("triplets", []),
            "context_enriched": bool(enriched_context != context)
        }
        
        return response
    
    def chat(self, messages: List[Dict[str, str]], thread_id: str = "default",
            context: Optional[str] = None, 
            instruction: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a conversation with knowledge graph enhancement.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            thread_id: Thread identifier for conversation tracking
            context: Optional additional context information
            instruction: Optional instruction for the orchestrator
            
        Returns:
            Response dictionary with enhanced information
        """
        # Extract entities and relationships from conversation history
        if self.auto_extract and self.entity_extractor:
            extracted_info = self.entity_extractor.extract_from_conversation(messages)
            
            # Update knowledge graph with extracted information
            if self.knowledge_graph_service and extracted_info.get("triplets"):
                self.knowledge_graph_service.add_triplets(
                    thread_id,
                    extracted_info.get("triplets", [])
                )
                
            # Update conversation state
            self.conversation_state["last_entities"] = extracted_info.get("entities", [])
            self.conversation_state["last_triplets"] = extracted_info.get("triplets", [])
        
        # Enrich context with knowledge graph information
        enriched_context = context or ""
        if self.context_enrichment and self.knowledge_graph_service:
            # Get related information from knowledge graph for key entities
            user_messages = [m for m in messages if m.get('role') == 'user']
            if user_messages:
                # Extract from latest user message
                latest_user_message = user_messages[-1].get('content', '')
                if latest_user_message and self.entity_extractor:
                    entities = self.entity_extractor.extract_main_entities(latest_user_message)
                    
                    # Get related information from existing graph
                    kg_context = self.knowledge_graph_service.get_related_information(
                        entities, thread_id
                    )
                    
                    if kg_context:
                        enriched_context = f"{enriched_context}\n\n{kg_context}" if enriched_context else kg_context
        
        # Pass to base orchestrator with enriched context
        response = self.orchestrator.chat(
            messages, 
            thread_id=thread_id,
            context=enriched_context,
            instruction=instruction
        )
        
        # Extract from assistant response if configured to do so
        if (self.auto_extract and self.entity_extractor and 
            self.entity_extraction_frequency == "every_turn"):
            assistant_text = response.get('response', '')
            if assistant_text:
                extracted_info = self._extract_and_update_graph(
                    assistant_text, thread_id, is_user_query=False
                )
                
                # Update response metadata
                response.setdefault("knowledge_graph", {}).update({
                    "assistant_entities": extracted_info.get("entities", []),
                    "assistant_triplets": extracted_info.get("triplets", [])
                })
        
        # Add knowledge graph info to response metadata
        response.setdefault("knowledge_graph", {}).update({
            "context_enriched": bool(enriched_context != context)
        })
        
        return response
    
    def _extract_and_update_graph(self, text: str, thread_id: str, 
                               is_user_query: bool = True) -> Dict[str, Any]:
        """
        Extract entities and relationships from text and update the knowledge graph.
        
        Args:
            text: Text to extract from
            thread_id: Thread identifier
            is_user_query: Whether this is from a user query
            
        Returns:
            Dictionary with extracted information
        """
        result = {
            "entities": [],
            "triplets": []
        }
        
        # Skip if feature is disabled or components missing
        if not (self.auto_extract and self.entity_extractor):
            return result
            
        # Skip assistant messages if configured to extract from user only
        if not is_user_query and self.entity_extraction_frequency == "user_only":
            return result
        
        # Extract entities and relationships
        entities = self.entity_extractor.extract_main_entities(text)
        triplets = self.entity_extractor.extract_relationships(text)
        
        # Update knowledge graph if available
        if self.knowledge_graph_service and triplets:
            self.knowledge_graph_service.add_triplets(thread_id, triplets)
        
        # Update result
        result["entities"] = entities
        result["triplets"] = triplets
        
        # Update conversation state
        self.conversation_state["last_entities"] = entities
        self.conversation_state["last_triplets"] = triplets
        
        return result
    
    def get_graph_visualization(self, thread_id: str = "default", 
                              output_path: Optional[str] = None) -> Optional[str]:
        """
        Generate a visualization of the knowledge graph.
        
        Args:
            thread_id: Thread identifier
            output_path: Path to save the visualization
            
        Returns:
            Path to saved visualization if successful
        """
        if not self.knowledge_graph_service:
            logger.warning("Knowledge graph service not available for visualization")
            return None
            
        return self.knowledge_graph_service.visualize_graph(thread_id, output_path)
    
    def export_graph(self, thread_id: str = "default", format: str = "json") -> Dict[str, Any]:
        """
        Export the knowledge graph data.
        
        Args:
            thread_id: Thread identifier
            format: Export format
            
        Returns:
            Dictionary with exported graph data
        """
        if not self.knowledge_graph_service:
            logger.warning("Knowledge graph service not available for export")
            return {}
            
        return self.knowledge_graph_service.export_graph(thread_id, format)
    
    def get_entity_info(self, entity: str, thread_id: str = "default", 
                      max_distance: int = 2) -> Dict[str, Any]:
        """
        Get detailed information about an entity from the knowledge graph.
        
        Args:
            entity: Entity to get information about
            thread_id: Thread identifier
            max_distance: Maximum path distance to consider
            
        Returns:
            Dictionary with entity information
        """
        if not self.knowledge_graph_service:
            logger.warning("Knowledge graph service not available for entity info")
            return {}
        
        # Get direct relationships
        neighbors = self.knowledge_graph_service.get_entity_neighbors(
            entity, thread_id, max_distance
        )
        
        # Format as readable text
        return {
            "entity": entity,
            "relationships": neighbors,
            "summary": self._format_entity_summary(entity, neighbors)
        }
    
    def _format_entity_summary(self, entity: str, 
                             relationships: Dict[str, List[Dict[str, Any]]]) -> str:
        """
        Format entity relationships as a readable summary.
        
        Args:
            entity: Entity name
            relationships: Relationship data from get_entity_neighbors
            
        Returns:
            Formatted summary text
        """
        summary_lines = [f"Information about {entity}:"]
        
        # Add incoming relationships
        if relationships.get("incoming"):
            summary_lines.append("\nThings that relate to this entity:")
            for rel in relationships["incoming"]:
                if "source" in rel and "relation" in rel:
                    summary_lines.append(f"- {rel['source']} {rel['relation']} {entity}")
        
        # Add outgoing relationships
        if relationships.get("outgoing"):
            summary_lines.append("\nThings this entity relates to:")
            for rel in relationships["outgoing"]:
                if "target" in rel and "relation" in rel:
                    summary_lines.append(f"- {entity} {rel['relation']} {rel['target']}")
        
        return "\n".join(summary_lines)
    
    # Delegate all other methods to base orchestrator
    def __getattr__(self, name):
        """Delegate all other attribute access to the base orchestrator."""
        return getattr(self.orchestrator, name) 