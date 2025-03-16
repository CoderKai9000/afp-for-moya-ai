import re
from typing import List, Tuple, Optional, Dict, Any

class EntityExtractor:
    """
    A class that uses LLMs to extract entity relationships from text.
    It processes text to identify entities and their relationships,
    returning them as triplets for use in knowledge graphs.
    """
    
    def __init__(self, llm_agent):
        """
        Initialize the entity extractor with an LLM agent.
        
        Args:
            llm_agent: The LLM agent to use for extraction (must have handle_message method)
        """
        self.llm_agent = llm_agent
    
    def extract_entities(self, text: str, thread_id: str = None) -> List[Tuple[str, str, str]]:
        """
        Extract entity relationships from text as triplets.
        
        Args:
            text: The text to extract entities from
            thread_id: Optional thread ID for the LLM agent
            
        Returns:
            A list of (entity1, relation, entity2) triplets
        """
        # Create a structured prompt for entity extraction
        prompt = self._create_extraction_prompt(text)
        
        # Call the LLM to extract entities
        response = self.llm_agent.handle_message(prompt, thread_id=thread_id)
        
        # Parse the response into triplets
        triplets = self._parse_response(response)
        
        return triplets
    
    def extract_main_entities(self, text: str, thread_id: str = None) -> List[str]:
        """
        Extract the main entities mentioned in a text, without relationships.
        
        Args:
            text: The text to extract entities from
            thread_id: Optional thread ID for the LLM agent
            
        Returns:
            A list of entity names
        """
        prompt = f"""
        Identify the most important entities (people, places, concepts, objects) 
        mentioned in the following text. Return only the entity names, one per line:
        
        Text: {text}
        """
        
        response = self.llm_agent.handle_message(prompt, thread_id=thread_id)
        
        # Parse the response into a list of entities
        entities = []
        for line in response.strip().split('\n'):
            entity = line.strip()
            if entity and len(entity) > 1:  # Avoid single characters
                entities.append(entity)
        
        return entities
    
    def _create_extraction_prompt(self, text: str) -> str:
        """
        Create a structured prompt for entity extraction.
        
        Args:
            text: The text to extract entities from
            
        Returns:
            A formatted prompt for the LLM
        """
        return f"""
        Extract key entities and their relationships from the following text as triplets.
        Format each triplet as (Entity1, Relation, Entity2) where:
        - Entity1 and Entity2 are specific nouns or noun phrases
        - Relation is a verb or preposition that connects them
        
        Guidelines:
        - Focus on facts, not opinions
        - Extract direct relationships only
        - Be specific and precise
        - Avoid too generic entities like "user" or "topic"
        - Use consistent naming for the same entities
        
        Return only the triplets, one per line, with no additional text. Format:
        (Entity1, Relation, Entity2)
        
        Text: {text}
        """
    
    def _parse_response(self, response: str) -> List[Tuple[str, str, str]]:
        """
        Parse the LLM response into structured triplets.
        
        Args:
            response: The LLM's response to the extraction prompt
            
        Returns:
            A list of (entity1, relation, entity2) triplets
        """
        triplets = []
        
        # Common pattern: (Entity1, Relation, Entity2)
        pattern = r'\(([^,]+),\s*([^,]+),\s*([^)]+)\)'
        matches = re.findall(pattern, response)
        
        for match in matches:
            if len(match) == 3:
                entity1 = match[0].strip()
                relation = match[1].strip()
                entity2 = match[2].strip()
                
                # Simple validation checks
                if entity1 and relation and entity2:
                    # Basic normalization
                    entity1 = entity1.lower()
                    relation = relation.lower()
                    entity2 = entity2.lower()
                    
                    triplets.append((entity1, relation, entity2))
        
        # If no matches with the primary pattern, try an alternative format
        if not triplets:
            lines = response.strip().split('\n')
            for line in lines:
                parts = line.split(',')
                if len(parts) >= 3:
                    entity1 = parts[0].strip().strip('(').lower()
                    relation = parts[1].strip().lower()
                    # Combine remaining parts if there are more than 3
                    entity2 = ','.join(parts[2:]).strip().strip(')').lower()
                    
                    if entity1 and relation and entity2:
                        triplets.append((entity1, relation, entity2))
        
        return triplets 