"""
Enhanced Entity Extraction Module for Moya

This module provides enhanced entity extraction capabilities for the knowledge graph,
including better entity normalization and coreference resolution.
"""

import re
import logging
import string
from typing import List, Dict, Tuple, Set, Any, Optional, Union
from collections import defaultdict

# Set up logging
logger = logging.getLogger(__name__)

class EnhancedEntityNormalizer:
    """
    Advanced entity normalization for improved consistency in the knowledge graph.
    
    Features:
    - Case normalization with preservation of proper nouns
    - Acronym handling
    - Stop word removal
    - Punctuation handling
    - Entity merging (e.g., "New York City" and "NYC" as the same entity)
    """
    
    def __init__(self, preserve_case: bool = False, 
                remove_stopwords: bool = True,
                merge_entities: bool = True,
                acronym_dict: Optional[Dict[str, str]] = None):
        """
        Initialize the enhanced entity normalizer.
        
        Args:
            preserve_case: Whether to preserve case for proper nouns
            remove_stopwords: Whether to remove stopwords from beginning of entities
            merge_entities: Whether to merge common entity variations
            acronym_dict: Dictionary of acronyms and their expansions
        """
        self.preserve_case = preserve_case
        self.remove_stopwords = remove_stopwords
        self.merge_entities = merge_entities
        self.acronym_dict = acronym_dict or {}
        
        # Common entity variations for merging
        self.entity_variations = {
            # Countries
            "usa": "united states",
            "us": "united states",
            "america": "united states",
            "uk": "united kingdom",
            "great britain": "united kingdom",
            
            # Companies
            "msft": "microsoft",
            "goog": "google",
            "googl": "google",
            "amzn": "amazon",
            "fb": "facebook",
            "meta": "facebook",
            
            # Technologies
            "ai": "artificial intelligence",
            "ml": "machine learning",
            "dl": "deep learning",
            "nn": "neural network",
            "nlp": "natural language processing",
            "cv": "computer vision",
            
            # Programming
            "js": "javascript",
            "ts": "typescript",
            "py": "python"
        }
        
        # Add reverse mappings for acronyms
        if self.merge_entities:
            # Create reverse mappings for canonical forms to variations
            self.canonical_forms = {}
            for variation, canonical in self.entity_variations.items():
                if canonical not in self.canonical_forms:
                    self.canonical_forms[canonical] = []
                self.canonical_forms[canonical].append(variation)
                
            # Also add acronyms to entity variations
            for acronym, expansion in self.acronym_dict.items():
                self.entity_variations[acronym.lower()] = expansion.lower()
        
        # Common stopwords to remove from beginning of entities
        self.stopwords = {
            "a", "an", "the", "this", "that", "these", "those", 
            "my", "your", "his", "her", "its", "our", "their"
        }
    
    def normalize(self, entity: str) -> str:
        """
        Normalize an entity string for improved consistency.
        
        Args:
            entity: The entity string to normalize
            
        Returns:
            Normalized entity string
        """
        if not entity:
            return ""
        
        # Basic cleanup
        entity = entity.strip()
        
        # Remove quotes and surrounding punctuation
        entity = entity.strip('"\'')
        entity = re.sub(r'^[^\w]+|[^\w]+$', '', entity)
        
        # Remove trailing punctuation preserving internal punctuation
        entity = entity.rstrip(string.punctuation)
        
        # Skip further processing if entity is now empty
        if not entity:
            return ""
        
        # Remove stopwords from beginning if enabled
        if self.remove_stopwords:
            entity_parts = entity.split()
            if entity_parts and entity_parts[0].lower() in self.stopwords:
                entity = " ".join(entity_parts[1:])
        
        # Handle case normalization
        if not self.preserve_case:
            # Convert to lowercase for consistent matching
            entity = entity.lower()
        else:
            # Preserve case for proper nouns, but lowercase common words
            pass  # Would implement more sophisticated case preservation here
        
        # Apply entity merging if enabled
        if self.merge_entities and entity.lower() in self.entity_variations:
            entity = self.entity_variations[entity.lower()]
        
        return entity
    
    def get_variations(self, entity: str) -> List[str]:
        """
        Get known variations of an entity.
        
        Args:
            entity: The entity to find variations for
            
        Returns:
            List of known entity variations
        """
        normalized = entity.lower()
        
        # Check if this is a known canonical form
        if normalized in self.canonical_forms:
            return self.canonical_forms[normalized]
            
        # Check if this is a variation with a canonical form
        if normalized in self.entity_variations:
            canonical = self.entity_variations[normalized]
            result = [normalized]  # Include the original
            
            # Add all other variations of the same canonical form
            if canonical in self.canonical_forms:
                for var in self.canonical_forms[canonical]:
                    if var != normalized:  # Don't duplicate the original
                        result.append(var)
            
            return result
            
        # No known variations
        return []
    
    def match_entities(self, entity1: str, entity2: str) -> bool:
        """
        Check if two entity strings might refer to the same entity.
        
        Args:
            entity1: First entity string
            entity2: Second entity string
            
        Returns:
            True if the entities likely refer to the same thing
        """
        # Normalize both entities
        norm1 = self.normalize(entity1)
        norm2 = self.normalize(entity2)
        
        # Direct match
        if norm1 == norm2:
            return True
            
        # Check variations
        if self.merge_entities:
            # Check if entity1 is a known variation of entity2
            if norm1 in self.entity_variations and self.entity_variations[norm1] == norm2:
                return True
                
            # Check if entity2 is a known variation of entity1
            if norm2 in self.entity_variations and self.entity_variations[norm2] == norm1:
                return True
                
            # Check if both map to same canonical form
            if (norm1 in self.entity_variations and norm2 in self.entity_variations and 
                self.entity_variations[norm1] == self.entity_variations[norm2]):
                return True
        
        # Strict non-match
        return False


class CoreferenceResolver:
    """
    Simple coreference resolution for entity mentions in text.
    
    Features:
    - Pronoun resolution
    - Mention grouping
    - Entity linkage across sentences
    """
    
    def __init__(self, window_size: int = 5):
        """
        Initialize the coreference resolver.
        
        Args:
            window_size: Number of sentences to look back for antecedents
        """
        self.window_size = window_size
        
        # Common pronoun-to-gender mappings
        self.pronoun_genders = {
            "he": "male",
            "him": "male",
            "his": "male",
            "himself": "male",
            
            "she": "female",
            "her": "female",
            "hers": "female",
            "herself": "female",
            
            "it": "neutral",
            "its": "neutral",
            "itself": "neutral",
            
            "they": "plural",
            "them": "plural",
            "their": "plural",
            "theirs": "plural",
            "themselves": "plural"
        }
        
        # Entity types for different categories
        self.entity_types = {
            "person": ["male", "female"],
            "organization": ["neutral"],
            "location": ["neutral"],
            "product": ["neutral"],
            "event": ["neutral"],
            "concept": ["neutral"],
            "group": ["plural"]
        }
    
    def resolve_coreferences(self, text: str, entities: List[str]) -> Dict[str, List[str]]:
        """
        Resolve coreferences in text.
        
        Args:
            text: The text to process
            entities: List of known entities in the text
            
        Returns:
            Dictionary mapping pronouns/mentions to their likely referents
        """
        # Split text into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Simple coreference resolution
        entity_mentions = {}
        current_focus = None
        last_entity_by_gender = defaultdict(list)
        
        for sentence in sentences:
            # Extract mentions in this sentence (simple approach)
            sentence_entities = []
            for entity in entities:
                if entity.lower() in sentence.lower():
                    sentence_entities.append(entity)
                    
                    # Update gender associations
                    # In a real implementation, would use NER to get entity types
                    gender = self._guess_gender(entity)
                    if gender:
                        last_entity_by_gender[gender] = entity
                    
                    # Update focus
                    current_focus = entity
            
            # Look for pronouns
            pronouns = self._extract_pronouns(sentence)
            for pronoun in pronouns:
                if pronoun.lower() in self.pronoun_genders:
                    gender = self.pronoun_genders[pronoun.lower()]
                    
                    # Try to find a matching entity with compatible gender
                    if gender in last_entity_by_gender and last_entity_by_gender[gender]:
                        antecedent = last_entity_by_gender[gender]
                        entity_mentions[pronoun] = antecedent
                    elif current_focus:
                        # Default to current focus if no gender match
                        entity_mentions[pronoun] = current_focus
        
        return entity_mentions
    
    def _extract_pronouns(self, text: str) -> List[str]:
        """Extract pronouns from text."""
        pronouns = []
        
        # Simple pronoun extraction using regex
        pronoun_pattern = r'\b(he|him|his|himself|she|her|hers|herself|it|its|itself|they|them|their|theirs|themselves)\b'
        matches = re.findall(pronoun_pattern, text, re.IGNORECASE)
        
        return matches
    
    def _guess_gender(self, entity: str) -> Optional[str]:
        """Guess gender/category of an entity (very simplistic)."""
        # This would use NER or other techniques in a real implementation
        # For now, just a simple heuristic example
        
        # Check for organization suffix
        if re.search(r'\b(Inc|Corp|LLC|Ltd|Company|Organization)\b', entity, re.IGNORECASE):
            return "neutral"  # Organizations
            
        # Check for location indicators
        if re.search(r'\b(Street|Road|Avenue|Boulevard|City|Country|State|Town|Village)\b', entity, re.IGNORECASE):
            return "neutral"  # Locations
            
        # Simple check for plurality
        if entity.lower().endswith('s') and not entity.lower().endswith('ss'):
            return "plural"  # Potential plural
            
        # Default - would use a name database or NER in real implementation
        return None


class EnhancedEntityExtractor:
    """
    Enhanced entity extraction with improved normalization and coreference resolution.
    
    This class extends the base EntityExtractor with better entity handling.
    """
    
    def __init__(self, base_extractor, normalizer=None, 
                coreference_resolver=None):
        """
        Initialize the enhanced entity extractor.
        
        Args:
            base_extractor: The base EntityExtractor to enhance
            normalizer: Entity normalizer to use
            coreference_resolver: Coreference resolver to use
        """
        self.base_extractor = base_extractor
        self.normalizer = normalizer or EnhancedEntityNormalizer()
        self.coreference_resolver = coreference_resolver or CoreferenceResolver()
    
    def extract_main_entities(self, text: str) -> List[str]:
        """
        Extract and normalize entities from text.
        
        Args:
            text: The text to extract entities from
            
        Returns:
            List of normalized entity strings
        """
        # Get base entities
        entities = self.base_extractor.extract_main_entities(text)
        
        # Normalize entities
        normalized_entities = [self.normalizer.normalize(e) for e in entities]
        normalized_entities = [e for e in normalized_entities if e]  # Remove empty
        
        # Remove duplicates while preserving order
        unique_entities = []
        seen = set()
        for entity in normalized_entities:
            if entity not in seen:
                seen.add(entity)
                unique_entities.append(entity)
        
        return unique_entities
    
    def extract_relationships(self, text: str) -> List[Tuple[str, str, str]]:
        """
        Extract normalized relationship triplets from text.
        
        Args:
            text: The text to extract relationships from
            
        Returns:
            List of normalized (subject, relation, object) tuples
        """
        # Get base triplets
        triplets = self.base_extractor.extract_relationships(text)
        
        # Get entities first for coreference resolution
        entities = self.extract_main_entities(text)
        
        # Resolve coreferences
        coreference_map = self.coreference_resolver.resolve_coreferences(text, entities)
        
        # Normalize and resolve triplets
        normalized_triplets = []
        for subj, rel, obj in triplets:
            # Normalize subject and object
            norm_subj = self.normalizer.normalize(subj)
            norm_rel = rel.lower().strip()
            norm_obj = self.normalizer.normalize(obj)
            
            # Resolve coreferences if applicable
            if norm_subj in coreference_map:
                norm_subj = coreference_map[norm_subj]
                
            if norm_obj in coreference_map:
                norm_obj = coreference_map[norm_obj]
            
            # Add if valid
            if norm_subj and norm_rel and norm_obj:
                normalized_triplets.append((norm_subj, norm_rel, norm_obj))
        
        # Remove duplicates
        unique_triplets = []
        seen = set()
        for triplet in normalized_triplets:
            triplet_key = tuple(t.lower() for t in triplet)
            if triplet_key not in seen:
                seen.add(triplet_key)
                unique_triplets.append(triplet)
        
        return unique_triplets
    
    def extract_from_conversation(self, 
                                messages: List[Dict[str, str]],
                                include_system: bool = False) -> Dict[str, Any]:
        """
        Extract entities and relationships from a conversation with enhanced processing.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            include_system: Whether to include system messages
            
        Returns:
            Dictionary with extracted entities and relationships
        """
        # Use the base extractor for initial extraction
        base_results = self.base_extractor.extract_from_conversation(messages, include_system)
        
        # Post-process the results with normalization and coreference resolution
        all_entities = set()
        all_triplets = []
        entities_by_turn = []
        triplets_by_turn = []
        
        # Collect all text for coreference resolution
        all_text = ""
        for message in messages:
            if message['role'] == 'system' and not include_system:
                continue
                
            content = message.get('content', '')
            if content and isinstance(content, str):
                all_text += content + " "
        
        # Get all entities for coreference resolution
        all_base_entities = base_results.get('entities', [])
        coreference_map = self.coreference_resolver.resolve_coreferences(all_text, all_base_entities)
        
        # Process each message
        for i, message_data in enumerate(base_results.get('entities_by_turn', [])):
            role = message_data.get('role', '')
            entities = message_data.get('entities', [])
            
            # Normalize entities
            normalized_entities = [self.normalizer.normalize(e) for e in entities]
            normalized_entities = [e for e in normalized_entities if e]  # Remove empty
            
            all_entities.update(normalized_entities)
            entities_by_turn.append({
                'role': role,
                'entities': normalized_entities
            })
        
        # Process triplets
        for i, message_data in enumerate(base_results.get('triplets_by_turn', [])):
            role = message_data.get('role', '')
            triplets = message_data.get('triplets', [])
            
            # Normalize triplets
            normalized_triplets = []
            for subj, rel, obj in triplets:
                norm_subj = self.normalizer.normalize(subj)
                norm_rel = rel.lower().strip()
                norm_obj = self.normalizer.normalize(obj)
                
                # Resolve coreferences if applicable
                if norm_subj in coreference_map:
                    norm_subj = coreference_map[norm_subj]
                    
                if norm_obj in coreference_map:
                    norm_obj = coreference_map[norm_obj]
                
                if norm_subj and norm_rel and norm_obj:
                    normalized_triplets.append((norm_subj, norm_rel, norm_obj))
            
            all_triplets.extend(normalized_triplets)
            triplets_by_turn.append({
                'role': role,
                'triplets': normalized_triplets
            })
        
        return {
            'entities': list(all_entities),
            'triplets': all_triplets,
            'entities_by_turn': entities_by_turn,
            'triplets_by_turn': triplets_by_turn
        }


def create_enhanced_extractor(base_extractor, custom_entity_mappings=None):
    """
    Create an enhanced entity extractor with custom settings.
    
    Args:
        base_extractor: Base EntityExtractor instance
        custom_entity_mappings: Optional dictionary of custom entity mappings
        
    Returns:
        Enhanced entity extractor instance
    """
    # Create normalizer with custom mappings if provided
    normalizer = EnhancedEntityNormalizer(
        preserve_case=False,
        remove_stopwords=True,
        merge_entities=True
    )
    
    # Add custom mappings if provided
    if custom_entity_mappings:
        normalizer.entity_variations.update(custom_entity_mappings)
        
        # Update canonical forms
        for variation, canonical in custom_entity_mappings.items():
            if canonical not in normalizer.canonical_forms:
                normalizer.canonical_forms[canonical] = []
            normalizer.canonical_forms[canonical].append(variation)
    
    # Create coreference resolver
    coreference_resolver = CoreferenceResolver(window_size=3)
    
    # Create and return enhanced extractor
    return EnhancedEntityExtractor(
        base_extractor=base_extractor,
        normalizer=normalizer,
        coreference_resolver=coreference_resolver
    ) 