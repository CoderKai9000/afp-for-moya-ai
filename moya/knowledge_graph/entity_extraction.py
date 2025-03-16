"""
Entity Extraction Module for Moya

This module provides functionality for extracting entities and relationships
from text using various techniques including dependency parsing, named entity
recognition, and LLM-based extraction.
"""

import re
import logging
from typing import List, Dict, Tuple, Set, Any, Optional, Union
from collections import defaultdict

# Set up logging
logger = logging.getLogger(__name__)

class EntityExtractor:
    """
    A class for extracting entities and relationships from text.
    
    Features:
    - Extraction of main entities from text
    - Identification of subject-relation-object triplets
    - Support for multiple extraction methods
    - Confidence scoring for extracted information
    """
    
    def __init__(self, llm_client=None, use_spacy: bool = True,
                confidence_threshold: float = 0.5,
                extraction_methods: List[str] = None):
        """
        Initialize the entity extractor.
        
        Args:
            llm_client: Optional language model client for extraction
            use_spacy: Whether to use spaCy for extraction
            confidence_threshold: Minimum confidence score for extracted entities
            extraction_methods: List of extraction methods to use
        """
        self.llm_client = llm_client
        self.use_spacy = use_spacy
        self.confidence_threshold = confidence_threshold
        self.extraction_methods = extraction_methods or ["rule_based", "llm"]
        
        # Initialize spaCy if requested
        self.nlp = None
        if self.use_spacy:
            try:
                import spacy
                # Try to load a larger model first, fall back to smaller one if not available
                try:
                    self.nlp = spacy.load("en_core_web_lg")
                    logger.info("Loaded spaCy model: en_core_web_lg")
                except OSError:
                    try:
                        self.nlp = spacy.load("en_core_web_md")
                        logger.info("Loaded spaCy model: en_core_web_md")
                    except OSError:
                        self.nlp = spacy.load("en_core_web_sm")
                        logger.info("Loaded spaCy model: en_core_web_sm")
            except ImportError:
                logger.warning("spaCy not installed. Install with: pip install spacy")
                logger.warning("Then download a model with: python -m spacy download en_core_web_sm")
                self.use_spacy = False
        
        logger.info(f"Initialized EntityExtractor with methods: {', '.join(self.extraction_methods)}")
    
    def extract_main_entities(self, text: str) -> List[str]:
        """
        Extract the main entities from text.
        
        Args:
            text: The text to extract entities from
            
        Returns:
            List of main entity strings
        """
        entities = set()
        
        # Apply each extraction method
        for method in self.extraction_methods:
            if method == "rule_based":
                rule_based_entities = self._extract_entities_rule_based(text)
                entities.update(rule_based_entities)
            
            elif method == "spacy" and self.use_spacy and self.nlp:
                spacy_entities = self._extract_entities_spacy(text)
                entities.update(spacy_entities)
                
            elif method == "llm" and self.llm_client:
                llm_entities = self._extract_entities_llm(text)
                entities.update(llm_entities)
        
        # Filter out very short entities and normalize
        filtered_entities = [self._normalize_entity(e) for e in entities 
                            if len(e.strip()) > 1]  # Require at least 2 chars
        
        # Remove duplicates while preserving order
        unique_entities = []
        seen = set()
        for entity in filtered_entities:
            normalized = entity.lower()
            if normalized not in seen and normalized:
                seen.add(normalized)
                unique_entities.append(entity)
        
        return unique_entities
    
    def extract_relationships(self, text: str) -> List[Tuple[str, str, str]]:
        """
        Extract relationship triplets (subject, relation, object) from text.
        
        Args:
            text: The text to extract relationships from
            
        Returns:
            List of (subject, relation, object) tuples
        """
        triplets = []
        
        # Apply each extraction method
        for method in self.extraction_methods:
            if method == "rule_based":
                rule_based_triplets = self._extract_triplets_rule_based(text)
                triplets.extend(rule_based_triplets)
            
            elif method == "spacy" and self.use_spacy and self.nlp:
                spacy_triplets = self._extract_triplets_spacy(text)
                triplets.extend(spacy_triplets)
                
            elif method == "llm" and self.llm_client:
                llm_triplets = self._extract_triplets_llm(text)
                triplets.extend(llm_triplets)
        
        # Normalize triplets
        normalized_triplets = []
        for subj, rel, obj in triplets:
            subj_norm = self._normalize_entity(subj)
            rel_norm = rel.lower().strip()
            obj_norm = self._normalize_entity(obj)
            
            if subj_norm and rel_norm and obj_norm:
                normalized_triplets.append((subj_norm, rel_norm, obj_norm))
        
        # Remove duplicates while preserving order
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
        Extract entities and relationships from a conversation history.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            include_system: Whether to include system messages
            
        Returns:
            Dictionary with extracted entities and relationships
        """
        all_entities = set()
        all_triplets = []
        entities_by_turn = []
        triplets_by_turn = []
        
        for message in messages:
            # Skip system messages if not included
            if message['role'] == 'system' and not include_system:
                continue
                
            content = message.get('content', '')
            if not content or not isinstance(content, str):
                continue
                
            # Extract from this message
            entities = self.extract_main_entities(content)
            triplets = self.extract_relationships(content)
            
            # Add to results
            all_entities.update(entities)
            all_triplets.extend(triplets)
            
            # Record by turn
            entities_by_turn.append({
                'role': message['role'],
                'entities': entities
            })
            
            triplets_by_turn.append({
                'role': message['role'],
                'triplets': triplets
            })
        
        return {
            'entities': list(all_entities),
            'triplets': all_triplets,
            'entities_by_turn': entities_by_turn,
            'triplets_by_turn': triplets_by_turn
        }
    
    def _extract_entities_rule_based(self, text: str) -> List[str]:
        """
        Extract entities using rule-based approach.
        
        Args:
            text: The text to extract entities from
            
        Returns:
            List of extracted entities
        """
        entities = []
        
        # Extract capitalized phrases (potential proper nouns)
        capitalized_pattern = r'\b[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*\b'
        capitalized_matches = re.findall(capitalized_pattern, text)
        entities.extend(capitalized_matches)
        
        # Extract quoted phrases
        quoted_pattern = r'"([^"]+)"'
        quoted_matches = re.findall(quoted_pattern, text)
        entities.extend(quoted_matches)
        
        # Extract technical terms and potential product names
        tech_pattern = r'\b[A-Za-z]+-?[0-9]+\b|\b[A-Z][A-Za-z]*[0-9]+\b'
        tech_matches = re.findall(tech_pattern, text)
        entities.extend(tech_matches)
        
        return entities
    
    def _extract_entities_spacy(self, text: str) -> List[str]:
        """
        Extract entities using spaCy NER.
        
        Args:
            text: The text to extract entities from
            
        Returns:
            List of extracted entities
        """
        if not self.nlp:
            return []
            
        doc = self.nlp(text)
        entities = []
        
        # Extract named entities
        for ent in doc.ents:
            entities.append(ent.text)
        
        # Extract noun chunks as potential entities
        for chunk in doc.noun_chunks:
            # Only add chunks that are not stopwords or single pronouns
            if not (len(chunk) == 1 and (chunk[0].is_stop or chunk[0].pos_ == 'PRON')):
                entities.append(chunk.text)
        
        return entities
    
    def _extract_entities_llm(self, text: str) -> List[str]:
        """
        Extract entities using LLM prompt engineering.
        
        Args:
            text: The text to extract entities from
            
        Returns:
            List of extracted entities
        """
        if not self.llm_client:
            return []
            
        try:
            # Define prompt for entity extraction
            prompt = f"""
            Extract the main entities from the following text. Include people, organizations, 
            products, locations, concepts, and other important nouns. Return only a list 
            of entities, one per line, without numbers or explanations.
            
            Text: {text}
            
            Entities:
            """
            
            # Send to LLM
            response = self.llm_client.completions.create(
                prompt=prompt,
                max_tokens=200,
                temperature=0.0,
                stop=["Text:", "---"]
            )
            
            # Parse response
            result = response.choices[0].text.strip()
            entities = [line.strip() for line in result.split('\n') if line.strip()]
            
            return entities
            
        except Exception as e:
            logger.error(f"Error extracting entities with LLM: {e}")
            return []
    
    def _extract_triplets_rule_based(self, text: str) -> List[Tuple[str, str, str]]:
        """
        Extract relationship triplets using rule-based patterns.
        
        Args:
            text: The text to extract triplets from
            
        Returns:
            List of (subject, relation, object) tuples
        """
        triplets = []
        
        # Simple SVO pattern matching
        patterns = [
            # "<entity> is a/an <entity>"
            r'([A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*)\s+is\s+(?:a|an)\s+([a-zA-Z]+(?:\s+[a-zA-Z]+)*)',
            # "<entity> is <adjective>"
            r'([A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*)\s+is\s+([a-zA-Z]+)',
            # "<entity> <verb> <entity>"
            r'([A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*)\s+([a-zA-Z]+(?:ed|s)?)\s+([A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*)',
            # "<entity> <verb> by <entity>"
            r'([A-Za-z]+(?:\s+[A-Za-z]+)*)\s+(?:was|is|are|were)\s+([a-zA-Z]+(?:ed)?)\s+by\s+([A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if len(match) == 2:  # Binary relation
                    subject, obj = match
                    relation = "is"
                    triplets.append((subject, relation, obj))
                elif len(match) == 3:  # Ternary relation
                    subject, relation, obj = match
                    triplets.append((subject, relation, obj))
        
        return triplets
    
    def _extract_triplets_spacy(self, text: str) -> List[Tuple[str, str, str]]:
        """
        Extract relationship triplets using spaCy dependency parsing.
        
        Args:
            text: The text to extract triplets from
            
        Returns:
            List of (subject, relation, object) tuples
        """
        if not self.nlp:
            return []
            
        doc = self.nlp(text)
        triplets = []
        
        for sent in doc.sents:
            # Extract subject-verb-object patterns
            root = None
            subject = None
            dobj = None
            
            # Find the root verb
            for token in sent:
                if token.dep_ == "ROOT" and token.pos_ == "VERB":
                    root = token
                    break
            
            if not root:
                continue
                
            # Find subject and object connected to the root
            subject_span = None
            object_span = None
            
            for child in root.children:
                # Find subject
                if child.dep_ in ("nsubj", "nsubjpass"):
                    subject_token = child
                    # Get the full noun phrase
                    subject_span = self._get_span_for_token(child, sent)
                
                # Find direct object
                elif child.dep_ in ("dobj", "pobj"):
                    object_token = child
                    # Get the full noun phrase
                    object_span = self._get_span_for_token(child, sent)
            
            # If we found both subject and object, add the triplet
            if subject_span and object_span:
                triplets.append((
                    subject_span.text,
                    root.text,
                    object_span.text
                ))
        
        return triplets
    
    def _get_span_for_token(self, token, sent):
        """Helper method to get a noun phrase span for a token"""
        # Start with the token itself
        start, end = token.i, token.i + 1
        
        # Expand left to include compounds and modifiers
        for left_token in reversed(list(token.lefts)):
            if left_token.dep_ in ("compound", "amod", "det") and left_token.i >= sent.start:
                start = min(start, left_token.i)
        
        # Expand right to include any attached prepositional phrases or compounds
        for right_token in token.rights:
            if right_token.dep_ in ("compound", "prep") and right_token.i < sent.end:
                end = max(end, right_token.i + 1)
                
                # If this is a preposition, include its object phrase too
                if right_token.dep_ == "prep":
                    for prep_object in right_token.children:
                        if prep_object.dep_ == "pobj":
                            obj_span = self._get_span_for_token(prep_object, sent)
                            end = max(end, obj_span.end)
        
        return token.doc[start:end]
    
    def _extract_triplets_llm(self, text: str) -> List[Tuple[str, str, str]]:
        """
        Extract relationship triplets using LLM prompt engineering.
        
        Args:
            text: The text to extract triplets from
            
        Returns:
            List of (subject, relation, object) tuples
        """
        if not self.llm_client:
            return []
            
        try:
            # Define prompt for triplet extraction
            prompt = f"""
            Extract relationship triplets from the following text in the format:
            (subject, relation, object)
            
            For example, from "John visited Paris last summer", extract:
            (John, visited, Paris)
            
            Text: {text}
            
            Triplets:
            """
            
            # Send to LLM
            response = self.llm_client.completions.create(
                prompt=prompt,
                max_tokens=250,
                temperature=0.0,
                stop=["Text:", "---"]
            )
            
            # Parse response
            result = response.choices[0].text.strip()
            
            # Extract triplets using regex pattern matching
            # Looking for patterns like: (subject, relation, object)
            triplet_pattern = r'\(([^,]+),\s*([^,]+),\s*([^)]+)\)'
            matches = re.findall(triplet_pattern, result)
            
            triplets = []
            for match in matches:
                if len(match) == 3:
                    subject = match[0].strip()
                    relation = match[1].strip()
                    obj = match[2].strip()
                    triplets.append((subject, relation, obj))
            
            return triplets
            
        except Exception as e:
            logger.error(f"Error extracting triplets with LLM: {e}")
            return []
    
    def _normalize_entity(self, entity: str) -> str:
        """
        Normalize an entity string for consistency.
        
        Args:
            entity: The entity string to normalize
            
        Returns:
            Normalized entity string
        """
        if not entity:
            return ""
            
        # Strip whitespace and quotes
        normalized = entity.strip().strip('"\'')
        
        # Remove trailing punctuation
        normalized = re.sub(r'[.,;:!?]+$', '', normalized)
        
        return normalized 