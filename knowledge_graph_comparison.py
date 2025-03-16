#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Knowledge Graph Comparison Visualizations

This script generates visualizations comparing knowledge graphs to traditional storage methods,
highlighting differences in data representation, storage efficiency, and query performance.
"""

import os
import json
import time
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

# Create output directory for visualizations
OUTPUT_DIR = "graph_comparisons"
os.makedirs(OUTPUT_DIR, exist_ok=True)

class KnowledgeGraphVisualizer:
    """Class for creating visualizations comparing knowledge graphs to traditional storage."""

    def __init__(self, output_dir=OUTPUT_DIR):
        """Initialize the visualizer."""
        self.output_dir = output_dir
        
    def create_storage_representation_comparison(self):
        """Create visualization comparing how data is represented in KG vs traditional storage."""
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # 1. Traditional Storage (Relational/Table-based) - Left side
        table_data = [
            ['1', 'Python', 'programming language', '2023-01-01'],
            ['2', 'Python', 'created by', 'Guido van Rossum'],
            ['3', 'TensorFlow', 'is a', 'machine learning library'],
            ['4', 'TensorFlow', 'used with', 'Python'],
            ['5', 'Python', 'has library', 'TensorFlow'],
            ['6', 'NLP', 'stands for', 'Natural Language Processing'],
            ['7', 'Python', 'used for', 'NLP'],
            ['8', 'JavaScript', 'runs in', 'browser'],
            ['9', 'HTML', 'used for', 'web development'],
            ['10', 'CSS', 'styles', 'web pages']
        ]
        
        table = ax1.table(
            cellText=table_data,
            colLabels=['ID', 'Subject', 'Relation', 'Object'],
            loc='center',
            cellLoc='center',
            colWidths=[0.1, 0.3, 0.3, 0.3]
        )
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        for key, cell in table._cells.items():
            if key[0] == 0:  # Header
                cell.set_text_props(weight='bold', color='white')
                cell.set_facecolor('#4472C4')
            else:
                cell.set_facecolor('#E6F0FF' if key[0] % 2 == 0 else '#C5D9F1')
        
        ax1.set_title('Traditional Storage (Relational Database)', fontsize=14)
        ax1.axis('off')
        
        # 2. Knowledge Graph Representation - Right side
        G = nx.DiGraph()
        
        # Add nodes
        nodes = {
            'Python': {'type': 'Technology'},
            'Guido van Rossum': {'type': 'Person'},
            'TensorFlow': {'type': 'Library'},
            'machine learning library': {'type': 'Concept'},
            'NLP': {'type': 'Concept'},
            'Natural Language Processing': {'type': 'Concept'},
            'JavaScript': {'type': 'Technology'},
            'browser': {'type': 'Environment'},
            'HTML': {'type': 'Technology'},
            'web development': {'type': 'Field'},
            'CSS': {'type': 'Technology'},
            'web pages': {'type': 'Artifact'}
        }
        
        for node, attrs in nodes.items():
            G.add_node(node, **attrs)
        
        # Add edges (relationships)
        edges = [
            ('Python', 'programming language', {'relation': 'is a'}),
            ('Python', 'Guido van Rossum', {'relation': 'created by'}),
            ('TensorFlow', 'machine learning library', {'relation': 'is a'}),
            ('TensorFlow', 'Python', {'relation': 'used with'}),
            ('Python', 'TensorFlow', {'relation': 'has library'}),
            ('NLP', 'Natural Language Processing', {'relation': 'stands for'}),
            ('Python', 'NLP', {'relation': 'used for'}),
            ('JavaScript', 'browser', {'relation': 'runs in'}),
            ('HTML', 'web development', {'relation': 'used for'}),
            ('CSS', 'web pages', {'relation': 'styles'})
        ]
        
        for src, tgt, attrs in edges:
            G.add_edge(src, tgt, **attrs)
        
        # Define node colors based on type
        color_map = {
            'Technology': '#00B050',  # Green
            'Person': '#FFC000',      # Yellow
            'Library': '#5B9BD5',     # Blue
            'Concept': '#ED7D31',     # Orange
            'Environment': '#7030A0',  # Purple
            'Field': '#C00000',       # Red
            'Artifact': '#525252'     # Gray
        }
        
        node_colors = [color_map.get(G.nodes[node].get('type', 'Other'), '#BFBFBF') for node in G.nodes()]
        
        # Draw the graph
        pos = nx.spring_layout(G, seed=42)  # Consistent layout
        
        nx.draw_networkx_nodes(G, pos, ax=ax2, node_size=2500, node_color=node_colors, edgecolors='black', linewidths=1)
        nx.draw_networkx_labels(G, pos, ax=ax2, font_size=9, font_weight='bold')
        
        # Draw edges with arrows
        nx.draw_networkx_edges(G, pos, ax=ax2, width=1.5, alpha=0.7, edge_color='gray', 
                              connectionstyle='arc3,rad=0.2', arrowsize=15)
        
        # Add edge labels
        edge_labels = {(src, tgt): data['relation'] for src, tgt, data in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, ax=ax2, edge_labels=edge_labels, font_size=8, 
                                     font_color='#404040', bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 3})
        
        # Create legend for node types
        legend_elements = [mpatches.Patch(color=color, label=node_type) for node_type, color in color_map.items()]
        ax2.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=4)
        
        ax2.set_title('Knowledge Graph Representation', fontsize=14)
        ax2.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'storage_representation_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Created storage representation comparison at {self.output_dir}/storage_representation_comparison.png")
    
    def create_query_performance_comparison(self):
        """Create visualization comparing query performance between KG and traditional storage."""
        query_types = ['Simple Query', 'Related Entities', 'Path Finding', 'Context Enrichment', 'Pattern Matching']
        
        # Mockup performance data (milliseconds)
        traditional_times = [50, 150, 450, 350, 500]
        kg_times = [60, 90, 120, 80, 100]
        
        x = np.arange(len(query_types))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 7))
        rects1 = ax.bar(x - width/2, traditional_times, width, label='Traditional Storage', color='#4472C4')
        rects2 = ax.bar(x + width/2, kg_times, width, label='Knowledge Graph', color='#70AD47')
        
        # Add labels and title
        ax.set_ylabel('Query Time (milliseconds)', fontsize=12)
        ax.set_title('Query Performance Comparison', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(query_types, fontsize=10)
        ax.legend(fontsize=12)
        
        # Add data labels on bars
        def add_labels(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height}ms',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom',
                            fontsize=9)
                
        add_labels(rects1)
        add_labels(rects2)
        
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add explanations below the chart
        explanations = [
            "Simple Query: Basic attribute lookup",
            "Related Entities: Finding directly connected entities",
            "Path Finding: Discovering relationships between entities",
            "Context Enrichment: Gathering context for understanding",
            "Pattern Matching: Finding complex patterns in data"
        ]
        
        explanation_text = '\n'.join(explanations)
        plt.figtext(0.5, 0.01, explanation_text, ha='center', fontsize=10, 
                   bbox={'facecolor':'#F2F2F2', 'alpha':0.5, 'pad':5})
        
        plt.tight_layout(rect=[0, 0.1, 1, 0.95])
        plt.savefig(os.path.join(self.output_dir, 'query_performance_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Created query performance comparison at {self.output_dir}/query_performance_comparison.png")
    
    def create_context_retrieval_comparison(self):
        """Create visualization showing how context is retrieved in both approaches."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Traditional approach - Sequential search through records
        table_data = [
            ['User asks about Python and machine learning'],
            ['Search database for "Python" → 3 records found'],
            ['Search database for "machine learning" → 2 records found'],
            ['Join results → 5 total records'],
            ['Run additional search for related terms'],
            ['Process results sequentially'],
            ['Format and return all matching records to user']
        ]
        
        for i, row in enumerate(table_data):
            ax1.text(0.1, 0.9 - (i * 0.125), row[0], fontsize=12, 
                    bbox=dict(facecolor='#C5D9F1', alpha=0.5, boxstyle='round,pad=0.5'))
            
            # Draw arrows connecting steps
            if i < len(table_data) - 1:
                ax1.annotate('', xy=(0.1, 0.875 - (i * 0.125)), xytext=(0.1, 0.925 - (i * 0.125)),
                            arrowprops=dict(facecolor='black', width=1, headwidth=8))
        
        ax1.set_title('Traditional Approach: Context Retrieval Process', fontsize=14)
        ax1.axis('off')
        
        # KG approach - Graph traversal and neighborhood search
        kg_process = [
            "User asks about Python and machine learning",
            "Identify 'Python' and 'machine learning' as entities",
            "Find nodes in knowledge graph",
            "Traverse connections (1-2 hops)",
            "Identify related concepts (TensorFlow, NLP, deep learning)",
            "Return connected subgraph with contextual relationships"
        ]
        
        # Create a small graph for illustration
        G = nx.DiGraph()
        
        # Core nodes
        G.add_node("Python", pos=(0.5, 0.5), size=2000, color='#5B9BD5')
        G.add_node("machine learning", pos=(0.8, 0.5), size=2000, color='#ED7D31')
        
        # Related nodes - first level
        G.add_node("TensorFlow", pos=(0.3, 0.7), size=1500, color='#4472C4')
        G.add_node("libraries", pos=(0.3, 0.3), size=1500, color='#70AD47')
        G.add_node("algorithms", pos=(1.0, 0.7), size=1500, color='#FFC000')
        G.add_node("data science", pos=(1.0, 0.3), size=1500, color='#A5A5A5')
        
        # Related nodes - second level
        G.add_node("deep learning", pos=(0.15, 0.85), size=1000, color='#C00000')
        G.add_node("NLP", pos=(0.15, 0.15), size=1000, color='#7030A0')
        G.add_node("neural networks", pos=(1.2, 0.85), size=1000, color='#00B050')
        G.add_node("statistics", pos=(1.2, 0.15), size=1000, color='#525252')
        
        # Add edges
        edges = [
            ("Python", "TensorFlow"), ("Python", "libraries"),
            ("machine learning", "algorithms"), ("machine learning", "data science"),
            ("TensorFlow", "deep learning"), ("libraries", "NLP"),
            ("algorithms", "neural networks"), ("data science", "statistics"),
            ("Python", "machine learning"), ("machine learning", "Python"),
            ("TensorFlow", "machine learning"), ("algorithms", "TensorFlow")
        ]
        
        for src, tgt in edges:
            G.add_edge(src, tgt)
        
        # 1. Draw the graph
        pos = nx.get_node_attributes(G, 'pos')
        sizes = [G.nodes[node].get('size', 1000) for node in G.nodes()]
        colors = [G.nodes[node].get('color', '#1f78b4') for node in G.nodes()]
        
        # Highlight core nodes
        highlighted_nodes = ["Python", "machine learning"]
        node_borders = ['red' if node in highlighted_nodes else 'black' for node in G.nodes()]
        border_widths = [3 if node in highlighted_nodes else 1 for node in G.nodes()]
        
        nx.draw_networkx_nodes(G, pos, ax=ax2, node_size=sizes, node_color=colors, 
                              edgecolors=node_borders, linewidths=border_widths)
        nx.draw_networkx_edges(G, pos, ax=ax2, width=1.5, alpha=0.7, edge_color='gray',
                              connectionstyle='arc3,rad=0.1', arrowsize=12)
        nx.draw_networkx_labels(G, pos, ax=ax2, font_size=10, font_weight='bold')
        
        # Add the process steps on the side
        for i, step in enumerate(kg_process):
            ax2.text(1.5, 0.9 - (i * 0.15), step, fontsize=12,
                    bbox=dict(facecolor='#E2EFDA', alpha=0.5, boxstyle='round,pad=0.5'))
            
            # Draw connecting arrows between steps
            if i < len(kg_process) - 1:
                ax2.annotate('', xy=(1.5, 0.875 - (i * 0.15)), xytext=(1.5, 0.925 - (i * 0.15)),
                            arrowprops=dict(facecolor='black', width=1, headwidth=8))
        
        ax2.set_title('Knowledge Graph Approach: Context Traversal', fontsize=14)
        ax2.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'context_retrieval_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Created context retrieval comparison at {self.output_dir}/context_retrieval_comparison.png")

    def create_data_growth_comparison(self):
        """Create visualization comparing how data grows in both approaches."""
        # Data points for growth comparison
        entities = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
        
        # Traditional storage (assume linear growth in records)
        trad_records = [e * 5 for e in entities]  # 5 records per entity on average
        
        # Knowledge graph (sub-linear growth due to shared connections)
        kg_records = [int(e * 3.5 * (1 - 0.0001 * e)) for e in entities]  # Diminishing returns formula
        
        # Storage size (KB)
        trad_size = [r * 0.5 for r in trad_records]  # 0.5 KB per record
        kg_size = [e * 0.3 + (e * 3.2) * 0.1 for e in entities]  # 0.3 KB per entity + 0.1 KB per relationship
        
        # Create the figure and subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # Number of Records plot
        ax1.plot(entities, trad_records, marker='o', linewidth=2, markersize=8, label='Traditional Storage Records')
        ax1.plot(entities, kg_records, marker='s', linewidth=2, markersize=8, label='Knowledge Graph Triplets')
        ax1.set_xlabel('Number of Entities', fontsize=12)
        ax1.set_ylabel('Number of Records/Triplets', fontsize=12)
        ax1.set_title('Data Growth: Records vs. Entities', fontsize=14)
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend(fontsize=10)
        
        # Storage Size plot
        ax2.plot(entities, trad_size, marker='o', linewidth=2, markersize=8, label='Traditional Storage Size (KB)')
        ax2.plot(entities, kg_size, marker='s', linewidth=2, markersize=8, label='Knowledge Graph Storage Size (KB)')
        ax2.set_xlabel('Number of Entities', fontsize=12)
        ax2.set_ylabel('Storage Size (KB)', fontsize=12)
        ax2.set_title('Storage Efficiency Comparison', fontsize=14)
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend(fontsize=10)
        
        # Add explanation text
        explanation = (
            "Knowledge graphs can be more efficient as entity count grows because:\n"
            "1. Entities are stored only once regardless of how many relationships they have\n"
            "2. Relationships are explicit triplets (subject-predicate-object) rather than denormalized records\n"
            "3. Data deduplication happens naturally in the graph structure"
        )
        
        plt.figtext(0.5, 0.01, explanation, ha='center', fontsize=11, 
                   bbox={'facecolor':'#F2F2F2', 'alpha':0.5, 'pad':5})
        
        plt.tight_layout(rect=[0, 0.1, 1, 0.95])
        plt.savefig(os.path.join(self.output_dir, 'data_growth_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Created data growth comparison at {self.output_dir}/data_growth_comparison.png")

    def create_context_quality_comparison(self):
        """Create visualization comparing context quality between approaches."""
        # Quality metrics
        metrics = ['Relevance', 'Completeness', 'Connection Discovery', 'Inference Capability', 'Context Depth']
        
        # Mock scores for comparison (0-10 scale)
        traditional_scores = [7, 5, 3, 2, 4]
        kg_scores = [8, 8, 9, 7, 9]
        
        # Create radar chart
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Close the loop
        
        traditional_scores += traditional_scores[:1]  # Close the loop
        kg_scores += kg_scores[:1]  # Close the loop
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        # Plot data
        ax.plot(angles, traditional_scores, 'o-', linewidth=2, label='Traditional Storage', color='#4472C4')
        ax.fill(angles, traditional_scores, alpha=0.25, color='#4472C4')
        
        ax.plot(angles, kg_scores, 'o-', linewidth=2, label='Knowledge Graph', color='#70AD47')
        ax.fill(angles, kg_scores, alpha=0.25, color='#70AD47')
        
        # Set category labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, fontsize=12)
        
        # Set y-axis limits
        ax.set_ylim(0, 10)
        ax.set_yticks(range(0, 11, 2))
        ax.set_yticklabels([str(i) for i in range(0, 11, 2)], fontsize=10)
        
        # Add legend
        ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=12)
        
        # Title
        plt.title('Context Quality Comparison', fontsize=16, y=1.08)
        
        # Add descriptions for each metric
        descriptions = {
            'Relevance': 'How well the system identifies truly relevant information',
            'Completeness': 'How comprehensive the returned context is',
            'Connection Discovery': 'Ability to find non-obvious connections between entities',
            'Inference Capability': 'Ability to infer new knowledge from existing facts',
            'Context Depth': 'How many layers of context are discoverable'
        }
        
        # Add description text
        desc_text = "\n".join([f"{m}: {descriptions[m]}" for m in metrics])
        plt.figtext(0.5, 0.01, desc_text, ha='center', fontsize=11, 
                   bbox={'facecolor':'#F2F2F2', 'alpha':0.5, 'pad':5})
        
        plt.tight_layout(rect=[0, 0.1, 1, 0.95])
        plt.savefig(os.path.join(self.output_dir, 'context_quality_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Created context quality comparison at {self.output_dir}/context_quality_comparison.png")

def main():
    """Run all visualizations."""
    print("Creating knowledge graph comparison visualizations...")
    visualizer = KnowledgeGraphVisualizer()
    
    visualizer.create_storage_representation_comparison()
    visualizer.create_query_performance_comparison()
    visualizer.create_context_retrieval_comparison()
    visualizer.create_data_growth_comparison()
    visualizer.create_context_quality_comparison()
    
    print(f"All visualizations saved to the '{OUTPUT_DIR}' directory")

if __name__ == "__main__":
    main() 