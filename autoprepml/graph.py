"""Graph data preprocessing module for AutoPrepML"""
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np


class GraphPrepML:
    """Graph data preprocessing class.
    
    Example:
        >>> prep = GraphPrepML(nodes_df=nodes, edges_df=edges, 
                               node_id_col='id', source_col='source', target_col='target')
        >>> clean_nodes, clean_edges = prep.clean()
        >>> prep.save_report('graph_report.html')
    """
    
    def __init__(self, 
                 nodes_df: Optional[pd.DataFrame] = None,
                 edges_df: Optional[pd.DataFrame] = None,
                 node_id_col: str = 'id',
                 source_col: str = 'source',
                 target_col: str = 'target'):
        """Initialize GraphPrepML.
        
        Args:
            nodes_df: DataFrame with node data
            edges_df: DataFrame with edge data
            node_id_col: Column name for node IDs
            source_col: Column name for edge source nodes
            target_col: Column name for edge target nodes
        """
        self.nodes_df = nodes_df.copy() if nodes_df is not None else None
        self.edges_df = edges_df.copy() if edges_df is not None else None
        self.original_nodes = nodes_df.copy() if nodes_df is not None else None
        self.original_edges = edges_df.copy() if edges_df is not None else None
        
        self.node_id_col = node_id_col
        self.source_col = source_col
        self.target_col = target_col
        self.log = []
        
        # Validation
        if self.nodes_df is not None and node_id_col not in self.nodes_df.columns:
            raise ValueError(f"Column '{node_id_col}' not found in nodes DataFrame")
        
        if self.edges_df is not None:
            if source_col not in self.edges_df.columns:
                raise ValueError(f"Column '{source_col}' not found in edges DataFrame")
            if target_col not in self.edges_df.columns:
                raise ValueError(f"Column '{target_col}' not found in edges DataFrame")
    
    def detect_issues(self) -> Dict[str, Any]:
        """Detect graph data quality issues.
        
        Returns:
            Dictionary with detected issues
        """
        issues = {}
        
        if self.nodes_df is not None:
            node_ids = set(self.nodes_df[self.node_id_col])
            
            issues['nodes'] = {
                'total_nodes': len(self.nodes_df),
                'duplicate_node_ids': int(self.nodes_df[self.node_id_col].duplicated().sum()),
                'missing_node_ids': int(self.nodes_df[self.node_id_col].isnull().sum()),
            }
        else:
            node_ids = set()
            issues['nodes'] = {'message': 'No nodes DataFrame provided'}
        
        if self.edges_df is not None:
            source_ids = set(self.edges_df[self.source_col].dropna())
            target_ids = set(self.edges_df[self.target_col].dropna())
            all_edge_nodes = source_ids.union(target_ids)
            
            issues['edges'] = {
                'total_edges': len(self.edges_df),
                'duplicate_edges': int(self.edges_df.duplicated(subset=[self.source_col, self.target_col]).sum()),
                'self_loops': int((self.edges_df[self.source_col] == self.edges_df[self.target_col]).sum()),
                'missing_sources': int(self.edges_df[self.source_col].isnull().sum()),
                'missing_targets': int(self.edges_df[self.target_col].isnull().sum()),
            }
            
            # Check for dangling edges (edges referencing non-existent nodes)
            if self.nodes_df is not None:
                dangling_sources = source_ids - node_ids
                dangling_targets = target_ids - node_ids
                issues['edges']['dangling_edges'] = len(dangling_sources) + len(dangling_targets)
            
        else:
            issues['edges'] = {'message': 'No edges DataFrame provided'}
        
        self.log.append({'action': 'detect_issues', 'result': issues})
        return issues
    
    def validate_node_ids(self) -> pd.DataFrame:
        """Validate and clean node IDs.
        
        Returns:
            Cleaned nodes DataFrame
        """
        if self.nodes_df is None:
            raise ValueError("No nodes DataFrame provided")
        
        original_len = len(self.nodes_df)
        
        # Remove rows with missing node IDs
        self.nodes_df = self.nodes_df.dropna(subset=[self.node_id_col])
        
        # Remove duplicate node IDs (keep first)
        self.nodes_df = self.nodes_df.drop_duplicates(subset=[self.node_id_col], keep='first')
        
        removed = original_len - len(self.nodes_df)
        self.log.append({'action': 'validate_node_ids', 'removed': removed})
        return self.nodes_df
    
    def validate_edges(self, remove_self_loops: bool = True, remove_dangling: bool = True) -> pd.DataFrame:
        """Validate and clean edges.
        
        Args:
            remove_self_loops: Remove self-referencing edges
            remove_dangling: Remove edges referencing non-existent nodes
            
        Returns:
            Cleaned edges DataFrame
        """
        if self.edges_df is None:
            raise ValueError("No edges DataFrame provided")
        
        original_len = len(self.edges_df)
        
        # Remove rows with missing source or target
        self.edges_df = self.edges_df.dropna(subset=[self.source_col, self.target_col])
        
        # Remove self-loops
        if remove_self_loops:
            self.edges_df = self.edges_df[self.edges_df[self.source_col] != self.edges_df[self.target_col]]
        
        # Remove dangling edges
        if remove_dangling and self.nodes_df is not None:
            valid_nodes = set(self.nodes_df[self.node_id_col])
            self.edges_df = self.edges_df[
                (self.edges_df[self.source_col].isin(valid_nodes)) &
                (self.edges_df[self.target_col].isin(valid_nodes))
            ]
        
        removed = original_len - len(self.edges_df)
        self.log.append({'action': 'validate_edges', 'removed': removed})
        return self.edges_df
    
    def remove_duplicate_edges(self, keep: str = 'first') -> pd.DataFrame:
        """Remove duplicate edges.
        
        Args:
            keep: 'first', 'last', or False
            
        Returns:
            Edges DataFrame with duplicates removed
        """
        if self.edges_df is None:
            raise ValueError("No edges DataFrame provided")
        
        original_len = len(self.edges_df)
        self.edges_df = self.edges_df.drop_duplicates(
            subset=[self.source_col, self.target_col], 
            keep=keep
        )
        removed = original_len - len(self.edges_df)
        
        self.log.append({'action': 'remove_duplicate_edges', 'removed': removed})
        return self.edges_df
    
    def add_node_features(self) -> pd.DataFrame:
        """Extract node-level features (degree centrality, etc.).
        
        Returns:
            Nodes DataFrame with added features
        """
        if self.nodes_df is None or self.edges_df is None:
            raise ValueError("Both nodes and edges DataFrames required")
        
        node_ids = self.nodes_df[self.node_id_col]
        
        # Calculate degrees
        out_degree = self.edges_df.groupby(self.source_col).size()
        in_degree = self.edges_df.groupby(self.target_col).size()
        
        self.nodes_df['out_degree'] = node_ids.map(out_degree).fillna(0).astype(int)
        self.nodes_df['in_degree'] = node_ids.map(in_degree).fillna(0).astype(int)
        self.nodes_df['total_degree'] = self.nodes_df['out_degree'] + self.nodes_df['in_degree']
        
        # Identify isolated nodes
        self.nodes_df['is_isolated'] = self.nodes_df['total_degree'] == 0
        
        self.log.append({'action': 'add_node_features', 'features': 4})
        return self.nodes_df
    
    def add_edge_features(self) -> pd.DataFrame:
        """Extract edge-level features.
        
        Returns:
            Edges DataFrame with added features
        """
        if self.edges_df is None:
            raise ValueError("No edges DataFrame provided")
        
        # Count multi-edges (same source and target pairs)
        edge_counts = self.edges_df.groupby([self.source_col, self.target_col]).size()
        self.edges_df['edge_count'] = self.edges_df.apply(
            lambda row: edge_counts.get((row[self.source_col], row[self.target_col]), 1),
            axis=1
        )
        
        self.log.append({'action': 'add_edge_features', 'features': 1})
        return self.edges_df
    
    def identify_components(self) -> pd.DataFrame:
        """Identify connected components (simple BFS).
        
        Returns:
            Nodes DataFrame with component IDs
        """
        if self.nodes_df is None or self.edges_df is None:
            raise ValueError("Both nodes and edges DataFrames required")
        
        # Build adjacency list
        adjacency = {}
        for _, row in self.edges_df.iterrows():
            source = row[self.source_col]
            target = row[self.target_col]
            
            if source not in adjacency:
                adjacency[source] = []
            if target not in adjacency:
                adjacency[target] = []
            
            adjacency[source].append(target)
            adjacency[target].append(source)
        
        # BFS to find components
        visited = set()
        component_id = 0
        node_to_component = {}
        
        for node_id in self.nodes_df[self.node_id_col]:
            if node_id not in visited:
                # Start new component
                queue = [node_id]
                visited.add(node_id)
                
                while queue:
                    current = queue.pop(0)
                    node_to_component[current] = component_id
                    
                    if current in adjacency:
                        for neighbor in adjacency[current]:
                            if neighbor not in visited:
                                visited.add(neighbor)
                                queue.append(neighbor)
                
                component_id += 1
        
        self.nodes_df['component_id'] = self.nodes_df[self.node_id_col].map(node_to_component).fillna(-1).astype(int)
        
        self.log.append({'action': 'identify_components', 'num_components': component_id})
        return self.nodes_df
    
    def filter_isolated_nodes(self) -> pd.DataFrame:
        """Remove nodes with no edges.
        
        Returns:
            Filtered nodes DataFrame
        """
        if self.nodes_df is None or self.edges_df is None:
            raise ValueError("Both nodes and edges DataFrames required")
        
        # Add node features if not already present
        if 'is_isolated' not in self.nodes_df.columns:
            self.add_node_features()
        
        original_len = len(self.nodes_df)
        self.nodes_df = self.nodes_df[~self.nodes_df['is_isolated']]
        removed = original_len - len(self.nodes_df)
        
        self.log.append({'action': 'filter_isolated_nodes', 'removed': removed})
        return self.nodes_df
    
    def to_edge_list(self) -> List[Tuple[Any, Any]]:
        """Convert edges DataFrame to edge list.
        
        Returns:
            List of (source, target) tuples
        """
        if self.edges_df is None:
            raise ValueError("No edges DataFrame provided")
        
        return list(zip(self.edges_df[self.source_col], self.edges_df[self.target_col]))
    
    def to_adjacency_dict(self) -> Dict[Any, List[Any]]:
        """Convert to adjacency dictionary representation.
        
        Returns:
            Dictionary mapping node IDs to lists of neighbors
        """
        if self.edges_df is None:
            raise ValueError("No edges DataFrame provided")
        
        adjacency = {}
        for _, row in self.edges_df.iterrows():
            source = row[self.source_col]
            target = row[self.target_col]
            
            if source not in adjacency:
                adjacency[source] = []
            adjacency[source].append(target)
        
        return adjacency
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """Get overall graph statistics.
        
        Returns:
            Dictionary with graph metrics
        """
        stats = {}
        
        if self.nodes_df is not None:
            stats['num_nodes'] = len(self.nodes_df)
        
        if self.edges_df is not None:
            stats['num_edges'] = len(self.edges_df)
            
            if self.nodes_df is not None:
                stats['density'] = (2 * len(self.edges_df)) / (len(self.nodes_df) * (len(self.nodes_df) - 1)) if len(self.nodes_df) > 1 else 0
            
            # Average degree
            if 'total_degree' in self.nodes_df.columns:
                stats['avg_degree'] = float(self.nodes_df['total_degree'].mean())
        
        return stats
    
    def report(self) -> Dict[str, Any]:
        """Generate preprocessing report.
        
        Returns:
            Report dictionary
        """
        return {
            'original_nodes_shape': self.original_nodes.shape if self.original_nodes is not None else None,
            'current_nodes_shape': self.nodes_df.shape if self.nodes_df is not None else None,
            'original_edges_shape': self.original_edges.shape if self.original_edges is not None else None,
            'current_edges_shape': self.edges_df.shape if self.edges_df is not None else None,
            'logs': self.log,
            'issues': self.detect_issues(),
            'graph_stats': self.get_graph_stats()
        }
    
    def save_report(self, output_path: str) -> None:
        """Save preprocessing report to file.
        
        Args:
            output_path: Path to save report (supports .json, .html)
        """
        from .reports import generate_json_report, generate_html_report
        
        report = self.report()
        
        if output_path.endswith('.json'):
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(generate_json_report(report))
        elif output_path.endswith('.html'):
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(generate_html_report(report))
        else:
            raise ValueError("Output path must end with .json or .html")
