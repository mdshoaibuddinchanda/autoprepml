"""Graph data preprocessing module for AutoPrepML"""

from collections import deque
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd


class GraphPrepML:
    """Graph data preprocessing class.

    Example:
        >>> prep = GraphPrepML(nodes_df=nodes, edges_df=edges,
                               node_id_col='id', source_col='source', target_col='target')
        >>> clean_nodes, clean_edges = prep.clean()
        >>> prep.save_report('graph_report.html')
    """

    def __init__(
        self,
        nodes_df: Optional[pd.DataFrame] = None,
        edges_df: Optional[pd.DataFrame] = None,
        node_id_col: str = "id",
        source_col: str = "source",
        target_col: str = "target",
        directed: bool = True,
    ):
        """Initialize GraphPrepML.

        Args:
            nodes_df: DataFrame with node data
            edges_df: DataFrame with edge data
            node_id_col: Column name for node IDs
            source_col: Column name for edge source nodes
            target_col: Column name for edge target nodes
            directed: Whether edge direction is significant. Defaults to True.
        """
        self.nodes_df = nodes_df.copy() if nodes_df is not None else None
        self.edges_df = edges_df.copy() if edges_df is not None else None
        self.original_nodes = nodes_df.copy() if nodes_df is not None else None
        self.original_edges = edges_df.copy() if edges_df is not None else None

        self.node_id_col = node_id_col
        self.source_col = source_col
        self.target_col = target_col
        if not isinstance(directed, bool):
            raise TypeError("directed must be a boolean")
        self.directed = directed
        self.log = []

        # Validation
        if self.nodes_df is not None and node_id_col not in self.nodes_df.columns:
            raise ValueError(f"Column '{node_id_col}' not found in nodes DataFrame")

        if self.edges_df is not None:
            if source_col not in self.edges_df.columns:
                raise ValueError(f"Column '{source_col}' not found in edges DataFrame")
            if target_col not in self.edges_df.columns:
                raise ValueError(f"Column '{target_col}' not found in edges DataFrame")

    def _edge_keys(self) -> pd.Series:
        """Return direction-aware keys used for duplicate and multi-edge logic."""
        if self.edges_df is None:
            return pd.Series(dtype=object)

        pairs = self.edges_df[[self.source_col, self.target_col]].itertuples(index=False, name=None)
        if self.directed:
            keys = list(pairs)
        else:
            keys = [
                tuple(
                    sorted((source, target), key=lambda value: (type(value).__name__, repr(value)))
                )
                for source, target in pairs
            ]
        return pd.Series(keys, index=self.edges_df.index, dtype=object)

    def detect_issues(self) -> Dict[str, Any]:
        """Detect graph data quality issues.

        Returns:
            Dictionary with detected issues
        """
        issues = {}

        if self.nodes_df is not None:
            node_ids = set(self.nodes_df[self.node_id_col])

            issues["nodes"] = {
                "total_nodes": len(self.nodes_df),
                "duplicate_node_ids": int(self.nodes_df[self.node_id_col].duplicated().sum()),
                "missing_node_ids": int(self.nodes_df[self.node_id_col].isnull().sum()),
            }
        else:
            node_ids = set()
            issues["nodes"] = {"message": "No nodes DataFrame provided"}

        if self.edges_df is not None:
            source_ids = set(self.edges_df[self.source_col].dropna())
            target_ids = set(self.edges_df[self.target_col].dropna())

            duplicate_edges = self._edge_keys().duplicated().sum()
            issues["edges"] = {
                "total_edges": len(self.edges_df),
                "duplicate_edges": int(duplicate_edges),
                "self_loops": int(
                    (self.edges_df[self.source_col] == self.edges_df[self.target_col]).sum()
                ),
                "missing_sources": int(self.edges_df[self.source_col].isnull().sum()),
                "missing_targets": int(self.edges_df[self.target_col].isnull().sum()),
            }

            # Check for dangling edges (edges referencing non-existent nodes)
            if self.nodes_df is not None:
                dangling_sources = source_ids - node_ids
                dangling_targets = target_ids - node_ids
                issues["edges"]["dangling_edges"] = len(dangling_sources) + len(dangling_targets)

        else:
            issues["edges"] = {"message": "No edges DataFrame provided"}

        self.log.append({"action": "detect_issues", "result": issues})
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
        self.nodes_df = self.nodes_df.drop_duplicates(subset=[self.node_id_col], keep="first")

        removed = original_len - len(self.nodes_df)
        self.log.append({"action": "validate_node_ids", "removed": removed})
        return self.nodes_df

    def validate_edges(
        self, remove_self_loops: bool = True, remove_dangling: bool = True
    ) -> pd.DataFrame:
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
            self.edges_df = self.edges_df[
                self.edges_df[self.source_col] != self.edges_df[self.target_col]
            ]

        # Remove dangling edges
        if remove_dangling and self.nodes_df is not None:
            valid_nodes = set(self.nodes_df[self.node_id_col])
            self.edges_df = self.edges_df[
                (self.edges_df[self.source_col].isin(valid_nodes))
                & (self.edges_df[self.target_col].isin(valid_nodes))
            ]

        removed = original_len - len(self.edges_df)
        self.log.append({"action": "validate_edges", "removed": removed})
        return self.edges_df

    def remove_duplicate_edges(self, keep: str = "first") -> pd.DataFrame:
        """Remove duplicate edges.

        Args:
            keep: 'first', 'last', or False

        Returns:
            Edges DataFrame with duplicates removed
        """
        if self.edges_df is None:
            raise ValueError("No edges DataFrame provided")

        if keep not in ("first", "last", False):
            raise ValueError("keep must be 'first', 'last', or False")

        original_len = len(self.edges_df)
        duplicate_mask = self._edge_keys().duplicated(keep=keep)
        self.edges_df = self.edges_df.loc[~duplicate_mask].copy()
        removed = original_len - len(self.edges_df)

        self.log.append({"action": "remove_duplicate_edges", "removed": removed})
        return self.edges_df

    def add_node_features(self) -> pd.DataFrame:
        """Extract node-level features (degree centrality, etc.).

        Returns:
            Nodes DataFrame with added features
        """
        if self.nodes_df is None or self.edges_df is None:
            raise ValueError("Both nodes and edges DataFrames required")

        node_ids = self.nodes_df[self.node_id_col]

        if self.directed:
            out_degree = self.edges_df.groupby(self.source_col).size()
            in_degree = self.edges_df.groupby(self.target_col).size()

            self.nodes_df["out_degree"] = node_ids.map(out_degree).fillna(0).astype(int)
            self.nodes_df["in_degree"] = node_ids.map(in_degree).fillna(0).astype(int)
            self.nodes_df["total_degree"] = self.nodes_df["out_degree"] + self.nodes_df["in_degree"]
        else:
            incident = pd.concat(
                [self.edges_df[self.source_col], self.edges_df[self.target_col]],
                ignore_index=True,
            ).value_counts()
            degree = node_ids.map(incident).fillna(0).astype(int)
            self.nodes_df["degree"] = degree
            # Keep the directed column names as compatible aliases while
            # reporting a single incident degree for undirected graphs.
            self.nodes_df["out_degree"] = degree
            self.nodes_df["in_degree"] = degree
            self.nodes_df["total_degree"] = degree

        # Identify isolated nodes
        self.nodes_df["is_isolated"] = self.nodes_df["total_degree"] == 0

        self.log.append({"action": "add_node_features", "features": 4})
        return self.nodes_df

    def add_edge_features(self) -> pd.DataFrame:
        """Extract edge-level features.

        Returns:
            Edges DataFrame with added features
        """
        if self.edges_df is None:
            raise ValueError("No edges DataFrame provided")

        # Count multi-edges using the same direction semantics as duplicate
        # detection and removal.
        edge_keys = self._edge_keys()
        edge_counts = edge_keys.value_counts()
        self.edges_df["edge_count"] = edge_keys.map(edge_counts).astype(int)

        self.log.append({"action": "add_edge_features", "features": 1})
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

            adjacency.setdefault(source, set()).add(target)
            adjacency.setdefault(target, set()).add(source)

        # BFS to find components
        visited = set()
        component_id = 0
        node_to_component = {}

        for node_id in self.nodes_df[self.node_id_col]:
            if node_id not in visited:
                # Start new component
                queue = deque([node_id])
                visited.add(node_id)

                while queue:
                    current = queue.popleft()
                    node_to_component[current] = component_id

                    if current in adjacency:
                        for neighbor in adjacency[current]:
                            if neighbor not in visited:
                                visited.add(neighbor)
                                queue.append(neighbor)

                component_id += 1

        self.nodes_df["component_id"] = (
            self.nodes_df[self.node_id_col].map(node_to_component).fillna(-1).astype(int)
        )

        self.log.append({"action": "identify_components", "num_components": component_id})
        return self.nodes_df

    def filter_isolated_nodes(self) -> pd.DataFrame:
        """Remove nodes with no edges.

        Returns:
            Filtered nodes DataFrame
        """
        if self.nodes_df is None or self.edges_df is None:
            raise ValueError("Both nodes and edges DataFrames required")

        # Add node features if not already present
        if "is_isolated" not in self.nodes_df.columns:
            self.add_node_features()

        original_len = len(self.nodes_df)
        self.nodes_df = self.nodes_df[~self.nodes_df["is_isolated"]]
        removed = original_len - len(self.nodes_df)

        self.log.append({"action": "filter_isolated_nodes", "removed": removed})
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

            adjacency.setdefault(source, [])
            if self.directed or target not in adjacency[source]:
                adjacency[source].append(target)
            if not self.directed:
                adjacency.setdefault(target, [])
                if source not in adjacency[target]:
                    adjacency[target].append(source)

        return adjacency

    def get_graph_stats(self) -> Dict[str, Any]:
        """Get overall graph statistics.

        Returns:
            Dictionary with graph metrics
        """
        stats = {}

        stats["directed"] = self.directed

        if self.nodes_df is not None:
            stats["num_nodes"] = len(self.nodes_df)

        if self.edges_df is not None:
            stats["num_edges"] = len(self.edges_df)

            if self.nodes_df is not None:
                denominator = len(self.nodes_df) * (len(self.nodes_df) - 1)
                if denominator:
                    multiplier = 1 if self.directed else 2
                    stats["density"] = multiplier * len(self.edges_df) / denominator
                else:
                    stats["density"] = 0

            # Average degree
            if self.nodes_df is not None and "total_degree" in self.nodes_df.columns:
                stats["avg_degree"] = float(self.nodes_df["total_degree"].mean())

        return stats

    def report(self) -> Dict[str, Any]:
        """Generate preprocessing report.

        Returns:
            Report dictionary
        """
        return {
            "original_nodes_shape": (
                self.original_nodes.shape if self.original_nodes is not None else None
            ),
            "current_nodes_shape": self.nodes_df.shape if self.nodes_df is not None else None,
            "original_edges_shape": (
                self.original_edges.shape if self.original_edges is not None else None
            ),
            "current_edges_shape": self.edges_df.shape if self.edges_df is not None else None,
            "logs": self.log,
            "issues": self.detect_issues(),
            "graph_stats": self.get_graph_stats(),
        }

    def save_report(self, output_path: str) -> None:
        """Save preprocessing report to file.

        Args:
            output_path: Path to save report (supports .json, .html)
        """
        from .reports import generate_json_report, generate_html_report

        report = self.report()

        if output_path.endswith(".json"):
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(generate_json_report(report))
        elif output_path.endswith(".html"):
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(generate_html_report(report))
        else:
            raise ValueError("Output path must end with .json or .html")
