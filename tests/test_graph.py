"""Tests for graph data preprocessing module"""
import pytest
import pandas as pd
from autoprepml.graph import GraphPrepML


@pytest.fixture
def sample_nodes():
    """Create sample nodes DataFrame"""
    return pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve']
    })


@pytest.fixture
def sample_edges():
    """Create sample edges DataFrame"""
    return pd.DataFrame({
        'source': [1, 1, 2, 2, 3, 4],
        'target': [2, 3, 3, 4, 4, 5]
    })


@pytest.fixture
def edges_with_issues():
    """Create edges with quality issues"""
    return pd.DataFrame({
        'source': [1, 1, 2, 2, 3, 3, 6],  # 6 is dangling
        'target': [2, 2, 3, 3, 3, 3, 7]   # Duplicates, self-loop, dangling
    })


def test_graphprepml_init(sample_nodes, sample_edges):
    """Test GraphPrepML initialization"""
    prep = GraphPrepML(
        nodes_df=sample_nodes,
        edges_df=sample_edges,
        node_id_col='id',
        source_col='source',
        target_col='target'
    )
    
    assert prep.node_id_col == 'id'
    assert prep.source_col == 'source'
    assert prep.target_col == 'target'
    assert len(prep.nodes_df) == 5
    assert len(prep.edges_df) == 6


def test_graphprepml_init_edges_only(sample_edges):
    """Test initialization with edges only"""
    prep = GraphPrepML(edges_df=sample_edges, source_col='source', target_col='target')
    
    assert prep.nodes_df is None
    assert prep.edges_df is not None


def test_graphprepml_init_invalid_column(sample_nodes):
    """Test initialization with invalid column"""
    with pytest.raises(ValueError, match="not found"):
        GraphPrepML(nodes_df=sample_nodes, node_id_col='nonexistent')


def test_detect_issues(sample_nodes, sample_edges):
    """Test graph issue detection"""
    prep = GraphPrepML(nodes_df=sample_nodes, edges_df=sample_edges,
                      node_id_col='id', source_col='source', target_col='target')
    issues = prep.detect_issues()
    
    assert 'nodes' in issues
    assert 'edges' in issues
    assert issues['nodes']['total_nodes'] == 5
    assert issues['edges']['total_edges'] == 6


def test_detect_duplicate_nodes():
    """Test detection of duplicate node IDs"""
    nodes = pd.DataFrame({
        'id': [1, 2, 2, 3],  # Duplicate ID 2
        'name': ['A', 'B', 'B2', 'C']
    })
    
    prep = GraphPrepML(nodes_df=nodes, node_id_col='id')
    issues = prep.detect_issues()
    
    assert issues['nodes']['duplicate_node_ids'] == 1


def test_detect_duplicate_edges():
    """Test detection of duplicate edges"""
    edges = pd.DataFrame({
        'source': [1, 1, 2],
        'target': [2, 2, 3]  # Duplicate edge 1->2
    })
    
    prep = GraphPrepML(edges_df=edges, source_col='source', target_col='target')
    issues = prep.detect_issues()
    
    assert issues['edges']['duplicate_edges'] == 1


def test_detect_self_loops():
    """Test detection of self-loops"""
    edges = pd.DataFrame({
        'source': [1, 2, 3],
        'target': [2, 2, 4]  # Self-loop: 2->2
    })
    
    prep = GraphPrepML(edges_df=edges, source_col='source', target_col='target')
    issues = prep.detect_issues()
    
    assert issues['edges']['self_loops'] == 1


def test_detect_dangling_edges(sample_nodes):
    """Test detection of dangling edges"""
    edges = pd.DataFrame({
        'source': [1, 2, 6],  # Node 6 doesn't exist
        'target': [2, 3, 7]   # Node 7 doesn't exist
    })
    
    prep = GraphPrepML(nodes_df=sample_nodes, edges_df=edges,
                      node_id_col='id', source_col='source', target_col='target')
    issues = prep.detect_issues()
    
    assert issues['edges']['dangling_edges'] > 0


def test_validate_node_ids():
    """Test node ID validation"""
    nodes = pd.DataFrame({
        'id': [1, 2, None, 3, 3],  # Missing and duplicate IDs
        'name': ['A', 'B', 'C', 'D', 'E']
    })
    
    prep = GraphPrepML(nodes_df=nodes, node_id_col='id')
    result = prep.validate_node_ids()
    
    # Should remove None and duplicate
    assert len(result) == 3
    assert result['id'].isnull().sum() == 0


def test_validate_edges_remove_self_loops(sample_nodes):
    """Test edge validation with self-loop removal"""
    edges = pd.DataFrame({
        'source': [1, 2, 3],
        'target': [2, 2, 4]  # Self-loop: 2->2
    })
    
    prep = GraphPrepML(nodes_df=sample_nodes, edges_df=edges,
                      node_id_col='id', source_col='source', target_col='target')
    result = prep.validate_edges(remove_self_loops=True)
    
    assert len(result) == 2  # Self-loop removed


def test_validate_edges_remove_dangling(sample_nodes):
    """Test edge validation with dangling edge removal"""
    edges = pd.DataFrame({
        'source': [1, 2, 6],  # Node 6 doesn't exist
        'target': [2, 3, 7]   # Node 7 doesn't exist
    })
    
    prep = GraphPrepML(nodes_df=sample_nodes, edges_df=edges,
                      node_id_col='id', source_col='source', target_col='target')
    result = prep.validate_edges(remove_dangling=True)
    
    # Only first two edges should remain
    assert len(result) == 2


def test_remove_duplicate_edges():
    """Test duplicate edge removal"""
    edges = pd.DataFrame({
        'source': [1, 1, 2, 3],
        'target': [2, 2, 3, 4]  # Duplicate 1->2
    })
    
    prep = GraphPrepML(edges_df=edges, source_col='source', target_col='target')
    result = prep.remove_duplicate_edges(keep='first')
    
    assert len(result) == 3


def test_add_node_features(sample_nodes, sample_edges):
    """Test node feature extraction"""
    prep = GraphPrepML(nodes_df=sample_nodes, edges_df=sample_edges,
                      node_id_col='id', source_col='source', target_col='target')
    result = prep.add_node_features()
    
    assert 'out_degree' in result.columns
    assert 'in_degree' in result.columns
    assert 'total_degree' in result.columns
    assert 'is_isolated' in result.columns
    
    # Node 1 has 2 outgoing edges
    node1 = result[result['id'] == 1].iloc[0]
    assert node1['out_degree'] == 2


def test_add_edge_features(sample_edges):
    """Test edge feature extraction"""
    prep = GraphPrepML(edges_df=sample_edges, source_col='source', target_col='target')
    result = prep.add_edge_features()
    
    assert 'edge_count' in result.columns


def test_identify_components(sample_nodes, sample_edges):
    """Test connected component identification"""
    prep = GraphPrepML(nodes_df=sample_nodes, edges_df=sample_edges,
                      node_id_col='id', source_col='source', target_col='target')
    result = prep.identify_components()
    
    assert 'component_id' in result.columns
    # All nodes should be in same component (graph is connected)
    assert result['component_id'].nunique() <= 2


def test_identify_components_disconnected():
    """Test component identification with disconnected graph"""
    nodes = pd.DataFrame({'id': [1, 2, 3, 4, 5, 6]})
    edges = pd.DataFrame({
        'source': [1, 2, 4],  # Two components: {1,2,3} and {4,5}
        'target': [2, 3, 5]
    })
    
    prep = GraphPrepML(nodes_df=nodes, edges_df=edges,
                      node_id_col='id', source_col='source', target_col='target')
    result = prep.identify_components()
    
    # Should have at least 2 components
    assert result['component_id'].nunique() >= 2


def test_filter_isolated_nodes(sample_nodes):
    """Test isolated node filtering"""
    # Add isolated node
    nodes = pd.concat([sample_nodes, pd.DataFrame({'id': [6], 'name': ['Frank']})], ignore_index=True)
    edges = pd.DataFrame({
        'source': [1, 2],
        'target': [2, 3]
    })
    
    prep = GraphPrepML(nodes_df=nodes, edges_df=edges,
                      node_id_col='id', source_col='source', target_col='target')
    result = prep.filter_isolated_nodes()
    
    # Isolated nodes should be removed
    assert 6 not in result['id'].values


def test_to_edge_list(sample_edges):
    """Test conversion to edge list"""
    prep = GraphPrepML(edges_df=sample_edges, source_col='source', target_col='target')
    edge_list = prep.to_edge_list()
    
    assert isinstance(edge_list, list)
    assert len(edge_list) == 6
    assert all(isinstance(e, tuple) for e in edge_list)
    assert (1, 2) in edge_list


def test_to_adjacency_dict(sample_edges):
    """Test conversion to adjacency dictionary"""
    prep = GraphPrepML(edges_df=sample_edges, source_col='source', target_col='target')
    adj_dict = prep.to_adjacency_dict()
    
    assert isinstance(adj_dict, dict)
    assert 1 in adj_dict
    assert 2 in adj_dict[1]  # Edge 1->2 exists


def test_get_graph_stats(sample_nodes, sample_edges):
    """Test graph statistics calculation"""
    prep = GraphPrepML(nodes_df=sample_nodes, edges_df=sample_edges,
                      node_id_col='id', source_col='source', target_col='target')
    prep.add_node_features()
    stats = prep.get_graph_stats()
    
    assert 'num_nodes' in stats
    assert 'num_edges' in stats
    assert 'density' in stats
    assert 'avg_degree' in stats
    
    assert stats['num_nodes'] == 5
    assert stats['num_edges'] == 6


def test_report(sample_nodes, sample_edges):
    """Test report generation"""
    prep = GraphPrepML(nodes_df=sample_nodes, edges_df=sample_edges,
                      node_id_col='id', source_col='source', target_col='target')
    prep.validate_node_ids()
    prep.validate_edges()
    prep.add_node_features()
    
    report = prep.report()
    
    assert 'original_nodes_shape' in report
    assert 'current_nodes_shape' in report
    assert 'original_edges_shape' in report
    assert 'current_edges_shape' in report
    assert 'logs' in report
    assert 'issues' in report
    assert 'graph_stats' in report
    assert len(report['logs']) > 0


def test_chained_operations(sample_nodes, edges_with_issues):
    """Test chaining multiple operations"""
    prep = GraphPrepML(nodes_df=sample_nodes, edges_df=edges_with_issues,
                      node_id_col='id', source_col='source', target_col='target')
    
    # Chain operations
    prep.validate_node_ids()
    prep.validate_edges(remove_self_loops=True, remove_dangling=True)
    prep.remove_duplicate_edges()
    prep.add_node_features()
    prep.identify_components()
    
    # All operations should be applied
    assert 'out_degree' in prep.nodes_df.columns
    assert 'component_id' in prep.nodes_df.columns
    assert len(prep.edges_df) < len(edges_with_issues)  # Issues removed


def test_weighted_graph():
    """Test graph with edge weights"""
    edges = pd.DataFrame({
        'source': [1, 2, 3],
        'target': [2, 3, 4],
        'weight': [0.5, 0.8, 1.2]
    })
    
    prep = GraphPrepML(edges_df=edges, source_col='source', target_col='target')
    prep.add_edge_features()
    
    # Weight column should be preserved
    assert 'weight' in prep.edges_df.columns


def test_directed_vs_undirected():
    """Test that graph treats edges as directed"""
    edges = pd.DataFrame({
        'source': [1, 2],
        'target': [2, 1]  # Bidirectional edge
    })
    
    prep = GraphPrepML(edges_df=edges, source_col='source', target_col='target')
    edge_list = prep.to_edge_list()
    
    # Should have 2 directed edges
    assert len(edge_list) == 2
    assert (1, 2) in edge_list
    assert (2, 1) in edge_list
