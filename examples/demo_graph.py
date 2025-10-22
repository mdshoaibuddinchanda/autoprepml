"""
Demo: Graph Data Preprocessing with AutoPrepML
Example: Social network analysis with node/edge validation
"""
import pandas as pd
from autoprepml.graph import GraphPrepML

# Sample data: Social network (users and friendships)
# Create nodes (users)
nodes_data = {
    'id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 11],  # Note: duplicate ID 11
    'username': [
        'alice', 'bob', 'charlie', 'david', 'eve',
        'frank', 'grace', 'henry', 'iris', 'jack',
        'kate', 'kate_duplicate'
    ],
    'followers': [120, 85, 200, 45, 150, 30, 95, 110, 70, 180, 60, 61],
}

# Create edges (friendships)
edges_data = {
    'source': [1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7, 8, 8, 15, 16],  # 15, 16 are dangling
    'target': [2, 3, 4, 3, 5, 4, 5, 5, 5, 6, 7, 7, 8, 9, 10, 99, 100],  # 99, 100 don't exist
    'interaction_count': [25, 40, 15, 30, 20, 35, 45, 10, 10, 50, 25, 30, 40, 20, 35, 1, 1]
}

nodes_df = pd.DataFrame(nodes_data)
edges_df = pd.DataFrame(edges_data)

print("=" * 80)
print("🕸️  GRAPH DATA PREPROCESSING DEMO - Social Network Analysis")
print("=" * 80)

# Initialize GraphPrepML
print("\n1️⃣  Initializing GraphPrepML...")
prep = GraphPrepML(
    nodes_df=nodes_df,
    edges_df=edges_df,
    node_id_col='id',
    source_col='source',
    target_col='target'
)
print(f"✓ Loaded {len(prep.nodes_df)} nodes and {len(prep.edges_df)} edges")

# Detect issues
print("\n2️⃣  Detecting graph data quality issues...")
issues = prep.detect_issues()

print("\n   Node Issues:")
print(f"   • Total nodes: {issues['nodes']['total_nodes']}")
print(f"   • Duplicate node IDs: {issues['nodes']['duplicate_node_ids']}")
print(f"   • Missing node IDs: {issues['nodes']['missing_node_ids']}")

print("\n   Edge Issues:")
print(f"   • Total edges: {issues['edges']['total_edges']}")
print(f"   • Duplicate edges: {issues['edges']['duplicate_edges']}")
print(f"   • Self-loops: {issues['edges']['self_loops']}")
print(f"   • Missing sources: {issues['edges']['missing_sources']}")
print(f"   • Missing targets: {issues['edges']['missing_targets']}")
print(f"   • Dangling edges: {issues['edges']['dangling_edges']}")

# Validate node IDs
print("\n3️⃣  Validating and cleaning node IDs...")
original_nodes = len(prep.nodes_df)
prep.validate_node_ids()
removed_nodes = original_nodes - len(prep.nodes_df)
print(f"✓ Removed {removed_nodes} invalid nodes (duplicates/missing IDs)")
print(f"✓ Valid nodes remaining: {len(prep.nodes_df)}")

# Validate edges
print("\n4️⃣  Validating and cleaning edges...")
original_edges = len(prep.edges_df)
prep.validate_edges(remove_self_loops=True, remove_dangling=True)
removed_edges = original_edges - len(prep.edges_df)
print(f"✓ Removed {removed_edges} invalid edges (self-loops/dangling)")
print(f"✓ Valid edges remaining: {len(prep.edges_df)}")

# Remove duplicate edges
print("\n5️⃣  Removing duplicate edges...")
original_edges = len(prep.edges_df)
prep.remove_duplicate_edges(keep='first')
removed_edges = original_edges - len(prep.edges_df)
print(f"✓ Removed {removed_edges} duplicate edges")

# Add node features
print("\n6️⃣  Extracting node-level features...")
prep.add_node_features()
print("✓ Added features:")
print("   - out_degree (outgoing connections)")
print("   - in_degree (incoming connections)")
print("   - total_degree (total connections)")
print("   - is_isolated (no connections)")

# Show top connected users
print("\n   Top 5 most connected users:")
top_users = prep.nodes_df.nlargest(5, 'total_degree')[['id', 'username', 'total_degree']]
for _, row in top_users.iterrows():
    print(f"   • {row['username']}: {row['total_degree']} connections")

# Add edge features
print("\n7️⃣  Extracting edge-level features...")
prep.add_edge_features()
print("✓ Added edge features: edge_count")

# Identify connected components
print("\n8️⃣  Identifying connected components...")
prep.identify_components()
num_components = prep.nodes_df['component_id'].nunique()
print(f"✓ Found {num_components} connected component(s)")

# Show component distribution
component_sizes = prep.nodes_df.groupby('component_id').size().sort_values(ascending=False)
print("\n   Component sizes:")
for comp_id, size in component_sizes.items():
    print(f"   • Component {comp_id}: {size} nodes")

# Check for isolated nodes
isolated_count = prep.nodes_df['is_isolated'].sum()
print(f"\n   Isolated nodes: {isolated_count}")

if isolated_count > 0:
    isolated_users = prep.nodes_df[prep.nodes_df['is_isolated']]['username'].tolist()
    print(f"   Isolated users: {isolated_users}")

# Get graph statistics
print("\n9️⃣  Computing graph statistics...")
stats = prep.get_graph_stats()
print("✓ Graph metrics:")
print(f"   • Number of nodes: {stats['num_nodes']}")
print(f"   • Number of edges: {stats['num_edges']}")
print(f"   • Graph density: {stats['density']:.4f}")
print(f"   • Average degree: {stats['avg_degree']:.2f}")

# Convert to different formats
print("\n🔟 Converting to different graph representations...")

# Edge list
edge_list = prep.to_edge_list()
print(f"✓ Edge list format: {len(edge_list)} edges")
print(f"   Sample edges: {edge_list[:3]}")

# Adjacency dictionary
adj_dict = prep.to_adjacency_dict()
print(f"✓ Adjacency dictionary: {len(adj_dict)} nodes")
print(f"   Sample: Node 1 connects to: {adj_dict.get(1, [])}")

# Display sample network data
print("\n📊 Sample Network Data:")
print("-" * 80)
print("\nNodes (Top 5):")
print(prep.nodes_df[['id', 'username', 'followers', 'total_degree', 'component_id']].head().to_string(index=False))

print("\nEdges (Top 5):")
print(prep.edges_df[['source', 'target', 'interaction_count']].head().to_string(index=False))

# Generate report
print("\n📈 Generating preprocessing report...")
report = prep.report()
print(f"✓ Original nodes shape: {report['original_nodes_shape']}")
print(f"✓ Current nodes shape:  {report['current_nodes_shape']}")
print(f"✓ Original edges shape: {report['original_edges_shape']}")
print(f"✓ Current edges shape:  {report['current_edges_shape']}")
print(f"✓ Operations performed: {len(report['logs'])}")

# Save cleaned data
nodes_output = 'social_network_nodes_cleaned.csv'
edges_output = 'social_network_edges_cleaned.csv'
prep.nodes_df.to_csv(nodes_output, index=False)
prep.edges_df.to_csv(edges_output, index=False)
print(f"\n💾 Saved cleaned graph data:")
print(f"   • Nodes: {nodes_output}")
print(f"   • Edges: {edges_output}")

# Summary
print("\n" + "=" * 80)
print("✨ GRAPH PREPROCESSING COMPLETE!")
print("=" * 80)
print("Network Statistics:")
print(f"   • Total users: {len(prep.nodes_df)}")
print(f"   • Total friendships: {len(prep.edges_df)}")
print(f"   • Connected components: {num_components}")
print(f"   • Isolated users: {isolated_count}")
print(f"   • Network density: {stats['density']:.4f}")
print(f"   • Avg connections per user: {stats['avg_degree']:.2f}")

print("\n💡 Use Cases:")
print("   • Social network analysis")
print("   • Community detection")
print("   • Influence analysis")
print("   • Recommendation systems")
print("   • Link prediction")

print("\n📊 Ready for graph algorithms:")
print("   • PageRank")
print("   • Centrality measures")
print("   • Community detection (Louvain, Label Propagation)")
print("   • Graph Neural Networks (GNN)")
print("   • Network visualization")
print("=" * 80)
