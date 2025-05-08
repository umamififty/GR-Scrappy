import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from matplotlib.colors import LinearSegmentedColormap, to_rgba
import networkx as nx


class PolicyDendrogramVisualizer:
    """
    Creates hierarchical visualizations of policy categories and their relationships
    based on policy data scraped from gov.uk
    """
    
    def __init__(self, base_dir="policy_data"):
        """
        Initialize the visualizer
        
        Args:
            base_dir (str): Base directory containing the policy data
        """
        self.base_dir = base_dir
        self.output_dir = os.path.join(base_dir, "visualizations")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Custom colormap for dendrograms
        self.colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", 
                       "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
        
        # Load policy data
        self.policies_path = os.path.join(base_dir, "all_policies.csv")
        self.categories_path = os.path.join(base_dir, "category_structure.json")
        
        self.policies_df = None
        self.categories = None
        self._load_data()
    
    def _load_data(self):
        """Load policy data from CSV and category structure from JSON"""
        try:
            if os.path.exists(self.policies_path):
                self.policies_df = pd.read_csv(self.policies_path)
                print(f"Loaded {len(self.policies_df)} policies from {self.policies_path}")
            else:
                print(f"Warning: Policy data file not found at {self.policies_path}")
                
            if os.path.exists(self.categories_path):
                with open(self.categories_path, 'r', encoding='utf-8') as f:
                    self.categories = json.load(f)
                print(f"Loaded category structure from {self.categories_path}")
            else:
                print(f"Warning: Category structure file not found at {self.categories_path}")
                
        except Exception as e:
            print(f"Error loading data: {e}")
    
    def create_hierarchy_dendrogram(self):
        """
        Creates a dendrogram visualization of policy categories and subcategories
        
        Returns:
            str: Path to the saved visualization file, or None if failed
        """
        if not self.categories:
            print("No category data available for visualization")
            return None
        
        # Build hierarchical structure
        nodes = []
        parent_map = {}
        node_id = 0
        
        # Add root node
        root_node = {"id": node_id, "name": "All Policies", "level": 0, "parent": None}
        nodes.append(root_node)
        root_id = node_id
        node_id += 1
        
        # Add category nodes
        for cat_name, cat_data in self.categories.items():
            cat_node = {
                "id": node_id,
                "name": cat_name,
                "level": 1,
                "parent": root_id,
                "count": cat_data.get("count", 0)
            }
            nodes.append(cat_node)
            parent_map[node_id] = root_id
            cat_id = node_id
            node_id += 1
            
            # Add subcategory nodes
            for subcat_name, subcat_data in cat_data.get("subcategories", {}).items():
                if subcat_name == "None" or not subcat_name:
                    continue
                    
                subcat_node = {
                    "id": node_id,
                    "name": subcat_name,
                    "level": 2,
                    "parent": cat_id,
                    "count": subcat_data.get("count", 0)
                }
                nodes.append(subcat_node)
                parent_map[node_id] = cat_id
                node_id += 1
        
        # Convert nodes to array for processing
        nodes = np.array(nodes)
        
        # Calculate distances between nodes
        Z = self._calculate_custom_linkage(nodes, parent_map)
        
        # Create figure
        plt.figure(figsize=(14, 8))
        
        # Create custom labels
        labels = [f"{node['name']} ({node.get('count', 0)})" if 'count' in node else node['name'] 
                 for node in nodes]
        
        # Plot dendrogram
        dendrogram(
            Z,
            labels=labels,
            orientation='left',
            leaf_font_size=10,
            color_threshold=1.5,  # Adjust to control coloring
            above_threshold_color='gray',
            leaf_rotation=0,
            link_color_func=lambda k: self.colors[k % len(self.colors)]
        )
        
        plt.title("Policy Categories Hierarchy", fontsize=16)
        plt.tight_layout()
        
        # Save visualization
        output_path = os.path.join(self.output_dir, "policy_hierarchy_dendrogram.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Hierarchy dendrogram saved to {output_path}")
        return output_path
    
    def create_department_dendrogram(self):
        """
        Creates a dendrogram visualization of policies by department
        
        Returns:
            str: Path to the saved visualization file, or None if failed
        """
        if self.policies_df is None or self.policies_df.empty:
            print("No policy data available for department visualization")
            return None
        
        # Group policies by department
        if 'department' not in self.policies_df.columns:
            print("Department information not available in policy data")
            return None
        
        # Fill missing departments
        self.policies_df['department'] = self.policies_df['department'].fillna('Unknown')
        
        # Count policies by department
        dept_counts = self.policies_df['department'].value_counts()
        
        # Only include departments with at least 2 policies
        valid_depts = dept_counts[dept_counts >= 2].index.tolist()
        
        if not valid_depts:
            print("No departments with sufficient policy counts for clustering")
            return None
        
        # Filter policies to those from valid departments
        dept_policies = self.policies_df[self.policies_df['department'].isin(valid_depts)]
        
        # Create a matrix with departments as rows
        dept_matrix = []
        dept_names = []
        
        for dept in valid_depts:
            dept_policies_subset = dept_policies[dept_policies['department'] == dept]
            
            # Create a feature vector for each department
            # We'll use category and subcategory as features
            dept_vector = []
            
            # For simplicity, we'll just use occurrence count of each category
            if 'category' in dept_policies_subset.columns:
                cat_counts = dept_policies_subset['category'].value_counts()
                
                # Get all unique categories
                all_cats = self.policies_df['category'].dropna().unique()
                
                for cat in all_cats:
                    dept_vector.append(cat_counts.get(cat, 0))
            
            dept_matrix.append(dept_vector)
            dept_names.append(f"{dept} ({len(dept_policies_subset)})")
        
        # Convert to numpy array
        dept_matrix = np.array(dept_matrix)
        
        # If we have at least two departments, create the dendrogram
        if len(dept_matrix) >= 2:
            from scipy.cluster.hierarchy import linkage
            
            # Calculate linkage
            Z = linkage(dept_matrix, method='ward')
            
            # Create figure
            plt.figure(figsize=(12, 8))
            
            # Plot dendrogram
            dendrogram(
                Z,
                labels=dept_names,
                orientation='right',
                leaf_font_size=10,
                color_threshold=1.5,
                above_threshold_color='gray'
            )
            
            plt.title("Departments Clustering by Policy Categories", fontsize=16)
            plt.tight_layout()
            
            # Save visualization
            output_path = os.path.join(self.output_dir, "department_clustering_dendrogram.png")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Department dendrogram saved to {output_path}")
            return output_path
        
        return None
    
    def create_graph_visualization(self):
        """
        Creates a network graph visualization of policy categories and subcategories
        
        Returns:
            str: Path to the saved visualization file, or None if failed
        """
        if not self.categories:
            print("No category data available for visualization")
            return None
        
        # Create directed graph
        G = nx.DiGraph()
        
        # Add root node
        G.add_node("All Policies", level=0, count=0)
        
        # Calculate total policy count
        total_policies = 0
        
        # Add category nodes and edges
        for cat_name, cat_data in self.categories.items():
            cat_count = cat_data.get("count", 0)
            total_policies += cat_count
            
            # Add category node
            G.add_node(cat_name, level=1, count=cat_count)
            
            # Add edge from root to category
            G.add_edge("All Policies", cat_name, weight=cat_count)
            
            # Add subcategory nodes and edges
            for subcat_name, subcat_data in cat_data.get("subcategories", {}).items():
                if subcat_name == "None" or not subcat_name:
                    continue
                
                subcat_count = subcat_data.get("count", 0)
                
                # Add subcategory node
                node_name = f"{cat_name}: {subcat_name}"
                G.add_node(node_name, level=2, count=subcat_count)
                
                # Add edge from category to subcategory
                G.add_edge(cat_name, node_name, weight=subcat_count)
        
        # Update root node count
        G.nodes["All Policies"]["count"] = total_policies
        
        # Create figure
        plt.figure(figsize=(16, 12))
        
        # Define positions using a hierarchical layout
        pos = nx.spring_layout(G, k=0.8, iterations=50)
        
        # Get node levels
        node_levels = nx.get_node_attributes(G, 'level')
        
        # Define node sizes based on count
        node_counts = nx.get_node_attributes(G, 'count')
        node_sizes = [max(300, count * 20) for count in node_counts.values()]
        
        # Define node colors based on level
        node_colors = [self.colors[level % len(self.colors)] for level in node_levels.values()]
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.8)
        
        # Draw edges with varying thickness
        edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
        edge_widths = [max(1, w/5) for w in edge_weights]
        nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color='gray', alpha=0.6, 
                              arrows=True, arrowsize=15, arrowstyle='->')
        
        # Draw labels
        label_dict = {node: f"{node}\n({G.nodes[node]['count']})" for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels=label_dict, font_size=8, font_weight='bold')
        
        plt.title("Policy Categories Network", fontsize=16)
        plt.axis('off')
        plt.tight_layout()
        
        # Save visualization
        output_path = os.path.join(self.output_dir, "policy_categories_network.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Network graph saved to {output_path}")
        return output_path
    
    def _calculate_custom_linkage(self, nodes, parent_map):
        """
        Calculate custom hierarchical clustering linkage matrix based on the predefined parent-child relationships
        
        Args:
            nodes (np.array): Array of node dictionaries
            parent_map (dict): Mapping of node IDs to their parent node IDs
            
        Returns:
            np.array: Linkage matrix in format expected by scipy's dendrogram
        """
        n = len(nodes)
        Z = np.zeros((n-1, 4))
        
        # Create a map of original indices to cluster indices
        cluster_map = {i: i for i in range(n)}
        
        # Track which nodes have been merged
        merged = set()
        
        # For each hierarchical level, merge nodes with their parents
        for level in range(2, 0, -1):  # Start from leaves and go up
            level_nodes = [node['id'] for node in nodes if node['level'] == level]
            
            for i, node_id in enumerate(level_nodes):
                if node_id in merged:
                    continue
                
                parent_id = parent_map.get(node_id)
                if parent_id is None:
                    continue
                
                # Find siblings (nodes with the same parent)
                siblings = [sibling_id for sibling_id in level_nodes 
                           if sibling_id != node_id 
                           and parent_map.get(sibling_id) == parent_id
                           and sibling_id not in merged]
                
                # If no siblings, connect directly to parent in the next iteration
                if not siblings:
                    continue
                
                # Otherwise, merge this node with a sibling
                sibling_id = siblings[0]
                
                # Mark both as merged
                merged.add(node_id)
                merged.add(sibling_id)
                
                # Add to linkage matrix
                Z_idx = len(merged) // 2 - 1
                
                # Create new cluster
                new_cluster = n + Z_idx
                
                # This is the key part where we need to get cluster indices, not original indices
                cluster_idx1 = cluster_map[node_id]
                cluster_idx2 = cluster_map[sibling_id]
                
                # Calculate distance (roughly based on node levels)
                distance = 1.0
                
                # Size of new cluster
                size = 2
                
                Z[Z_idx] = [cluster_idx1, cluster_idx2, distance, size]
                
                # Update cluster map for the new cluster
                cluster_map[new_cluster] = new_cluster
                
                # Update cluster map for the merged nodes
                for merged_id in [node_id, sibling_id]:
                    cluster_map[merged_id] = new_cluster
        
        # Handle any remaining nodes (connect to their parents)
        remaining = [node['id'] for node in nodes if node['id'] not in merged and node['level'] > 0]
        
        for i, node_id in enumerate(remaining):
            parent_id = parent_map.get(node_id)
            if parent_id is None or parent_id in merged:
                continue
            
            # Mark both as merged
            merged.add(node_id)
            merged.add(parent_id)
            
            # Add to linkage matrix
            Z_idx = len(merged) // 2 - 1
            if Z_idx >= len(Z):
                break
                
            # Create new cluster
            new_cluster = n + Z_idx
            
            # Get cluster indices
            cluster_idx1 = cluster_map[node_id]
            cluster_idx2 = cluster_map[parent_id]
            
            # Calculate distance (parent-child relationship)
            distance = 0.8
            
            # Size of new cluster
            size = 2
            
            Z[Z_idx] = [cluster_idx1, cluster_idx2, distance, size]
            
            # Update cluster map
            cluster_map[new_cluster] = new_cluster
            for merged_id in [node_id, parent_id]:
                cluster_map[merged_id] = new_cluster
        
        # Handle the root node if needed
        if len(merged) < n - 1:
            root_id = nodes[0]['id']
            if root_id not in merged:
                # Connect any remaining clusters to the root
                remaining_clusters = set(cluster_map.values()) - {cluster_map[root_id]}
                
                for i, cluster_id in enumerate(remaining_clusters):
                    Z_idx = len(merged) // 2 + i
                    if Z_idx >= len(Z):
                        break
                        
                    # Create new cluster
                    new_cluster = n + Z_idx
                    
                    # Get cluster indices
                    cluster_idx1 = cluster_map[root_id]
                    cluster_idx2 = cluster_id
                    
                    # Higher distance for dissimilar clusters
                    distance = 1.5
                    
                    # Size approximation
                    size = n - len(remaining_clusters) + 1
                    
                    Z[Z_idx] = [cluster_idx1, cluster_idx2, distance, size]
                    
                    # Update cluster map
                    cluster_map[new_cluster] = new_cluster
                    cluster_map[root_id] = new_cluster
                    cluster_map[cluster_id] = new_cluster
        
        return Z
    
    def create_all_visualizations(self):
        """
        Creates all types of visualizations
        
        Returns:
            list: Paths to all created visualization files
        """
        outputs = []
        
        # Create hierarchy dendrogram
        hierarchy_path = self.create_hierarchy_dendrogram()
        if hierarchy_path:
            outputs.append(hierarchy_path)
        
        # Create department dendrogram
        dept_path = self.create_department_dendrogram()
        if dept_path:
            outputs.append(dept_path)
        
        # Create network graph visualization
        graph_path = self.create_graph_visualization()
        if graph_path:
            outputs.append(graph_path)
        
        return outputs


if __name__ == "__main__":
    # Test the visualizer with sample data
    visualizer = PolicyDendrogramVisualizer()
    outputs = visualizer.create_all_visualizations()
    print(f"Created {len(outputs)} visualizations")