import os
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import networkx as nx
from PIL import Image, ImageDraw, ImageFont
import pydot
from networkx.drawing.nx_pydot import graphviz_layout

def calculate_size(path):
    """Calculate the size of a file or directory in KB."""
    try:
        if path.is_file():
            return path.stat().st_size // 1024
        else:
            return sum(calculate_size(p) for p in path.glob('*') if not p.is_symlink())
    except Exception:
        return 0  # For files/folders we can't access

def print_tree(directory, prefix="", show_size=True, max_depth=None, current_depth=0):
    """Print a tree structure of the directory."""
    directory = Path(directory)
    
    if max_depth is not None and current_depth > max_depth:
        return
    
    # Get all items in the directory
    try:
        items = sorted(list(directory.iterdir()), key=lambda x: (x.is_file(), x.name))
    except PermissionError:
        print(f"{prefix}├── Error: Permission denied")
        return
    
    # Process each item
    for i, item in enumerate(items):
        is_last = i == len(items) - 1
        
        # Determine the new prefix for the next level
        new_prefix = prefix + ("└── " if is_last else "├── ")
        next_prefix = prefix + ("    " if is_last else "│   ")
        
        # Calculate size if required
        size_str = ""
        if show_size:
            if item.is_file():
                size = item.stat().st_size // 1024
                size_str = f" ({size} KB)"
            elif item.is_dir():
                size = calculate_size(item)
                size_str = f" ({size} KB)"
        
        # Print the current item
        print(f"{new_prefix}{item.name}{size_str}")
        
        # Recursively process directories
        if item.is_dir() and not item.is_symlink():
            print_tree(item, next_prefix, show_size, max_depth, current_depth + 1)

def build_graph(directory, parent=None, graph=None, show_size=True, max_depth=None, current_depth=0):
    """Build a NetworkX graph representation of the directory structure."""
    directory = Path(directory)
    
    if graph is None:
        graph = nx.DiGraph()
        parent = directory.name
        graph.add_node(parent)
    
    if max_depth is not None and current_depth > max_depth:
        return graph
    
    try:
        items = sorted(list(directory.iterdir()), key=lambda x: (x.is_file(), x.name))
    except PermissionError:
        node_name = f"{parent}/Error: Permission denied"
        graph.add_node(node_name)
        graph.add_edge(parent, node_name)
        return graph
    
    for item in items:
        # Create a node name
        if show_size:
            if item.is_file():
                size = item.stat().st_size // 1024
                node_name = f"{item.name} ({size} KB)"
            else:
                size = calculate_size(item)
                node_name = f"{item.name} ({size} KB)"
        else:
            node_name = item.name
        
        # Create a unique ID for the node (to avoid duplicates)
        node_id = f"{parent}/{node_name}"
        
        # Add the node and edge
        graph.add_node(node_id, label=node_name, is_file=item.is_file())
        graph.add_edge(parent, node_id)
        
        # Recursively process directories
        if item.is_dir() and not item.is_symlink():
            build_graph(item, node_id, graph, show_size, max_depth, current_depth + 1)
    
    return graph

def generate_graph_image(graph, output_file, root_dir):
    """Generate a visual representation of the directory tree using NetworkX and pydot."""
    try:
        # Convert to pydot graph for better layout control
        pdot = nx.drawing.nx_pydot.to_pydot(graph)
        
        # Set graph attributes for better visualization
        pdot.set_rankdir('LR')  # Left to right layout
        pdot.set_fontname('Arial')
        pdot.set_fontsize('10')
        pdot.set_nodesep('0.5')
        pdot.set_ranksep('0.5')
        
        # Set node attributes
        for node in pdot.get_nodes():
            node_name = node.get_name().strip('"')
            if '/' not in node_name:  # Root node
                node.set_shape('box')
                node.set_style('filled')
                node.set_fillcolor('#ADD8E6')  # Light blue
                node.set_fontsize('12')
                node.set_fontweight('bold')
            else:
                parent_path, node_label = node_name.rsplit('/', 1)
                # Check if the node is a file
                is_file = any(attr.get_key() == 'is_file' and attr.get_value() == 'True' 
                              for attr in node.get_attributes().get_attributes())
                
                if is_file:
                    node.set_shape('ellipse')
                    node.set_style('filled')
                    node.set_fillcolor('#FFFACD')  # Light yellow
                else:
                    node.set_shape('box')
                    node.set_style('filled')
                    node.set_fillcolor('#E6E6FA')  # Lavender
                
                # Set the displayed label to just the filename, not the path
                node.set_label(node_label)
        
        # Save the image in the requested format
        extension = output_file.split('.')[-1].lower()
        
        if extension in ['jpg', 'jpeg', 'png']:
            pdot.write(output_file, format=extension)
            print(f"Graph image saved to: {output_file}")
        else:
            print(f"Unsupported image format: {extension}. Please use jpg, jpeg, or png.")
    except Exception as e:
        print(f"Error generating graph image: {e}")
        print("Make sure Graphviz is installed on your system.")
        print("You can install it with:")
        print("- Windows: Download from https://graphviz.org/download/")
        print("- macOS: brew install graphviz")
        print("- Ubuntu/Debian: sudo apt-get install graphviz")

def generate_tree(root_dir, output_file=None, show_size=True, max_depth=None, image_output=None):
    """Generate a tree structure and optionally save to a file and/or create an image."""
    root_dir = Path(root_dir)
    
    # Generate text tree
    if output_file:
        import sys
        original_stdout = sys.stdout
        with open(output_file, 'w', encoding='utf-8') as f:
            sys.stdout = f
            
            print(f"{root_dir.name}{' (' + str(calculate_size(root_dir)) + ' KB)' if show_size else ''}")
            print_tree(root_dir, "", show_size, max_depth)
            
            sys.stdout = original_stdout
        print(f"Tree structure saved to: {output_file}")
    else:
        print(f"{root_dir.name}{' (' + str(calculate_size(root_dir)) + ' KB)' if show_size else ''}")
        print_tree(root_dir, "", show_size, max_depth)
    
    # Generate image tree if requested
    if image_output:
        print(f"Generating image representation in {image_output}...")
        graph = build_graph(root_dir, show_size=show_size, max_depth=max_depth)
        generate_graph_image(graph, image_output, root_dir)


# ===============================================================
# CONFIGURATION SECTION - EDIT THESE VALUES TO CUSTOMIZE OUTPUT
# ===============================================================

# The directory you want to visualize
DIRECTORY_TO_VISUALIZE = "policy_data"

# Text output file (set to None if you don't want text output)
TEXT_OUTPUT_FILE = "directory_tree.txt"

# Image output file (set to None if you don't want image output)
# Must end with .png, .jpg, or .jpeg
IMAGE_OUTPUT_FILE = "directory_tree.png"

# Show file and directory sizes
SHOW_SIZES = True

# Maximum depth to display (set to None for unlimited)
MAX_DEPTH = None

# ===============================================================
# RUN THE PROGRAM
# ===============================================================

if __name__ == "__main__":
    # Check if this script is being run directly
    print(f"Generating directory tree for: {DIRECTORY_TO_VISUALIZE}")
    generate_tree(
        root_dir=DIRECTORY_TO_VISUALIZE,
        output_file=TEXT_OUTPUT_FILE,
        show_size=SHOW_SIZES,
        max_depth=MAX_DEPTH,
        image_output=IMAGE_OUTPUT_FILE
    )
    print("Done!")