#!/usr/bin/env python3

def count_graph_stats(filename):
    """Count the number of vertices and edges in a graph file."""
    vertices = set()
    edge_count = 0
    
    try:
        with open(filename, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:  # Ensure there are at least source and destination vertices
                    a, b = int(parts[0]), int(parts[1])
                    vertices.add(a)
                    vertices.add(b)
                    edge_count += 1
        
        # Find the max vertex ID and count
        max_vertex = max(vertices) if vertices else 0
        vertex_count = len(vertices)
        
        return vertex_count, edge_count, max_vertex
    except Exception as e:
        print(f"Error processing file: {e}")
        return 0, 0, 0

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <edge-file>")
        sys.exit(1)
    
    filename = sys.argv[1]
    vertex_count, edge_count, max_vertex = count_graph_stats(filename)
    
    print(f"Graph statistics for {filename}:")
    print(f"Number of vertices: {vertex_count}")
    print(f"Highest vertex ID: {max_vertex}")
    print(f"Number of edges: {edge_count}")
    print(f"\nRun command: ./csr {max_vertex} {edge_count} {filename}") 