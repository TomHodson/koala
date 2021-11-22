from .graph_utils import vertex_neighbours, clockwise_about
from .voronization import Lattice
from .graph_utils import get_edge_vectors
import itertools as it
import numpy as np

def normalised(a): return a / np.linalg.norm(a)

def vertices_to_triangles(g, edge_labels, triangle_size = 0.05):
    """
    Map every vertex to a triangle.
    The original vertex labeled by i get mapped to vertices 3*i, 3*i + 1, 3*i + 2
    Where the mapping with those three is determined by the edge coloring which allows us to connect the right
    vertices of neighbouring triangles together.
    """
    new_vertices = np.zeros(shape = (g.vertices.shape[0]*3, 2), dtype = float)
    new_adjacency = np.zeros(shape = (g.adjacency.shape[0] + g.vertices.shape[0]*3, 2), dtype = int)
    new_adjacency_crossing = np.zeros(shape = (g.adjacency.shape[0] + g.vertices.shape[0]*3, 2), dtype = int)
    
    # loop over each vertex, look at its three neighbours
    # make 3 new vertices in its place shifted towards the nieghbours
    for vertex_i in range(g.vertices.shape[0]):
        
        # get vertex and edge neighbours of the vertex
        this_vertex = g.vertices[vertex_i]
        vertex_indices, edge_indices = vertex_neighbours(vertex_i, g.adjacency)
        vertex_indices, edge_indices = clockwise_about(vertex_i, g)
        
        # this function takes into account the fact that edges can cross boundaries
        edge_vectors = get_edge_vectors(vertex_i, edge_indices, g)

        # loop over the neigbours, the new vertices will have label = vertex_i + edge_label
        for k, vertex_j, edge_j in zip(it.count(), vertex_indices, edge_indices):
            # use the color of the edge in question to determine where to put it
            # inside new_vertices
            edge_label = edge_labels[edge_j]
            index = 3*vertex_i + edge_label
            
            # push the new vertex out from the center along the edge
            this_triangle_size = min(0.2 * np.linalg.norm(edge_vectors[k]), triangle_size)
            new_vertex_position = (this_vertex + this_triangle_size * normalised(edge_vectors[k]))
            other_vertex_position = (this_vertex + edge_vectors[k] - this_triangle_size * normalised(edge_vectors[k]))
            new_vertices[index] = new_vertex_position
            
            # make external edge
            other_index = 3*vertex_j + edge_label #index of the vertex on another site
            new_adjacency[edge_j] = (index, other_index)
            new_adjacency_crossing[edge_j] = np.floor(other_vertex_position) - np.floor(new_vertex_position)
            
            # make internal edges
            next_edge_j = edge_indices[(k+1)%3]
            next_edge_label = edge_labels[next_edge_j]
            other_index = 3*vertex_i + next_edge_label #index of the next vertex inside the site
            new_adjacency[g.adjacency.shape[0] + index] = (index, other_index)
            #new_adjacency_crossing[g.adjacency.shape[0] + index] = np.floor(new_vertices[other_index]) - np.floor(new_vertices[index])
           
    # now that all the vertices and edges have been assigned
    # go back and set adjacency_crossing for the internal vertices
    # I'm not 100% sure why you need to do the external vertices up there and the
    # internal onces down here, but it seems to work.
    for edge_j in np.arange(g.adjacency.shape[0], new_adjacency.shape[0]):
        start, end = new_vertices[new_adjacency[edge_j]]
        new_adjacency_crossing[edge_j] = (np.floor(end) - np.floor(start)).astype(int)

    new_vertices = new_vertices % 1
            
            
    return Lattice(
        vertices = new_vertices,
        adjacency = new_adjacency,
        adjacency_crossing = new_adjacency_crossing,
        vor = None,
    )