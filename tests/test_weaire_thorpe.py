import numpy as np
from matplotlib import pyplot as plt

from koala.weaire_thorpe import vertices_to_triangles
from koala.voronization import Lattice, generate_pbc_voronoi_adjacency
from koala.graph_utils import clockwise_edges_about
from koala.pointsets import generate_bluenoise
from koala.graph_color import edge_color
from koala.plotting import plot_lattice

def test_smoketest_weaire_thorpe():
    """This test doesn't do any assertions, it just checks that the normal usage doesn't crash"""
    fig, ax = plt.subplots()

    points =  generate_bluenoise(k = 100, nx = 5, ny = 5)
    g = generate_pbc_voronoi_adjacency(points)

    ordered_edge_indices = clockwise_edges_about(vertex_i = 0, g=g)
    solveable, edge_labels = edge_color(g.adjacency, n_colors = 3, fixed = enumerate(ordered_edge_indices))

    solveable, edge_labels = edge_color(g.adjacency, n_colors = 3, fixed = enumerate(ordered_edge_indices))
    WT_g = vertices_to_triangles(g, edge_labels);

    #ax.scatter(*g.vertices.T, color = 'k', alpha = 0.9)
    #ax.scatter(*WT_g.vertices.T, color = 'g')

    edge_labels = np.where(np.arange(WT_g.adjacency.shape[0]) < g.adjacency.shape[0], 0, 1)
    edge_arrows = np.where(np.arange(WT_g.adjacency.shape[0]) < g.adjacency.shape[0], 0, 1)

    plot_lattice(WT_g, edge_arrows = edge_arrows, ax = ax, edge_labels = edge_labels)


def test_multi_graphs():
    """The following graph has vertices with edges that link the same vertex together,
     this checks that clockwise edges about handles them correctly as part of the Weaire-Thorpe process"""

    g = Lattice(
        vertices = np.array([[0.5,0.7], [0.5,0.3]]),
        adjacency = np.array([[0,1],[0,0],[0,1],[1,1]]),
        adjacency_crossing = np.array([[0,0],[1,0],[1,0],[1,0]]),
        vor = None,
    )

    ordered_edge_indices = clockwise_edges_about(vertex_i = 0, g=g)
    solveable, edge_labels = edge_color(g.adjacency, n_colors = 3, fixed = enumerate(ordered_edge_indices))

    WT_g = vertices_to_triangles(g, edge_labels)

    edge_labels = np.where(np.arange(WT_g.adjacency.shape[0]) < g.adjacency.shape[0], 0, 1)
    internal_edges = np.where(np.arange(WT_g.adjacency.shape[0]) < g.adjacency.shape[0], 0, 1)

    fig, axes = plt.subplots(ncols = 2)
    plot_lattice(g, edge_arrows = True, ax = axes[0])#, edge_labels = edge_labels)
    plot_lattice(WT_g, edge_arrows = internal_edges, ax = axes[1], edge_labels = edge_labels)

def test_all():
    from scipy import linalg as la
    from numpy import pi
    from koala.weaire_thorpe import cut_boundary, make_weire_thorpe_model
        
    V = 1
    W = 0.66
    E_bound = 2*(V+W)

    Es = np.linspace(-E_bound , E_bound , 100)

    #generate points
    n = 15
    points =  generate_bluenoise(k = 100, nx = n, ny = n)

    #generate graph
    g = generate_pbc_voronoi_adjacency(points)

    #color it
    ordered_edge_indices = clockwise_edges_about(vertex_i = 0, g=g)
    solveable, edge_labels = edge_color(g.adjacency, n_colors = 3, fixed = enumerate(ordered_edge_indices))

    #transform it to a Weaire-Thorpe model
    solveable, edge_labels = edge_color(g.adjacency, n_colors = 3, fixed = enumerate(ordered_edge_indices))
    WT_g = vertices_to_triangles(g, edge_labels);

    #label the internal and external edges
    edge_labels = np.where(np.arange(WT_g.adjacency.shape[0]) < g.adjacency.shape[0], 0, 1)
    internal_edges = np.where(np.arange(WT_g.adjacency.shape[0]) < g.adjacency.shape[0], False, True)

    def compute_observables(WT_g, internal_edges, Es):

        #solve the hamiltonian
        eigvals, eigvecs = make_weire_thorpe_model(WT_g, internal_edges = internal_edges, phi = 1.3, V = 1, W = 0.66)
        density = np.abs(eigvecs)
        IPR = 1 / np.sum(np.abs(eigvecs)**4, axis = 0) / len(WT_g.vertices)
        
        #bin the energies
        DOS, _ = np.histogram(eigvals, Es)
        return eigvals, eigvecs, DOS, density, IPR

    eigvals, eigvecs, DOS, density, IPR = compute_observables(WT_g, internal_edges, Es)

    #cut all the edges that cross boundaries
    kept_edges, WT_g_open = cut_boundary(WT_g)
    internal_edges_open = internal_edges[kept_edges]
    edge_labels_open = edge_labels[kept_edges]

    eigvals_open, eigvecs_open, DOS_open, density_open, IPR_open = compute_observables(WT_g_open, internal_edges[kept_edges], Es)

    from koala.plotting import plot_scalar

    fig, axes = plt.subplots(ncols = 4, figsize = (20,5))

    #plot the barcharts

    def colormap(a): return plt.get_cmap('viridis')((a - min(a)) / (max(a) - min(a)))
    print(min(IPR), max(IPR))
    print(min(IPR_open), max(IPR_open))

    width = (Es[1] - Es[0]) / (V + W)
    axes[0].bar(Es[:-1] / (V + W), DOS, width = width, align = 'edge', color = colormap(IPR))
    axes[2].bar(Es[:-1] / (V + W), DOS_open, width = width, align = 'edge', color = colormap(IPR_open))

    plot_lattice(WT_g, edge_arrows = internal_edges, ax = axes[1], edge_labels = edge_labels)

    E_target = 0.63 * (V + W)
    E_interval = np.array([0.60, 0.67]) * (V + W)

    closest_i = np.searchsorted(eigvals_open, E_target)
    E_closest = eigvals_open[closest_i]
    print(f"E_target = {E_target}, closest_i = {closest_i}, E_closest = {E_closest}")

    plot_lattice(WT_g_open, edge_arrows = internal_edges_open, ax = axes[3], edge_labels = edge_labels_open)

    in_interval = np.logical_and(E_interval[0] < eigvals_open, eigvals_open < E_interval[1])
    local_DOS_open = np.sum(density_open[:, in_interval], axis = 1)

    in_interval = np.logical_and(E_interval[0] < eigvals, eigvals < E_interval[1])
    local_DOS = np.sum(density[:, in_interval], axis = 1)

    plot_scalar(WT_g, local_DOS, ax = axes[1])
    plot_scalar(WT_g_open, local_DOS_open, ax = axes[3])
    #plot_wavefunction(WT_g_open, density_open[closest_i, :], ax = axes[3])
    #axes[0].axvline(x = E_closest / (V + W), linestyle = 'dotted')
    s = 0.1
    axes[3].set(xlim = (-s,1+s), ylim = (-s,1+s))

    axes[0].set(ylabel = "DOS", xlabel = "E / (V + W)")
    axes[3].set(title = "$|\psi|^2$")

    for i in [0,2]:
        axes[0].axvline(x = E_closest / (V + W), linestyle = 'dotted')
        for e in E_interval: 
            axes[i].axvline(x = e / (V+W), linestyle = 'dotted', color = 'k')
            
    axes[0].set(xlim = (-E_bound/(V+W), E_bound/(V+W)))