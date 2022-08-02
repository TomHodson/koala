import itertools
import re

import matplotlib
import numpy as np
import scipy
import scipy.interpolate
from matplotlib import (
    patheffects as path_effects,
)  # https://stackoverflow.com/questions/11578760/matplotlib-control-capstyle-of-line-collection-large-number-of-lines
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection

from . import graph_utils
from .graph_utils import clockwise_edges_about, vertex_neighbours
from .lattice import LatticeException
from .voronization import Lattice, generate_point_array

import mpl_toolkits.mplot3d as a3
from math import pi
from matplotlib.colors import to_rgb

import polyscope as ps

#### New plotting interface ####
def torus(R1, R2, offsets=[0, 0]):
    def transform(points):
        u = (points[..., 0] - offsets[0]) * 2 * pi
        v = (points[..., 1] - offsets[1]) * 2 * pi

        x = (R1 + R2 * np.cos(v)) * np.cos(u)
        y = (R1 + R2 * np.cos(v)) * np.sin(u)
        z = R2 * np.sin(v)

        new_shape = list(points.shape)
        new_shape[-1] = 3
        new_points = np.zeros(dtype=float, shape=new_shape)
        new_points[..., 0] = x
        new_points[..., 1] = y
        new_points[..., 2] = z
        return new_points

    return transform


transform3D = torus(1, 0.5)


def plot_vertices(
    lattice: Lattice,
    labels: np.ndarray = 0,
    color_scheme: np.ndarray = "black",
    subset: np.ndarray = slice(None, None, None),
    **kwargs,
):
    """Plot the vertices of a lattice.
    This uses matplotlib.pyplot.scatter under the hood and you may
    pass in any keyword to be passed along to scatter.
    You can use this to, for instance, plot some of the vertices as
     triangles and some as squares.

    :param lattice: The lattice to use.
    :type lattice: Lattice
    :param labels: int or array of ints specifying the colors, defaults to 0. May be the same size as the vertices or of the subset.
    :type labels: np.ndarray, optional
    :param color_scheme: List or array of colors, defaults to ['black', ]
    :type color_scheme: np.ndarray, optional
    :param subset: An array of indices, boolean array or slice that selects which elements to plot, defaults to plotting all.
    :type subset: np.ndarray, optional
    :param ax: The axis to plot on, defaults to plt.gca()
    :type subset: axis, optional
    """
    labels, colors, color_scheme, subset, ax = _process_plot_args(
        lattice, ax, labels, color_scheme, subset, lattice.n_vertices, kwargs
    )

    args = dict(
        x=lattice.vertices.positions[subset, 0],
        y=lattice.vertices.positions[subset, 1],
        c=colors,
        zorder=3,
    )
    args.update(**kwargs)  # doing this means the user can override zorder
    ax.scatter(**args)
    return ax


def plot_edges(
    lattice: Lattice,
    labels: np.ndarray = 0,
    color_scheme: np.ndarray = ["k", "r", "b"],
    subset: np.ndarray = slice(None, None, None),
    directions: np.ndarray = None,
    arrow_head_length=None,
    **kwargs,
):
    """
    Plot the edges of a lattice with optional arrows.
    This uses matplotlib.collections.LineColection under the hood and you may
    pass in any keyword to be passed along to it.
    Note that arrays for alpha or linestyle don't currently work since they would have to be tiled correctly, and are not currently.

    If directions is not none, arrows are plotted from the first vertex to the second unless direction[i] == -1

    :param lattice: The lattice to use.
    :type lattice: Lattice
    :param labels: int or array of ints specifying the colors, defaults to 0. May be the same size as the vertices or of the subset.
    :type labels: np.ndarray, optional
    :param color_scheme: List or array of colors, defaults to ['black', ]
    :type color_scheme: np.ndarray, optional
    :param subset: An array of indices, boolean array or slice that selects which elements to plot, defaults to plotting all.
    :type subset: np.ndarray, optional
    :param directions: An array of arrow directions +/-1, defaults to None.
    :type directions: np.ndarray, optional
    :param ax: The axis to plot on, defaults to plt.gca()
    :type subset: axis, optional
    """
    labels, colors, color_scheme, subset = _process_plot_args(
        lattice, labels, color_scheme, subset, lattice.n_edges, kwargs
    )
    edges_args = dict()
    edges_args.update(kwargs)

    original_edges = lattice.edges.indices[subset]
    vertices3D = transform3D(lattice.vertices.positions)
    vertices = []
    edges = []
    for i, edge in enumerate(original_edges):
        edges.append([2 * i, 2 * i + 1])
        vertices.extend(vertices3D[edge])

    ps_edges = ps.register_curve_network(
        "edges", np.array(vertices), np.array(edges), **edges_args
    )
    colors = np.array([to_rgb(c) for c in colors])
    ps_edges.add_color_quantity("color", colors, defined_on="edges", enabled=True)

    return ps_edges


def plot_plaquettes(
    lattice: Lattice,
    labels: np.ndarray = 0,
    color_scheme: np.ndarray = ["r", "b", "k"],
    subset: np.ndarray = slice(None, None, None),
    **kwargs,
):
    """
    Plot the plaquettes of a lattice.
    This uses matplotlib.collections.PolyColection under the hood and you may
    pass in any keyword to be passed along to it.
    Note that currently the calls are done per plaquette so you can't for instance have multiple alpha values.
    Adding a color argument overides the color_scheme and labels.

    :param lattice: The lattice to use.
    :type lattice: Lattice
    :param labels: int or array of ints specifying the colors, defaults to 0. May be the same size as the vertices or of the subset.
    :type labels: np.ndarray, optional
    :param color_scheme: List or array of colors, defaults to ['black', ]
    :type color_scheme: np.ndarray, optional
    :param subset: An array of indices, boolean array or slice that selects which elements to plot, defaults to plotting all.
    :type subset: np.ndarray, optional
    :param ax: The axis to plot on, defaults to plt.gca()
    :type subset: axis, optional
    """

    labels, colors, color_scheme, subset = _process_plot_args(
        lattice, labels, color_scheme, subset, lattice.n_plaquettes, kwargs
    )

    indices = np.arange(lattice.n_plaquettes)[subset]
    plaquettes = lattice.plaquettes[subset]

    triangle_sets = []
    triangle_colors = []
    for p_i, color, p in zip(indices, colors, plaquettes):
        N = p.n_sides
        N_verts = lattice.n_vertices
        triangles = np.array(
            [[p.vertices[i], p.vertices[(i + 1) % N], N_verts + p_i] for i in range(N)]
        )

        triangle_sets.extend(triangles)
        color = to_rgb(color)
        triangle_colors.extend([color for _ in triangles])

    vertices = transform3D(lattice.vertices.positions)
    p_centers = transform3D(np.array([p.center for p in lattice.plaquettes]))

    verts = np.concatenate([vertices, p_centers])

    mesh_args = dict(smooth_shade=True, material="clay", edge_width=0)
    mesh_args.update(kwargs)

    plaquettes = ps.register_surface_mesh(
        f"plaquettes", verts, np.array(triangle_sets), **mesh_args
    )
    plaquettes.add_color_quantity(
        "coloring", np.array(triangle_colors), enabled=True, defined_on="faces"
    )
    return plaquettes


def plot_dual(lattice, subset=slice(None, None), **kwargs):
    """Given a lattice, plot the edges in it's dual or a subset of them, other args are passed through to plot_edges.

    :param lattice: The lattice to use.
    :type lattice: Lattice
    :param subset: a subset of edges to plot, defaults to all.
    :type subset: slice, boolean array or indices, optional

    :return: The dual lattice represented as a second Lattice object.
    :rtype: Lattice
    """
    st_as_lattice = graph_utils.make_dual(lattice, subset)
    plot_edges(st_as_lattice, **kwargs)
    return st_as_lattice


def _plot_edge_arrows(
    colors, edges, directions, linecollection, unit_cell, arrow_head_length=None
):
    n_edges = edges.shape[0]
    linewidth = linecollection.get_linewidths()[
        0
    ]  # currently don't support multiple linewidths
    for color, (end, start), dir in zip(colors, edges, directions):
        start, end = [start, end][::dir]
        center = 1 / 2 * (end + start)
        length = np.linalg.norm(end - start)
        head_length = arrow_head_length or min(0.2 * length, 0.02 * linewidth / 1.5)
        direction = head_length * (start - end) / length
        arrow_start = center - direction
        ax.arrow(
            x=arrow_start[0],
            y=arrow_start[1],
            dx=direction[0],
            dy=direction[1],
            color=color,
            head_width=head_length,
            head_length=head_length,
            width=0,
            zorder=4,
            head_starts_at_zero=True,
            length_includes_head=True,
        )


def _broadcast_args(arg, subset, N, dtype=int):
    """Normalise an argument for plotting that can take three forms:
        1) a single thing of type [dtype]
        2) an array of size l.n_vertices, l.n_edges or l.n_plaquettes
        3) a smaller array that matches the subset
    Returns an array of type 3
    """
    # Fix 1) if it's just a single int, broadcast it to the size of the lattice.
    if isinstance(arg, dtype):
        arg = np.full(N, arg, dtype=dtype)

    # make sure it's a numpy array (of the right type) and not a list.
    arg = np.array(arg).astype(dtype)

    # if it refers to the entire lattice, subset it down
    subset_size = np.sum(np.ones(N)[subset], dtype=int)
    if arg.shape[0] == N:
        arg = arg[subset]
    elif arg.shape[0] == subset_size:
        arg = arg
    else:
        raise ValueError(
            f"Argument shape {arg.shape} should be either lattice.n_* ({N}) or the size of the subset ({subset_size})"
        )

    return arg


def _process_plot_args(lattice, labels, color_scheme, subset, N, kwargs):
    """
    Deals with housekeeping operations common to all plotting functions.
    Specifically:
        Broadcast single values to be the size of the lattice.
        Allow labels to refer to either the whole lattice or the subset.
    """
    if isinstance(color_scheme, str):
        color_scheme = [
            color_scheme,
        ]
    color_scheme = np.array([to_rgb(c) for c in color_scheme])
    if "color" in kwargs:
        color_scheme[0] = kwargs["color"]
    subset = np.arange(N)[subset]
    labels = _broadcast_args(labels, subset, N, dtype=int)

    colors = color_scheme[labels]

    return labels, colors, color_scheme, subset
