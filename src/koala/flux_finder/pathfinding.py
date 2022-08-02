import numpy as np
from queue import PriorityQueue

###############################
# General A* implementation   #
###############################


class PathFindingError(Exception):
    pass


def a_star_search_forward_pass(
    start: int, goal: int, heuristic, adjacency, early_stopping: bool, maxits: int
):
    frontier = PriorityQueue()  # represents nodes we might explore next
    frontier.put((0, start))  # nodes are represented by (priority, index)

    # came_from represents how we found a way to each node, also tells us which plaq we've visited
    # came_from[p_i] = (p_j, e_k) means we got to node i from j through edge e_k
    came_from = {}
    cost_so_far = {}  # represents the current lowest known cost d(n) for a node
    estimated_total_cost = {}

    # put the start node in
    came_from[start] = (None, None)
    cost_so_far[start] = 0
    estimated_total_cost[start] = 0

    for i in range(maxits):
        if frontier.empty():
            break

        _, current = frontier.get()
        if current == goal:
            return came_from, cost_so_far

        for next, shared_edge in zip(*adjacency(current)):
            if early_stopping and next == goal:
                came_from[next] = (current, shared_edge)
                return came_from, cost_so_far

            new_cost = cost_so_far[current] + heuristic(current, next)
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost + heuristic(next, goal)
                estimated_total_cost[next] = priority
                frontier.put((priority, next))
                came_from[next] = (current, shared_edge)

    raise PathFindingError(
        f"Could not find a path from start={start} to goal={goal} in {maxits} steps."
    )


def a_star_search_backward_pass(came_from, start, goal):
    nodes, edges = [
        goal,
    ], []
    while nodes[-1] != start:
        next_plaquette, shared_edge = came_from[nodes[-1]]
        edges.append(shared_edge)
        nodes.append(next_plaquette)
    return nodes[::-1], edges[::-1]


###############################
# specialised to plaquettes   #
###############################

from ..lattice import Lattice
from ..graph_utils import (
    adjacent_plaquettes,
    vertex_neighbours,
    closest_plaquette,
    closest_vertex,
)


def straight_line_length(a, b) -> float:
    "Return the shortest path between two points on the unit square."
    return np.linalg.norm(a - b, ord=2)


def periodic_straight_line_length(a, b) -> float:
    "Return the shortest path between two points on the unit torus."
    delta = a - b
    delta = (
        1 - delta[delta > 0.5]
    )  # if the x or y delta is > 0.5 then we actually want 1 - delta
    return np.linalg.norm(delta, ord=2)


def path_between_plaquettes(
    l: Lattice,
    start: int,
    goal: int,
    heuristic=straight_line_length,
    early_stopping: bool = True,
    maxits: int = 1000_000,
):
    """Find a path along dual edges between two plaquettes. Returns (vertices, edges) in the path.
    Note that the edges and dual edges are 1-1 so the same indices are used for interchangeably.
    If early_stopping = False it's guaranteed to be the shortest measured by center to center distance.
    If early_stopping = True it's probably still pretty short but it stops as soon as it finds a path.
    The heuristic function must give a lower bound on the distance, it could be swapped out for a periodic one for instance.

    :param l: The lattice to pathfind on.
    :type l: Lattice
    :param start: The index of the plaquette to start from.
    :type start: int
    :param goal: The index of the plaquette to go to.
    :type goal: int
    :param heuristic: The distance metric (and periodicity) to use, defaults to straight_line_length
    :type heuristic: function
    :param early_stopping: If True, return as soon as a path is found, defaults to True
    :type early_stopping: bool, optional
    :param maxits: How many iterations to do before giving up, defaults to 1000
    :type maxits: int, optional
    :return: plaquettes, edges, lists of plaquettes visited by a path and the indices of the edges used to get there.
    :rtype: Tuple[array, array]
    """
    # all information about the graph is stored in the adjacency(a) and heuristic(a,b) functions
    def adjacency(a):
        return adjacent_plaquettes(l, a)

    def pos(p):
        return l.plaquettes[p].center

    def _heuristic(a, b):
        return heuristic(pos(a), pos(b))

    came_from, _ = a_star_search_forward_pass(
        start, goal, _heuristic, adjacency, early_stopping, maxits
    )
    return a_star_search_backward_pass(came_from, start, goal)


def adjacent_vertices(lattice, a):
    return vertex_neighbours(lattice, a)


def path_between_vertices(
    l: Lattice,
    start: int,
    goal: int,
    heuristic=straight_line_length,
    early_stopping: bool = True,
    maxits: int = 1000,
):
    """Find a path along edges between two vertices. Returns (vertices, edges) in the path.
    Note that the edges and dual edges are 1-1 so the same indices are used for interchangeably.
    If early_stopping = False it's guaranteed to be the shortest measured by center to center distance.
    If early_stopping = True it's probably still pretty short but it stops as soon as it finds a path.
    The heuristic function must give a lower bound on the distance, it could be swapped out for a periodic one for instance.

    :param l: The lattice to pathfind on.
    :type l: Lattice
    :param start: The index of the vertex to start from.
    :type start: int
    :param goal: The index of the vertex to go to.
    :type goal: int
    :param heuristic: The distance metric (and periodicity) to use, defaults to straight_line_length
    :type heuristic: function
    :param early_stopping: If True, return as soon as a path is found, defaults to True
    :type early_stopping: bool, optional
    :param maxits: How many iterations to do before giving up, defaults to 1000
    :type maxits: int, optional
    :return: vertices, edges, lists of vertices visited by a path and the indices of the edges used to get there.
    :rtype: Tuple[array, array]
    """

    def adjacency(a):
        return adjacent_vertices(l, a)

    def pos(a):
        return l.vertices.positions[a]

    def _heuristic(a, b):
        return heuristic(pos(a), pos(b))

    came_from, _ = a_star_search_forward_pass(
        start, goal, _heuristic, adjacency, early_stopping, maxits
    )
    return a_star_search_backward_pass(came_from, start, goal)


def _round_the_back_metric(x=False, y=False):
    """
    A metric function that prefers to go around the major or minor axis of the torus
    Used to trick the path finder into a giving a path that goes around such an axis
    """

    def length(a, b) -> float:
        delta = a - b
        if x:
            delta[0] = 1 - delta[0]
        if y:
            delta[1] = 1 - delta[1]
        return np.linalg.norm(delta, ord=2)

    return length


def __loop(lattice, direction, kind, center=0.5):
    """Construct loops on the torus,
    either on the dual or the normal lattice
    and in either the x or y direction
    """
    if kind == "graph":
        closest_thing = closest_vertex
        path = path_between_vertices
    elif kind == "dual":
        closest_thing = closest_plaquette
        path = path_between_plaquettes
    else:
        assert False

    points = np.array([[0.1, center], [0.5, center], [0.9, center]])
    if direction == "y":
        points = points[:, ::-1]
    a, b, c = [closest_thing(lattice, p) for p in points]
    heuristic = _round_the_back_metric(x=(direction == "x"), y=(direction == "y"))

    t1, e1 = path(lattice, a, b)
    t2, e2 = path(lattice, b, c)
    t3, e3 = path(lattice, c, a, heuristic=heuristic)

    return np.concatenate([t1[:-1], t2[:-1], t3[:-1]], axis=0), np.concatenate(
        [e1, e2, e3], axis=0
    )


def graph_loop(lattice, direction="x", center=0.5):
    """Construct a loop winding around the torus in the 'direction' direction.
    If direction == 'x' then it will pass near y = center and vice versa

    Returns the indices of vertices and edges in the loop
    as well as +/-1 indicating if we traverse that edge in the forward or backward sense
    """
    verts, edges = __loop(lattice, direction=direction, kind="graph", center=center)
    directions = 2 * np.equal(verts, lattice.edges.indices[edges][:, 0]) - 1
    return verts, edges, directions


def dual_loop(lattice, direction="x"):
    """Construct a loop winding around the torus in the 'direction' direction.
    If direction == 'x' then it will pass near y = center and vice versa

    Returns the indices of vertices and edges in the loop
    """
    return __loop(lattice, direction=direction, kind="dual")
