import numpy as np
from torch_geometric.utils import to_networkx
import matplotlib.pyplot as plt


def plot_2d(data, lim=10):
    # The graph to visualize
    G = to_networkx(data)
    pos = data.pos.numpy()

    # Extract node and edge positions from the layout
    node_xyz = np.array([pos[v, :2] for v in sorted(G)])
    edge_xyz = np.array([(pos[u, :2], pos[v, :2]) for u, v in G.edges()])

    # Create the 2D figure
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Plot the nodes - alpha is scaled by "depth" automatically
    ax.scatter(*node_xyz.T, s=100, c=data.atoms.numpy(), cmap="rainbow")

    # Plot the edges
    for vizedge in edge_xyz:
        ax.plot(*vizedge.T, color="tab:gray")

    # Turn gridlines off
    # ax.grid(False)
        
    # Suppress tick labels
    # for dim in (ax.xaxis, ax.yaxis, ax.zaxis):
    #     dim.set_ticks([])
        
    # Set axes labels and limits
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim([-lim, lim])
    ax.set_ylim([-lim, lim])
    ax.set_aspect('equal', 'box')

    # fig.tight_layout()
    plt.show()


def plot_3d(data, lim=10):
    # The graph to visualize
    G = to_networkx(data)
    pos = data.pos.numpy()

    # Extract node and edge positions from the layout
    node_xyz = np.array([pos[v] for v in sorted(G)])
    edge_xyz = np.array([(pos[u], pos[v]) for u, v in G.edges()])

    # Create the 3D figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Plot the nodes - alpha is scaled by "depth" automatically
    ax.scatter(*node_xyz.T, s=100, c=data.atoms.numpy(), cmap="rainbow")

    # Plot the edges
    for vizedge in edge_xyz:
        ax.plot(*vizedge.T, color="tab:gray")

    # Turn gridlines off
    # ax.grid(False)
        
    # Suppress tick labels
    # for dim in (ax.xaxis, ax.yaxis, ax.zaxis):
    #     dim.set_ticks([])
        
    # Set axes labels and limits
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_xlim([-lim, lim])
    ax.set_ylim([-lim, lim])
    ax.set_zlim([-lim, lim])

    # fig.tight_layout()
    plt.show()
