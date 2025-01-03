import meshio
import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
from collections import Counter
from numpy.typing import NDArray

class Mesh:
    def __init__(self) -> None:
        """
        x_range, tuple : The x-axis range of the mesh domain. 
        y_range, tuple : The y-axis range of the mesh domain.
        """

        self._boundary_nodes = None
        self._interior_nodes = None
        self._P = None  # Point matrix
        self._C = None  # Connectivity matrix
        self._no_mesh_message = "The mesh is not defined yet. Read in a mesh or define a mesh"

    @property
    def P(self) -> NDArray:
        if self._P is None:
            raise AttributeError(self._no_mesh_message)
        return self._P

    @property
    def C(self) -> NDArray:
        if self._C is None:
            raise AttributeError(self._no_mesh_message)
        return self._C

    @property
    def interior_nodes(self) -> NDArray:
        if self._boundary_nodes is None:
            self._find_boundary_nodes()
        return self._interior_nodes

    @property
    def boundary_nodes(self) -> NDArray:
        if self._boundary_nodes is None:
            self._find_boundary_nodes()
        return self._boundary_nodes

    def _find_boundary_nodes(self) -> None:
        """
        Find edge nodes and interior nodes of mesh.

        return tuple(array, array) :  (edge_points, int_points)
        """

        edges = []
        for triangle in self.C:
            i, j, k = triangle
            edges.append(tuple(sorted((i, j))))
            edges.append(tuple(sorted((j, k))))
            edges.append(tuple(sorted((k, i))))
        
        # dict tracking occurrences of each edge
        edge_counts = Counter(edges)
        boundary_edges = [edge for edge, count in edge_counts.items() if count == 1]
        
        # all unique points from boundary edges
        edge_nodes = set()
        for edge in boundary_edges:
            edge_nodes.update(edge)
            
        edge_nodes = np.array(list(edge_nodes))
        int_nodes = np.setdiff1d(self.C, edge_nodes)
        self._boundary_nodes = edge_nodes
        self._interior_nodes = int_nodes

    def square_mesh(self, n: int, x_range: tuple, y_range: tuple) -> tuple:
        """
        Create an n times n rectangular mesh.

        n, int : The number of nodes of the side of the rectangle.
        x_range, tuple : Width (left, right).
        y_range, tuple : Height (bottom, top).
        """
        
        x_start, x_end = x_range
        y_start, y_end = y_range
        x = np.linspace(x_start, x_end, n)
        y = np.linspace(y_start, y_end, n)

        X, Y = np.meshgrid(x, y)
        P = np.column_stack([X.ravel(), Y.ravel()])
        C = sp.spatial.Delaunay(P).simplices

        self._P, self._C = P, C
        return P, C
    
    def read_msh(self, path: str) -> tuple:
        """
        Read a .msh file and return point matrix and connectivity matrix.

        path, str: Path to .msh file.
        """

        mesh = meshio.read(path)
        P = mesh.points

        C = None
        for cell in mesh.cells:
            if cell.type == "triangle":
                C = cell.data
        
        self._P, self._C = P, C
        return P, C

    def mesh_plot(
            self, 
            path: str = './results/MeshPlot.jpg',
            show_labels: bool = False, 
            figsize: tuple = (7, 7), 
            marker: str = 'o'
            ) -> None:
        """
        Create a plot of the mesh.

        show_labels, bool : Label each triangle and point.
        figsize, tuple    : Figure size used for matplotlib.
        marker, str       : Marker style used for points.
        """

        x, y = self.P[:, 0], self.P[:, 1]
        triang = mpl.tri.Triangulation(x, y, self.C)

        # plot the triangulation
        plt.figure(figsize=figsize)
        plt.triplot(triang, marker=marker, markersize=2)
        plt.gca().set_aspect('equal')
        plt.title(f"Mesh plot ({len(self.P)} nodes, {len(self.C)} triangles)", size=15)

        if show_labels:
            # node labels
            for i, (xi, yi) in enumerate(zip(x, y)):
                plt.text(xi, yi, f'$N_{{{i}}}$', fontsize=12,
                        ha = 'left', va = 'bottom', c = 'orange')
        
            # triangle labels
            for i, triangle in enumerate(self.C):
                plt.text(x[triangle].mean(), y[triangle].mean(), f'$K_{{i}}$',
                         fontsize = 12, ha = 'center', va = 'center', color = 'blue') 
        
        plt.savefig(path)
        plt.close()