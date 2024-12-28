import numpy as np
from src.mesh import Mesh
from src.fem import Fem2D
from src.graphics import MeshGraphics2D


def poisson2D_Dirichlet_example():
    # FEM params
    mesh_path = "mesh_files/donut.msh"
    f = lambda x, y: 2 * np.pi**2 * np.sin(0.25 * np.pi * x) * np.sin(0.25 * np.pi * y)
    gD = lambda x, y: 0

    # image params
    cmap = 'viridis'
    picture_name = 'PoissonExample'

    # create triangulation mesh
    mesh = Mesh()
    mesh.read_msh(mesh_path)

    # FEM
    fem2D = Fem2D(mesh)
    fem2D.Poisson_solver_Dirichlet(f, gD)

    graphics = MeshGraphics2D(fem2D, cmap)
    graphics.create_solution_image(picture_name)
