import numpy as np
from src.mesh import Mesh
from src.fem import Fem2D
from src.graphics import MeshGraphics2D


def poisson2D_Dirichlet_example():
    # create mesh
    mesh = Mesh()
    mesh.read_msh(path = "mesh_files/donut.msh")

    # fem solver
    fem2D = Fem2D(mesh)
    fem2D.Poisson_solver_Dirichlet(
        f  = lambda x, y: 2 * np.pi**2 * np.sin(0.25 * np.pi * x) * np.sin(0.25 * np.pi * y),
        gD = lambda x, y: 0
    )

    # create image
    graphics = MeshGraphics2D(fem2D)
    graphics.create_solution_image(
        name = 'PoissonDirichletExample', 
        cmap = 'viridis'
    )
