import numpy as np
from src.mesh import Mesh
from src.fem import Fem1D, Fem2D
from src.graphics import MeshGraphics1D, MeshGraphics2D


def heat2D_Neumann_example():
    # create mesh
    mesh = Mesh()
    mesh.read_msh(path = "mesh_files/small_gap.msh")

    # fem solver
    fem2D = Fem2D(mesh)
    fem2D.heat_solver_Neumann(
        T  = 2,
        m  = 199,
        f  = lambda x, y: 0,
        u0 = lambda x, y: 0 if x > 3.5 else 1
    )

    # generate video
    graphics = MeshGraphics2D(fem2D)
    graphics.create_solution_video(
        title      = 'Heat equation FEM solution',
        video_name = 'HeatNeumannExample', 
        style = 'surface', 
        crange     = (0, 1), 
        cmap       = 'coolwarm', 
        fps        = 20
    )


def heat2D_Dirichlet_example():
    # create mesh
    mesh = Mesh()
    mesh.read_msh(path = "mesh_files/donut.msh")

    # fem solver
    fem2D = Fem2D(mesh)
    fem2D.heat_solver_Dirichlet(
        T  = 0.5,
        m  = 199,
        f  = lambda x, y: np.sin(np.pi * x) * np.cos(np.pi * y),
        u0 = lambda x, y: 0 if x > 0 else 1
    )

    # generate video
    graphics = MeshGraphics2D(fem2D)
    graphics.create_solution_video(
        title      = 'Heat equation FEM solution',
        video_name = 'HeatDirichletExample', 
        style = 'surface', 
        crange     = (0, 1), 
        cmap       = 'coolwarm', 
        fps        = 20
    )

def heat1D_Dirichlet_example():
    # create grid
    X = np.linspace(-1, 1, 100)

    # fem solver
    fem1D = Fem1D(X)
    fem1D.heat_solver_Dirichlet(
        m  = 500,
        T  = 1,
        f  = lambda x: x**2,
        u0 = lambda x: np.heaviside(x, 0)
    )

    # generate video
    graphics = MeshGraphics1D(fem1D)
    graphics.create_solution_video(
        video_name = 'HeatDirichletExample1D',
        color = 'red',
        fps = 15
    )
    
