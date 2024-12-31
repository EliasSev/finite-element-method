import numpy as np
from src.mesh import Mesh
from src.fem import Fem2D
from src.graphics import MeshGraphics2D

def wave2D_Dirichlet_example():
    # create mesh
    mesh = Mesh()
    mesh.square_mesh(
        n = 75,           # n by n grid
        x_range = (0, 1), # x-limit
        y_range = (0, 1)  # y-limit
    )

    # fem solver
    fem2D = Fem2D(mesh)
    fem2D.wave_solver_Dirichlet(
        T  = 2,               # stopping time
        m  = 199,             # time intervals
        c  = .75,             # wave speed
        f  = lambda x, y: 0,  # source function
        gD = lambda k, l: np.sin(8 * np.pi * k * l), # Dirichlet condition (dt=k, timstep=l)
        dnodes = tuple(range(8))  # Dirichlet nodes
    )

    graphics = MeshGraphics2D(fem2D)
    graphics.create_solution_video(
        title = 'Wave equation FEM solution',
        video_name = 'WaveDirichletExample', 
        style = 'heatmap', 
        crange     = (-0.3, 0.3), 
        cmap       = 'viridis', 
        fps        = 20
    )
