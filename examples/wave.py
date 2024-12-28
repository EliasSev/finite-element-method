import numpy as np
from src.mesh import Mesh
from src.fem import Fem2D
from src.graphics import MeshGraphics2D

def wave2D_Dirichlet_example():
    # FEM params
    n = 75  # n x n square mesh
    x_range = (0, 1)  # mesh x-range
    y_range = (0, 1)  # mesh y-range
    f  = lambda x, y: 0
    gD = lambda k, l: np.sin(8 * np.pi * k * l)  # dt=k, timstep=l
    Dirichlet_nodes = tuple(range(8))
    c = 0.75  # wave speed
    T = 2     # stop time
    m = 199   # time intervals

    # video params
    cmap = 'inferno'
    plot_style = 'heatmap'
    vid_name = 'WaveExample'
    fps = 30

    # create mesh
    mesh = Mesh()
    mesh.square_mesh(n, x_range, y_range)

    # solve
    fem2D = Fem2D(mesh)
    fem2D.wave_solver_Dirichlet(T, m, c, f, gD, Dirichlet_nodes)

    # create video
    vrange = (-.5, .5)
    graphics = MeshGraphics2D(fem2D, cmap)
    graphics.create_images(plot_style, vrange)
    graphics.create_video(vid_name, fps)
