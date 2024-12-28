import numpy as np
from src.mesh import Mesh
from src.fem import Fem2D
from src.graphics import MeshGraphics2D


def heat2D_Neumann_example():
    # FEM params
    mesh_path = "mesh_files/small_gap.msh"
    u0 = lambda x, y: 0 if x > 3.5 else 1
    f =  lambda x, y: 0
    T = 2    # stop time
    m = 199  # time intervals

    # video params
    cmap = 'coolwarm'
    plot_style = 'surface'
    vid_name = 'NeumannExample'
    fps = 15

    # create triangulation mesh
    mesh = Mesh()
    mesh.read_msh(mesh_path)

    # FEM
    fem2D = Fem2D(mesh)
    fem2D.heat_solver_Neumann(T, m, f, u0)
    
    # create video
    vrange = (np.min(fem2D.solution[0]), np.max(fem2D.solution[0]))
    graphics = MeshGraphics2D(fem2D, cmap)
    graphics.create_images(plot_style, vrange)
    graphics.create_video(vid_name, fps)


def heat2D_Dirichlet_example():
    # FEM params
    mesh_path = "mesh_files/donut.msh"
    u0 = lambda x, y: 0 if x > 0 else 1
    f =  lambda x, y: np.sin(np.pi * x) * np.cos(np.pi * y)
    T = 0.5  # stop time
    m = 199  # time intervals

    # video params
    cmap = 'coolwarm'
    plot_style = 'surface'
    vid_name = 'DirichletExample'
    fps = 15

    # create triangulation mesh
    mesh = Mesh()
    mesh.read_msh(mesh_path)

    # FEM
    fem2D = Fem2D(mesh)
    fem2D.heat_solver_Dirichlet(T, m, f, u0)
    
    # create video
    vrange = (np.min(fem2D.solution[0]), np.max(fem2D.solution[0]))
    graphics = MeshGraphics2D(fem2D, cmap)
    graphics.create_images(plot_style, vrange)
    graphics.create_video(vid_name, fps)

