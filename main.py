import numpy as np
from src.mesh import Mesh
from src.fem import Fem1D, Fem2D
from src.graphics import MeshGraphics1D, MeshGraphics2D

# examples
from examples.heat import heat2D_Neumann_example, heat2D_Dirichlet_example, heat1D_Dirichlet_example
from examples.wave import wave2D_Dirichlet_example
from examples.poisson import poisson2D_Dirichlet_example


if __name__ == '__main__':

    # Examples
    #heat2D_Neumann_example()
    #heat2D_Dirichlet_example()
    wave2D_Dirichlet_example()
    #poisson2D_Dirichlet_example()
    #heat1D_Dirichlet_example()
