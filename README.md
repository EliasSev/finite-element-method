# Finite element method
This program uses finite elements (a mesh of triangles) to solve differential equations. The following equations
are supported so far
- the Poisson equation in 2D
- the heat equation in 1D and 2D
- the wave equation in 2D

## How to use
Download the required packages by using the following command in cmd (in the finite-element-method path):
```bash
pip install -r requirements.txt
```
Define a mesh with the `Mesh` class:

```python
mesh = Mesh()
# either read in a .msh file
mesh.read_mesh(path = "mesh_files/donut.msh") 
# or use a built in method to create a mesh
# mesh.square_mesh(n=50, x_range=(0,1), y_range=(0,1))
```

Pass the mesh into the `Fem1D` or `Fem2D` class, and use one of the following solvers:
`Fem1D.heat_solver_Dirichlet`, `Fem2D.Poisson_solver_Dirichlet`, `Fem2D.heat_solver_Dirichlet`, `Fem2D.heat_solver_Neumann`, `Fem2D.wave_solver_Dirichlet`:

```python
fem = Fem2D(mesh)  # or Fem1D
# use a solver
fem.heat_solver_Dirichlet(
    T = 1, 
    m = 100, 
    f = lambda x , y: 0, 
    u0 = lambda x, y: x + y)
# get the solution
solution = fem.solution
```

Create a video or image of the solution using the `MeshGraphics1D` or `MeshGraphics2D` class:

```python
graphics = MeshGraphics2D(fem)
# generate video in case of time dependent problem
graphics.create_solution_video(
    title      = 'Heat equation FEM solution'
    video_name = 'HeatDirichletExample', 
    style      = 'surface', 
    crange     = (0, 1), 
    cmap       = 'coolwarm', 
    fps        = 20
)

# or in case of time independent problem, do
# graphics.create_solution_image('ExampleName')
```

## Examples
Multiple examples are given in `/examples` and imported into `main.py`.

## Assembly
Methods for assembly of the mass matrix, stiffness matrix and load vector are avaiable.

## Theory
The code is based on the theory found in Larson G. L. & Bengzon F., The Finite Element Method: Theory, Implementation, and Applications (2013), chapter 1 to 5.

## Issues
The wave solver gets unstable after many time steps, and is sensitive to uneven meshes.
