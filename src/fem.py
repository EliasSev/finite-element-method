import numpy as np
import scipy as sp
from time import time
from scipy import sparse
from numpy.typing import NDArray
from collections.abc import Callable
from .graphics import progress_bar


class Fem:
    def __init__(self) -> None:
        self._solution = None
        self._horizontal_line = '-' * 47

    @property
    def solution(self) -> None:
        if self._solution is None:
            raise AttributeError("Solution not generated yet. Use a solver to generate a solution")
        return self._solution


class Fem2D(Fem):
    def __init__(self, mesh) -> None:
        """
        Finite element method class. Given a mesh, solve a PDE on the mesh with any of
        the available solvers: heat_solver, wave_solver.

        mesh, Mesh: Instance of the Mesh class.
        """

        super().__init__()
        self.mesh = mesh
        self.P = mesh.P
        self.C = mesh.C
        self._boundary_nodes = mesh.boundary_nodes
        self._interior_nodes = mesh.interior_nodes

    def _triangle_area(self, x: NDArray, y: NDArray) -> float:
        """
        Get the area of a single triangle element.
        
        x, array : (3,) array of x-coordiantes for the 3 nodes.
        y, array : (3,) array of y-coordiantes for the 3 nodes.
        
        return, float: Area of triangle.
        """

        area = 0.5 * (x[0]*(y[1]-y[2]) + x[1]*(y[2]-y[0]) + x[2]*(y[0]-y[1]))
        if area <= 0:
            raise ValueError("Invalid area:", area)
        return area

    def _hat_grad(self, x: NDArray, y: NDArray) -> tuple:
        """
        Get the area and gradients of the 3 hat functions
        which are located on the 3 nodes of a single triangle.
        
        x, array : (3,) array of x-coordiantes for the 3 nodes.
        y, array : (3,) array of y-coordiantes for the 3 nodes.

        return tuple(float, array, array): area, b, c.
        """

        area = self._triangle_area(x, y)
        b = np.array([y[1]-y[2], y[2]-y[0], y[0]-y[1]])
        c = np.array([x[2]-x[1], x[0]-x[2], x[1]-x[0]])
        b = b / (2 * area)
        c = c / (2 * area)
        return area, b, c

    def stiffness_assembler_2D(self) -> NDArray: 
        """
        Assemble the stiffness matrix.
        
        return array : stiffness matrix A of size (np, np)
        """

        n = len(self.P)
        A = np.zeros((n, n))
        
        # for triangle K in triangulation
        for K in self.C:
            x, y = self.P[K][:, 0],  self.P[K][:, 1]
            area, b, c = self._hat_grad(x, y)
            AK = (np.outer(b, b.T) + np.outer(c, c.T)) * area
            
            # add to global stiffness matrix
            loc2glb = np.ix_(K, K)  # (K^T, K)
            A[loc2glb] += AK
            
        return A

    def mass_assembler_2D(self) -> NDArray:
        """
        Assemble the mass matrix.

        return array : (np, np) stiffness matrix M
        """

        n = len(self.P)
        M = np.zeros((n, n))
        
        # for triangle K in triangulation
        for K in self.C:
            x, y = self.P[K][:, 0],  self.P[K][:, 1]
            area = self._triangle_area(x, y)
            MK = np.array([[2, 1, 1], [1, 2, 3], [1, 1, 2]]) * area / 12
            
            # add to global stiffness matrix
            loc2glb = np.ix_(K, K)  # (K^T, K)
            M[loc2glb] += MK
            
        return M

    def load_assembler_2D(self, f: Callable[[float, float], float]):
        """
        Assemble the load vector.

        f, func : A function f = f(x, y) to calculate the load.

        return array : Load vector b of size (np,)
        """

        n = len(self.P)
        b = np.zeros(n)

        for K in self.C:
            # create local load bK
            x, y = self.P[K][:, 0],  self.P[K][:, 1]
            area = self._triangle_area(x, y)
            bK = np.array([f(x[i], y[i]) for i in range(3)]) * area / 3
                        
            # add to global load vector
            loc2glb = K
            b[loc2glb] += bK
            
        return b 

    def heat_solver_Dirichlet(
            self, 
            T: float, 
            m: int, 
            f: Callable[[float, float], float], 
            u0: Callable[[float, float], float]
            ) -> NDArray:
        """
        Solver for the 2D heat equation ut - (uxx + uyy) = f using finite elements
        for space and backward-Euler for time with Dirichlet boundary condition.

        T, float : Stopping time
        m, int   : Number of time intervals
        f, func  : Heat source fucntion f = f(x, y)
        u0, func : Initial condition function, u0 = u0(x, y) 

        return, array : Solutions of each time step, size (m+1, n)
        """

        print("Backward Euler heat solver (Dirichlet, 2D)\n" + self._horizontal_line)
        t0 = time()

        k = T / m          # time step size
        n_p = len(self.P)  # number of nodes
        
        # assemble
        A = self.stiffness_assembler_2D()
        M = self.mass_assembler_2D()
        b = self.load_assembler_2D(f)
        
        # interior nodes
        int_idx = np.ix_(self._interior_nodes, self._interior_nodes)  # (v^T, v)
        A = sp.sparse.csr_array(A[int_idx])
        M = sp.sparse.csr_array(M[int_idx])
        b = b[self._interior_nodes]
        
        # initial condition
        xi = np.array([u0(N[0], N[1]) for N in self.P[self._interior_nodes]])
        
        # backward-Euler method
        xi_record = [xi]
        for l in range(m):
            xi = sparse.linalg.spsolve(M + k * A, M @ xi + k * b)
            xi_record.append(xi)
            
            # print progress
            progress_bar(l + 2, m + 1, end_text=f" ({time()-t0:.1f}s)")
        print('\n')

        # add boundary
        Xi = np.zeros((m+1, n_p))
        for i, xi in enumerate(xi_record):
            xi_glb = np.zeros(n_p)
            xi_glb[self._interior_nodes] = xi
            Xi[i] = xi_glb
        
        self._solution = Xi
        return Xi

    def heat_solver_Neumann(
            self, 
            T: float, 
            m: int, 
            f: Callable[[float, float], float], 
            u0: Callable[[float, float], float]
            ) -> NDArray:
        """
        Solver for the 2D heat equation ut - (uxx + uyy) = f using finite elements
        for space and backward-Euler for time with Neumann boundary condition.

        T, float : Stopping time
        m, int   : Number of time intervals
        f, func  : Heat source fucntion f = f(x, y)
        u0, func : Initial condition function, u0 = u0(x, y) 

        return, array : Solutions of each time step, size (m+1, n)
        """

        print("Backward Euler heat solver (Neumann, 2D)\n" + self._horizontal_line)
        t0 = time()
        
        # time step size
        k = T / m
        
        # assemble
        A = sp.sparse.csr_array(self.stiffness_assembler_2D())
        M = sp.sparse.csr_array(self.mass_assembler_2D())
        b = self.load_assembler_2D(f)
        
        # initial condition
        xi = np.array([u0(N[0], N[1]) for N in self.P])
        
        # Backward Euler
        xi_record = [xi]
        for l in range(m):
            xi = sparse.linalg.spsolve(M + k * A, M @ xi + k * b)
            xi_record.append(xi)

            # progress
            progress_bar(l + 2, m + 1, end_text=f" ({time()-t0:.1f}s)")
        print('\n')

        Xi = np.array(xi_record)
        self._solution = Xi
        return Xi

    def wave_solver_Dirichlet(
            self, 
            T: float, 
            m: int, 
            c: float, 
            f: Callable[[float, float], float], 
            gD: Callable[[float, int], float], 
            dnodes: tuple,
            drop_tol: float = 1e-4,
            rtol: float = 1e-6
            ) -> NDArray:
        """
        Solve the 2-dimensional wave equation utt + c^2 uyy = f(x, y) with 0 on the boundary
        using the finite elements for space and the Crank-Nicolson method for time.

        T, float        : Stopping time
        m, int          : Number of time intervals
        c, int          : Wave speed
        f, func         : Function f(x: float, y: float) -> float
        gD, func        : Boundary condition at Dirichlet boundary nodes, gD(dt: float, timestep: int) -> float.
        dnodes, tuple   : Index of Dirichlet nodes (triangle index).
        drop_tol, float : Ilu tolerance.
        rtol, float     : gmres tolerance.
        """
        
        print(f"Crank-Nicolson wave solver (Dirichlet, 2D)\n" + self._horizontal_line)
        t0 = time()

        k = T / m          # time step size
        n_p = len(self.P)  # number of nodes
        dnodes = np.array(dnodes)
        
        # assemble
        A = sp.sparse.csr_array(self.stiffness_assembler_2D())
        M = sp.sparse.csr_array(self.mass_assembler_2D())
        b = self.load_assembler_2D(f)
        
        # sparse matrix in block-form
        b = np.block([np.zeros(n_p), k * b])
        B_left = sparse.bmat([[M,          -M*k/2],
                              [A*c**2*k/2, M     ]], format='csc')
        B_right = sparse.bmat([[M,           M*k/2],
                               [-A*c**2*k/2, M    ]], format='csc')
        
         # precondition
        ilu = sparse.linalg.spilu(B_left, drop_tol=drop_tol)  # appoximate inverse
        precond = sparse.linalg.LinearOperator(B_left.shape, ilu.solve)

        # initial conditions
        xi = np.zeros(n_p)
        eta = np.zeros(n_p)
        sol = np.block([xi, eta])
        
        # Crank-Nicolson method
        xi_record = [xi]
        for l in range(m):
            # solve using gmres with precoditioner
            sol, info = sparse.linalg.gmres(B_left, B_right @ sol + b, M=precond, rtol=rtol)
            if info != 0:
                raise ValueError(f"GMRES solver failed to converge at iteration {l}. Info: {info}")
        
            # Dirichlet condition
            sol[dnodes] = gD(k, l)
            xi = sol[:n_p]
            xi_record.append(xi)
            
            # print progress
            progress_bar(l + 2, m + 1, end_text=f" ({time()-t0:.1f}s)")
        print('\n')
        
        Xi = np.array(xi_record)
        self._solution = Xi
        return Xi

    def Poisson_solver_Dirichlet(
            self, 
            f: Callable[[float, float], float], 
            gD: Callable[[float, float], float],
            ) -> NDArray:
        """
        Solve the 2-dimensional Poisson equation u_xx + u_yy = f(x, y) with
        gD(x, y) on the boundary.

        f, func  : Source function f = f(x, y)
        gD, func : Boundary function gD = gD(x, y)
        """
        
        print(f"Poisson solver (Dirichlet, 2D)\n" + self._horizontal_line)
        t0 = time()
        
        # get indicies of interior nodes
        int_idxs = np.ix_(self._interior_nodes, self._interior_nodes)  # (K^T, K)
        n_p = len(self.P)  # number of nodes
        
        A = self.stiffness_assembler_2D()
        b = self.load_assembler_2D(f)
        xig = np.array([gD(N[0], N[1]) for N in self.P[self._boundary_nodes]])
        
        # solve for interior points
        A0 = A[int_idxs]                      # ni x ni
        Ag = A[np.ix_(self._interior_nodes, self._boundary_nodes)]  # ni x ng
        b0 = b[self._interior_nodes]          # ni x 1
        bg = Ag @ xig                         # ni x 1
        xi0 = np.linalg.solve(A0, b0 - bg)    # ni x 1

        # combine interior solution and edge solution
        xi = np.zeros(n_p)            
        xi[self._interior_nodes] = xi0  # add interior values             
        xi[self._boundary_nodes] = xig  # add edge values
        
        progress_bar(1, 1, end_text=f" ({time()-t0:.1f}s)")
        print('\n')

        self._solution = xi
        return xi
    

class Fem1D(Fem):
    def __init__(self, X: NDArray) -> None:
        super().__init__()
        self.X = X

    def stiffness_assembler_1D(self) -> NDArray:
        """
        Assemble the (n+1)x(n+1) stiffness matrix A.

        x, array : array of n+1 points {x_i} making up n intervals {I_i}

        Returns array A of size (n+1)x(n+1).
        """
        n = len(self.X) - 1
        A = np.zeros((n+1 ,n+1))

        # add local contribution A^Ii
        for i in range(n):
            hi = self.X[i+1] - self.X[i]
            Ai = np.array([[1, -1], [-1, 1]]) / hi
            A[i:(i+2), i:(i+2)] += Ai
            
        return A

    def mass_assembler_1D(self) -> NDArray:
        """
        Assemble the (n+1)x(n01) mass matrix M using Simpsons rule.

        x, array : array of n+1 points {x_i} making up n intervals {I_i}

        Returns array M of size (n+1)x(n+1).
        """
        n = len(self.X) - 1
        M = np.zeros((n+1 ,n+1))

        # add local contribution M^Ii
        for i in range(n):
            hi = self.X[i+1] - self.X[i]
            Mi = np.array([[2, 1], [1, 2]]) * hi / 6
            M[i:(i+2), i:(i+2)] += Mi
            
        return M

    def load_assembler_1D(self, f: Callable[[float], float]) -> NDArray:
        """
        Assemble the (n+1)x(1) load vector b using the trapezoidal rule.

        x, array : array of n+1 points {x_i} making up n intervals {I_i}
        f, func  : function f = f(x) 

        Returns array b of size (n+1).
        """
        n = len(self.X) - 1
        b = np.zeros(n+1)

        # add local contribution b^Ii
        for i in range(n):
            hi = self.X[i+1] - self.X[i]
            bi = np.array([f(self.X[i]), f(self.X[i+1])]) * hi / 2
            b[i:(i+2)] += bi
            
        return b

    def heat_solver_Dirichlet(
            self, 
            m: int, 
            T: float, 
            f: Callable[[float], float], 
            u0: Callable[[float], float]
            ) -> NDArray:
        """
        Finite element method with backward Euler for the 1D heat equation
        with 0 on the boundary.

        m, int   : number of intervals in time, J1, ..., Jm
        T, float : stopping time, J = (0, T]
        f, func  : function f = f(x)
        u0, func : initial condition, u0 = u0(x)

        Returns m+1 xi's as an array Xi of size (m+1, n+1)
        """

        print(f"Poisson solver (Dirichlet, 1D)\n" + self._horizontal_line)
        t0 = time()

        # n+1 space points and m+1 time points
        n = len(self.X) - 1
        t = np.linspace(0, T, m+1)

        # only interior points must be used
        xi = u0(self.X[1:-1])
        A = self.stiffness_assembler_1D()[1:-1, 1:-1]
        M = self.mass_assembler_1D()[1:-1, 1:-1]
        b = self.load_assembler_1D(f)[1:-1]

        # backwards Euler method
        xi_record = [xi]
        for l in range(m):
            kl = t[l+1] - t[l]
            xi = np.linalg.solve(M + kl * A, M @ xi + kl * b)
            xi_record.append(xi)

            progress_bar(l + 2, m + 1, end_text=f" ({time()-t0:.1f}s)")
        print('\n')
            
        # add boundaries
        Xi = np.zeros((m+1, n+1))
        Xi[:, 1:-1] = np.array(xi_record)
        self._solution = Xi
        return Xi
