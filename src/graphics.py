import cv2
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from time import time, sleep
from numpy.typing import NDArray
import concurrent.futures


class MeshGraphics:
    """Mesh graphics parent class."""

    def __init__(self) -> None:
        self._images_path = "./images/"
        self._results_path = "./results/"
        self._n_images = None
        self._horizontal_line = '-' * 47

    def _create_images(self) -> None:
        """
        Save image in self._images_path, which will be used to generate a video
        using the create_video method
        """
        raise NotImplementedError("'_create_images' method is not implemented.")

    def _create_video(self, video_name: str, fps: int, video_format: str='mp4') -> None:
        """
        Create a video using the images in /images.

        video_name, str   : Name of video.
        fps, int          : Frames per seconds used for video.
        video_format, str : Video format. 'mp4' or 'avi'.
        """
        
        print("Creating video\n" + self._horizontal_line)

        if self._n_images is None:
            raise AttributeError("Number of images not defined")
        
        fformat = {'mp4': 'mp4v', 'avi': 'XVID'}
        if video_format not in fformat.keys():
            raise ValueError(f"Invalid video format: {video_format}. Valid formats: {', '.join(fformat.keys())}")

        # create names
        images = [f"img{i}.jpg" for i in range(self._n_images)]
        output_video = self._results_path + video_name + '.' + video_format

        # define video dimension
        first_image = cv2.imread(self._images_path + images[0])
        height, width, layers = first_image.shape

        fourcc = cv2.VideoWriter_fourcc(*fformat[video_format])
        video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

        for image in images:
            img_path = self._images_path + image
            frame = cv2.imread(img_path)
            video.write(frame)

        video.release()
        print(f"fps            : {fps}")
        print(f"Duration       : {self._n_images/fps:.1f}s")
        print(f"Resolution     : {height} x {width}")
        print(f"Video saved as : {output_video}")
        print()


class MeshGraphics1D(MeshGraphics):
    def __init__(self, fem1D) -> None:
        super().__init__()
        self.X = fem1D.X
        self.solution = fem1D.solution

        # plotting params
        self.figsize = (9, 4)
        self.xlim = (self.X[0], self.X[-1])
        self.ylim = (np.min(self.solution), np.max(self.solution))

    def create_solution_video(self, video_name: str, color: str, fps: int=15, video_format: str='mp4') -> None:
        """
        Create a video using the images in /images.

        vid_name, str     : Name of video.
        fps, int          : Frames per seconds used for video.
        video_format, str : Video format. 'mp4' or 'avi'.
        """

        self._create_images(color)
        self._create_video(video_name, fps, video_format)

    def _create_images(self, color: str) -> None:

        print("Creating images\n" + self._horizontal_line)
        t0 = time()

        m = len(self.solution)
        for i, solution_i in enumerate(self.solution):
            name = f"/img{i}.jpg"
            self._create_image(solution_i, name, color)

            progress_bar(i + 1, m, end_text=f" ({time()-t0:.1f}s)")
        print('\n')

        # define once images are created
        self._n_images = len(self.solution)

    def _create_image(self, solution: NDArray, name: str, color: str) -> None:
        # figure setup
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.set_xlim(self.xlim[0], self.xlim[1])
        ax.set_ylim(self.ylim[0], self.ylim[1])
        ax.plot(self.X, solution, color)
        ax.set_title("1D heat equation using backward Euler")
        ax.set_label(f"t = ")
        plt.savefig(self._images_path + name)
        plt.close()
    

class MeshGraphics2D(MeshGraphics):
    def __init__(self, fem2D) -> None:
        """
        fem2D, Fem2D : Instance of Fem2D class.
        """

        super().__init__()
        self.P = fem2D.mesh.P
        self.C = fem2D.mesh.C
        self.solution = fem2D.solution
        self.n_nodes = len(self.P)
        self.plot_functions = {"heatmap": self._heatmap_plot, 'surface': self._surface_plot}

    def create_solution_video_old(
            self, 
            video_name: str, 
            plot_style: str, 
            crange: tuple, 
            cmap: str='viridis', 
            fps: int=15, 
            video_format: str='mp4'
            ) -> None:
        """
        Create a video using the images in /images.

        vid_name, str     : Name of video.
        plot_style, str   : Plot style for images. Surface or heat.
        fps, int          : Frames per seconds used for video.
        video_format, str : Video format. 'mp4' or 'avi'.
        crange, tuple     : Fixed value range used for coloring.
        """

        self._create_images(plot_style, crange, cmap)
        self._create_video(video_name, fps, video_format)

    def create_solution_video(
            self,
            title: str,
            video_name: str,
            style: str,
            crange: tuple, 
            cmap: str = 'viridis', 
            fps: int = 15, 
            video_format: str = 'mp4'
            ) -> None:
        """
        Generate a video of the solution.

        title, str        : Title used in the images.
        video_name, str   : Name of video.
        style, str        : Plotting style for images. Surface or heat.
        crange, tuple     : Fixed value range used for coloring.
        cmap, str         : Color map.
        fps, int          : Frames per seconds used for video.
        video_format, str : Video format. 'mp4' or 'avi'.
        """

        if style not in self.plot_functions.keys():
            raise ValueError(f"Unknown plotting style: \"{style}\". Available styles: {self.plot_functions.keys()}")

        self._create_images(title, style, crange, cmap)
        self._create_video(video_name, fps, video_format)

    def _create_images(self, title, style, crange, cmap) -> None:
        """
        Generate images for all time steps in self.solutions. Images used to generate a video.

        title, str    : Title used in the images.
        style, str    : Plotting style for images. Surface or heat.
        crange, tuple : Fixed value range used for coloring.
        cmap, str     : Color map.
        """

        print("Creating images\n" + self._horizontal_line)
        t0 = time()

        m = len(self.solution)
        arguments = [(
            title,
            self._images_path + f"img{i}.jpg",
            self.P,
            self.C,
            solution_i,
            crange,
            cmap
            ) for i, solution_i in enumerate(self.solution)]
        plotting_function = self.plot_functions[style]

        # context manager will wait for all processes to finish
        with concurrent.futures.ProcessPoolExecutor() as executor:
            # create a process for each argument
            finished = 0
            futures = [executor.submit(plotting_function, *args) for args in arguments]
            for f in concurrent.futures.as_completed(futures):
                f.result()
                finished += 1
                progress_bar(finished, m, end_text=f" ({time()-t0:.1f}s)")

        print('\n')
        self._n_images = len(self.solution)


    @staticmethod
    def _heatmap_plot(
        title: str, 
        image_name: str, 
        P: NDArray, 
        C: NDArray, 
        solution: NDArray, 
        crange: str, 
        cmap: str
        ) -> None:
        """
        Create a heat map of the given solution and mesh.
        
        title, str         : Title used for the plot.
        image_name, str    : Path and name of image.
        P, NDArray         : Point matrix.
        C, NDArray         : Connectivity matrix (triangles).
        solution, np.array : Finite element solution.
        crange, tuple      : Fixed value range used for coloring.
        """

        x, y = P[:, 0], P[:, 1]
        vmin, vmax = crange
        triang = mpl.tri.Triangulation(x, y, C)
        fig, ax = plt.subplots(1, 1, figsize=(8, 8), constrained_layout=True)
        heatmap_num = ax.tripcolor(triang, solution, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.axis('equal')

        # color bar
        cbar = fig.colorbar(heatmap_num, ax=ax)
        cbar.formatter = mpl.ticker.FormatStrFormatter('%.2f')
        cbar.update_ticks()
        
        plt.savefig(image_name)
        plt.close()

    @staticmethod
    def _surface_plot(
        title: str, 
        image_name: str, 
        P: NDArray, 
        C: NDArray, 
        solution: NDArray, 
        crange: str, 
        cmap: str
        ) -> None:
        """
        Create a heat map of the given solution and mesh.
        
        title, str         : Title used for the plot.
        image_name, str    : Path and name of image.
        P, NDArray         : Point matrix.
        C, NDArray         : Connectivity matrix (triangles).
        solution, np.array : Finite element solution.
        crange, tuple      : Fixed value range used for coloring.
        """

        x, y = P[:, 0], P[:, 1]
        vmin, vmax = crange
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_trisurf(x, y, solution, triangles=C, cmap=cmap, edgecolor='none', vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlim((c*4 for c in crange))

        plt.savefig(image_name)
        plt.close()

    def create_solution_image(
            self, 
            name: str, 
            cmap: str = 'viridis', 
            figsize: tuple = (7, 7)
            ) -> None:
        """
        Create an image from self.solution.

        name, str      : Image name.
        cmap, str      : Color map.
        figsize, tuple : Image size.
        """

        print("Creating image\n" + self._horizontal_line)

        x, y = self.P[:, 0], self.P[:, 1]
        triang = mpl.tri.Triangulation(x, y, self.C)
        output_image = self._results_path + name + '.png'
        
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        heatmap_num = ax.tripcolor(triang, self.solution, cmap=cmap)
        ax.set_title("Numerical solution $u_h$")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.axis('equal')
        fig.colorbar(heatmap_num, ax=ax)

        plt.savefig(output_image)
        plt.close()

        print(f"Image saved as: {output_image}")

    def _create_images_old(self, plot_style: str, crange: tuple, cmap: str) -> None:
        """
        Generate images for all time steps in self.solutions. Images used to generate a video.

        plot_style, str : Plot style for images. Surface or heat.
        crange, tuple   : Fixed value range used for coloring.
        cmap, str       : Color map.
        """

        print("Creating images\n" + self._horizontal_line)
        t0 = time()

        available_styles = self.plot_functions.keys()
        if plot_style not in available_styles:
            raise ValueError(f"Unknown plotting style: \"{plot_style}\". Available styles: {available_styles}")

        m = len(self.solution)
        plot_func = self.plot_functions[plot_style]
        
        for i, solution_i in enumerate(self.solution):
            self._handle_mpl_bug(plot_func, solution_i, crange, i, cmap)

            progress_bar(i + 1, m, end_text=f" ({time()-t0:.1f}s)")
        print('\n')

        # define once images are created
        self._n_images = len(self.solution)         
    
    def _plot_2D_old(self, solution: NDArray, crange: tuple, i: int, cmap: str) -> None:
        """
        Create a heat map of the solution on a mesh.

        solution, np.array : Finite element solution.
        crange, tuple      : Fixed value range used for coloring.
        i, int             : Image index used for naming.
        """

        x, y = self.P[:, 0], self.P[:, 1]
        vmin, vmax = crange
        triang = mpl.tri.Triangulation(x, y, self.C)

        fig, ax = plt.subplots(1, 1, figsize=(8, 8), constrained_layout=True)
        heatmap_num = ax.tripcolor(triang, solution, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(f"Numerical solution $u_h$, $n_p$={self.n_nodes}x{self.n_nodes}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.axis('equal')
        cbar = fig.colorbar(heatmap_num, ax=ax)
        cbar.formatter = mpl.ticker.FormatStrFormatter('%.2f')
        cbar.update_ticks()
        
        plt.savefig(self._images_path + f"img{i}.jpg")
        plt.close()
    
    def _plot_3D_old(self, solution: NDArray, crange: tuple, i: int, cmap: str) -> None:
        """
        Create a surface plot the solution on a mesh.

        solution, np.array : Finite element solution.
        crange, tuple      : Fixed value range used for coloring.
        i, int             : Image index used for naming.
        """

        x, y = self.P[:, 0], self.P[:, 1]
        vmin, vmax = crange

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_trisurf(x, y, solution, triangles=self.C, cmap=cmap, edgecolor='none', vmin=vmin, vmax=vmax)
        ax.set_title(f"Numerical solution $u_h$, $n_p$={self.n_nodes}x{self.n_nodes}")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        
        if crange:
            zrange = tuple(v*1.5 for v in crange)
            ax.set_zlim(zrange)

        plt.savefig(self._images_path + f"img{i}.jpg")
        plt.close()

    @staticmethod
    def _handle_mpl_bug(plot_func, *args, max_attempts=15):
        """
        Try to create image 'max_attempts' times in case of matplotlib bug
        ValueError: PyCapsule_New called with null pointer, which comes from
        GetForegroundWindow() returning NULL. This can be avioded by waiting a bit (~0.1s) 
        and retrying (bug fixed in matplotlib 3.10, https://github.com/matplotlib/matplotlib/pull/28269)
        """

        attempts = 0
        image_created = False
        while not image_created:
            try:
                plot_func(*args)
                image_created = True
            except Exception as e:
                if isinstance(e, ValueError) and str(e) == "PyCapsule_New called with null pointer":
                    attempts += 1
                    if attempts >= max_attempts:
                        print(f"Creating image failed {max_attempts} time(s):")
                        raise e
                    print(f"Creating image failed ({str(e)}). Retrying.")
                    sleep(0.1)
                else:
                    # any other exception
                    raise e
    
    def triangulation_plot(
            self, 
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
        plt.title(r"Triangulation $\mathcal{K}$", size=15)

        if show_labels:
            # node labels
            for i, (xi, yi) in enumerate(zip(x, y)):
                plt.text(xi, yi, f'$N_{{{i}}}$', fontsize=12,
                        ha = 'left', va = 'bottom', c = 'orange')
        
            # triangle labels
            for i, triangle in enumerate(self.C):
                plt.text(x[triangle].mean(), y[triangle].mean(), f'$K_{{i}}$',
                         fontsize = 12, ha = 'center', va = 'center', color = 'blue') 


def progress_bar(
        step: int, 
        total_steps: int, 
        bar_length: int = 30,
        end_text: str = ' test'
        ) -> None:
    """
    Simple progress bar.

    step, int        : current step of process
    total_steps, int : total number of steps in progress
    bar_length, int  : length of bar
    end_text, str    : add additional text at the end of the bar
    """
    
    filled = int(bar_length * step / total_steps)
    text = f"[{filled * '#' :<{bar_length}}] {step}/{total_steps}"
    print(text + end_text, end='\r')
