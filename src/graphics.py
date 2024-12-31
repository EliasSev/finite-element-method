import cv2
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from time import time, sleep
from numpy.typing import NDArray
import concurrent.futures
from functools import wraps


def handle_mpl_bug(func):
    """
    Wrapper function to handle the matplotlib 3.9.0 'PyCapsule_New called with null pointer' bug.

    Try to create image 'max_attempts' times in case of matplotlib bug
    ValueError: PyCapsule_New called with null pointer, which comes from
    GetForegroundWindow() returning NULL. This can be avioded by waiting a bit (~0.1s) 
    and retrying (bug fixed in matplotlib 3.10, https://github.com/matplotlib/matplotlib/pull/28269)
    """

    def wrapper(*args, **kwargs):
        attempts = 0
        max_attempts = 15
        image_created = False
        while not image_created:
            try:
                func(*args, **kwargs)  # try to run function
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
                    raise e  # any other exception
                
    return wrapper


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
        images = [f'img{i}.jpg' for i in range(self._n_images)]
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

    def create_solution_video(
            self, 
            video_name: str, 
            title: str, 
            color: str, 
            fps: int = 15, 
            video_format: str = 'mp4'
            ) -> None:
        """
        Create a video using the images in /images.

        vid_name, str     : Name of video.
        fps, int          : Frames per seconds used for video.
        video_format, str : Video format. 'mp4' or 'avi'.
        """

        self._create_images(title, color)
        self._create_video(video_name, fps, video_format)

    def _create_images(self, title: str, color: str) -> None:
        """
        Generate images for all time steps in self.solutions. Images used to generate a video.

        title, str : Title used in the images.
        color, str : Color of line.
        """

        print("Creating images\n" + self._horizontal_line)
        t0 = time()

        m = len(self.solution)
        arguments = [(title, self._images_path + f"img{i}.jpg", solution_i, color,
                      self.figsize) for i, solution_i in enumerate(self.solution)]

        # context manager will wait for all processes to finish
        with concurrent.futures.ProcessPoolExecutor() as executor:
            # create a process for each argument
            finished = 0
            futures = [executor.submit(self._create_image, *args) for args in arguments]
            for f in concurrent.futures.as_completed(futures):
                f.result()
                finished += 1
                progress_bar(finished, m, end_text=f" ({time()-t0:.1f}s)")

        print('\n')
        self._n_images = len(self.solution)

    def _create_image(
            self,
            title: str, 
            image_name: str, 
            solution: NDArray, 
            color: str,
            figsize: tuple
            ) -> None:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_xlim(self.X[0], self.X[-1])
        ax.set_ylim(np.min(solution), np.max(solution))
        ax.plot(self.X, solution, color)
        ax.set_title(title)
        plt.savefig(image_name)
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

    def _create_images(self, title: str, style: str, crange: tuple, cmap: str) -> None:
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
        arguments = [(title, self._images_path + f"img{i}.jpg", solution_i,
                      crange, cmap) for i, solution_i in enumerate(self.solution)]
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

    def _heatmap_plot(self,
        title: str, 
        image_name: str, 
        solution: NDArray, 
        crange: tuple, 
        cmap: str
        ) -> None:
        """
        Create a heat map of the given solution and mesh.
        
        title, str         : Title used for the plot.
        image_name, str    : Path and name of image.
        solution, np.array : Finite element solution.
        crange, tuple      : Fixed value range used for coloring.
        """

        x, y = self.P[:, 0], self.P[:, 1]
        triang = mpl.tri.Triangulation(x, y, self.C)
        vmin, vmax = crange
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

    def _surface_plot(
        self,
        title: str, 
        image_name: str, 
        solution: NDArray, 
        crange: str, 
        cmap: str
        ) -> None:
        """
        Create a heat map of the given solution and mesh.
        
        title, str         : Title used for the plot.
        image_name, str    : Path and name of image.
        solution, np.array : Finite element solution.
        crange, tuple      : Fixed value range used for coloring.
        """

        x, y = self.P[:, 0], self.P[:, 1]
        vmin, vmax = crange
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_trisurf(x, y, solution, triangles=self.C, cmap=cmap, edgecolor='none', vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlim((c*2 for c in crange))

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

