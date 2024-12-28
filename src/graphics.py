import matplotlib as mpl
import matplotlib.pyplot as plt
import cv2
from time import sleep


class MeshGraphics:
    """
    Mesh graphics parent class
    """
    def __init__(self):
        self._images_path = "./images/"
        self._results_path = "./results/"
        self._n_images = None


    def create_images(self):
        """
        Save image in self._images_path, which will be used to generate a video
        using the create_video method
        """
        raise NotImplementedError("'create_images' method is not implemented.")


    def create_video(self, vid_name, fps, video_format='mp4'):
        """
        Create a video using the images in /images.

        vid_name, str     : Name of video.
        fps, int          : Frames per seconds used for video.
        video_format, str : Video format. 'mp4' or 'avi'.
        """
        
        print("Creating video\n" + '-'*40)

        if self._n_images is None:
            raise AttributeError("Number of images not defined. Try creating the images with 'create_images'.")
        
        fformat = {'mp4': 'mp4v', 'avi': 'XVID'}
        if video_format not in fformat.keys():
            raise ValueError(f"Invalid video format: {video_format}. Valid formats: {', '.join(fformat.keys())}")

        # create names
        images = [f"img{i}.jpg" for i in range(self._n_images)]
        output_video = self._results_path + vid_name + '.' + video_format

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


class MeshGraphics1D(MeshGraphics):
    def __init__(self, fem1D, color, figsize=(9, 4)) -> None:
        """
        fem2D, Fem2D : Instance of Fem2D class.
        color, str    : Matplotlib color for line.
        """

        super().__init__()
        self.X = fem1D.X
        self.solution = fem1D.solution
        self.color = color
        self.figsize = figsize
        self.xlim = (self.X[0], self.X[-1])
        self.ylim = (min(self.solution), max(self.solution))


    def create_images(self, solution, i):

        print("Creating image\n" + '-'*40)

        # figure setup
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.set_xlim(self.xlim[0], self.xlim[1])
        ax.set_ylim(self.ylim[0], self.ylim[1])
        ax.plot(self.X, solution, self.color)
        ax.set_title("1D heat equation using backward Euler")
        ax.set_label(f"t = ")
        plt.legend()
        plt.savefig(self._images_path + f"/img{i}.jpg")
        plt.close()
    

class MeshGraphics2D(MeshGraphics):
    def __init__(self, fem2D, cmap) -> None:
        """
        fem2D, Fem2D : Instance of Fem2D class.
        cmap, str    : Matplotlib color map.
        """

        super().__init__()
        self.P = fem2D.mesh.P
        self.C = fem2D.mesh.C
        self.solution = fem2D.solution
        self.cmap = cmap
        self.n_nodes = len(self.P)
        self.plot_functions = {"heatmap": self.plot_2D, 'surface': self.plot_3D}

    
    def create_solution_image(self, name, figsize=(7,7)):

        print("Creating image\n" + '-'*40)

        x, y = self.P[:, 0], self.P[:, 1]
        triang = mpl.tri.Triangulation(x, y, self.C)
        output_image = self._results_path + name + '.png'
        
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        heatmap_num = ax.tripcolor(triang, self.solution, cmap=self.cmap)
        ax.set_title("Numerical solution $u_h$")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.axis('equal')
        fig.colorbar(heatmap_num, ax=ax)

        plt.savefig(output_image)
        plt.close()

        print(f"Image saved as: {output_image}")
    

    def triangulation_plot(self, show_labels=False, figsize=(7,7), marker='o'):
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
                plt.text(xi, yi, f'$N_{{{i}}}$',
                        fontsize=12,
                        ha = 'left',
                        va = 'bottom',
                        c = 'orange')
        
            # triangle labels
            for i, triangle in enumerate(self.C):
                plt.text(x[triangle].mean(), y[triangle].mean(), f'$K_{{i}}$',
                         fontsize = 12,
                         ha = 'center',
                         va = 'center',
                         color = 'blue') 
                
    
    def plot_2D(self, solution, vrange, i):
        """
        Create a heat map of the solution on a mesh.

        solution, np.array : Finite element solution.
        vrange, tuple      : Fixed value range used for coloring.
        i, int             : Image index used for naming.
        """

        x, y = self.P[:, 0], self.P[:, 1]
        vmin, vmax = vrange
        triang = mpl.tri.Triangulation(x, y, self.C)

        fig, ax = plt.subplots(1, 1, figsize=(8, 8), constrained_layout=True)
        heatmap_num = ax.tripcolor(triang, solution, cmap=self.cmap, vmin=vmin, vmax=vmax)
        ax.set_title(f"Numerical solution $u_h$, $n_p$={self.n_nodes}x{self.n_nodes}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.axis('equal')
        cbar = fig.colorbar(heatmap_num, ax=ax)
        cbar.formatter = mpl.ticker.FormatStrFormatter('%.2f')
        cbar.update_ticks()
        
        plt.savefig(self._images_path + f"img{i}.jpg")
        plt.close()

    
    def plot_3D(self, solution, vrange, i):
        """
        Create a surface plot the solution on a mesh.

        solution, np.array : Finite element solution.
        vrange, tuple      : Fixed value range used for coloring.
        i, int             : Image index used for naming.
        """

        x, y = self.P[:, 0], self.P[:, 1]
        vmin, vmax = vrange

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_trisurf(x, y, solution, triangles=self.C, cmap=self.cmap, edgecolor='none', vmin=vmin, vmax=vmax)
        ax.set_title(f"Numerical solution $u_h$, $n_p$={self.n_nodes}x{self.n_nodes}")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        
        if vrange:
            zrange = tuple(v*1.5 for v in vrange)
            ax.set_zlim(zrange)

        plt.savefig(self._images_path + f"img{i}.jpg")
        plt.close()

    
    def create_images(self, plot_style, vrange):
        """
        Generate images for all time steps in self.solutions. Images used to generate a video.

        plot_style, str : Plot style for images. Surface or heat.
        vrange, tuple   : Fixed value range used for coloring.
        """

        print("Creating images\n" + '-'*40)

        available_styles = self.plot_functions.keys()
        if plot_style not in available_styles:
            raise ValueError(f"Unknown plotting style: \"{plot_style}\". Available styles: {available_styles}")

        m = len(self.solution) - 1
        plot_func = self.plot_functions[plot_style]
        
        for i, solution_i in enumerate(self.solution):
            # try to create image 3 times in case of matplotlib bug
            # ValueError: PyCapsule_New called with null pointer, which comes from
            # GetForegroundWindow() returning NULL. This can be avioded by waiting a bit (~0.1s) 
            # and retrying (bug fixed in matplotlib 3.10, https://github.com/matplotlib/matplotlib/pull/28269)
            attempts = 0
            max_attempts = 5
            image_created = False
            while not image_created:
                try:
                    plot_func(solution_i, vrange, i)
                    image_created = True
                except Exception as e:
                    if isinstance(e, ValueError) and str(e) == "PyCapsule_New called with null pointer":
                        attempts += 1
                        if attempts >= max_attempts:
                            print(f"Creating image {i} failed {max_attempts} time(s):")
                            raise e
                        print(f"Creating image {i} failed ({str(e)}). Retrying.")
                        sleep(0.1)
                    else:
                        # any other exception
                        raise e
                    
            progress_bar(i + 1, m + 1)
        print('\n')

        # give value once all images are created
        self._n_images = len(self.solution)


def progress_bar(step, total_steps, bar_length=30):
    """
    Simple progress bar.

    step, int        : current step of process
    total_steps, int : total number of steps in progress
    bar_length, int  : length of bar
    """
    
    filled = int(bar_length * step / total_steps)
    print(f"[{filled * '#' :<{bar_length}}] {step}/{total_steps}", end='\r')
