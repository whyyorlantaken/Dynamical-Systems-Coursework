# Libraries
import numpy as np
from scipy import ndimage 
import matplotlib.pyplot as plt
from IPython.display import Image
import imageio
from pathlib import Path

# Class
class LifeLikeCA:
    """
    Simulate Life-like cellular automata.

    Parameters
    ----------
    rule: int
        The rule of the cellular automata, should be between 0 and 2^18 - 1.
    ics: np.ndarray
        The initial state of the cellular automata as a 2D numpy array.
    """
    def __init__(self, 
                 rule: int = 5, 
                 res: int = 100,
                 ics: np.ndarray = None):
        """
        Initialize the cellular automata.

        Parameters
        ----------
        rule: int
            The rule of the cellular automata.
            It should be between 0 and 2^18 - 1.
        ics: np.ndarray
            The initial state of the cellular automata. Default is None.
        """
        # Rule validation
        if not 0 <= rule <= 2**18-1:
            raise ValueError(f"Rule must be in [0, {2**18 - 1}].")

        # Set attributes
        self.rule = rule

        # If no ics
        if ics is None:
            self.matrix = np.zeros((res, res), dtype = int)
            self.matrix[res // 2, res // 2] = 1
        else:
            self.matrix = ics

        # Matrix validation
        if self.matrix is None or self.matrix.ndim != 2:
            raise ValueError("Matrix must be a 2D numpy array.")
        
        if not np.isin(self.matrix, [0, 1]).all():
            raise ValueError("Matrix must contain only 0s and 1s.")
        
        # Shape
        self.n, self.m = self.matrix.shape
        
        # Rules extraction
        self.birth_rule, self.survive_rule = self._rules()
        
        # Moore neighborhood kernel for convolution
        self.kernel = np.array([[1, 1, 1],
                                [1, 0, 1], 
                                [1, 1, 1]])
        
        # Print the rules
        print(self)

    def _rules(self):
        """
        To get the birth and survive rules from the rule integer.
        It cannot be larger than 2^18 - 1.

        Returns
        -------
        np.ndarray
            The birth and survive rules.
        """
        # Convert the integer to binary
        binary = np.binary_repr(self.rule, width = 18)

        # Reverse the string
        binary = binary[::-1]

        # Extract the birth and survive rules
        birth = np.array([int(x) for x in binary[:9]])
        survive = np.array([int(x) for x in binary[9:]])

        # Get the indices of the 1s
        birth_rule = np.where(birth)[0]
        survive_rule = np.where(survive)[0]

        return birth_rule, survive_rule

    def count_neighbors(self):
        """
        Count neighbors using convolution.
        
        Returns
        -------
        np.ndarray
            Array of neighbor counts for each cell.
        """
        # Periodic boundary conditions with convolution
        return ndimage.convolve(self.matrix, self.kernel, mode = 'wrap')

    def images(self, 
               generations: int, 
               path: str = None,
               shape: tuple = (6, 6), 
               cmap: str = "Blues"):
        """
        Generate images of the cellular automata.

        Parameters
        ----------
        generations: int
            The number of generations to simulate.
        path: str
            The path to save the images. Default is None.
        shape: tuple
            The shape of the images. Default is (6, 6).
        cmap: str
            The colormap to use. Default is "Blues".
        """
        # Create the path
        Path(path).mkdir(exist_ok = True)

        # Save the initial state
        self.image(state = self.matrix, 
                   path = path + "0000.png",
                   shape = shape, 
                   cmap = cmap)
        
        # Pre-allocate matrix for reuse
        new_matrix = np.zeros_like(self.matrix, dtype=int)

        # Loop over generations
        for m in range(generations):

            # Neighbor counting
            neighbor_counts = self.count_neighbors()

            # Rule application
            birth_mask = (self.matrix == 0) & np.isin(neighbor_counts, self.birth_rule)
            survive_mask = (self.matrix == 1) & np.isin(neighbor_counts, self.survive_rule)

            # Update matrix
            new_matrix[:] = 0
            new_matrix[birth_mask | survive_mask] = 1
            
            # Swap matrices for next iteration
            self.matrix, new_matrix = new_matrix, self.matrix

            # Save the image
            self.image(state = self.matrix, 
                       path = f"{path}{str(m+1).zfill(4)}.png",
                       shape = shape, 
                       cmap = cmap)
        
        # Print a message
        print(f"All images have been generated.\n")

    def gif(self, 
            path: str = None, 
            name: str = "evolution", 
            fps: int = 5,
            display: bool = False, 
            loop: int = 2):
        """
        Generate a GIF from the images.

        Parameters
        ----------
        path: str 
            The path to the images. Default is None.
        name: str
            The name of the GIF. Default is "evolution".
        fps: int
            The frames per second. Default is 5.
        display: bool
            Whether to display the GIF. Default is False.
        loop: int
            The number of loops. Default is 2.

        Returns
        -------
        If display is True, it returns the GIF.
        """
        # Get all files
        files = Path(path).glob('*.png')

        # Sort them numerically
        files = sorted(files, key = lambda x: int(x.stem))
        
        # Create a list to save the images
        images = []

        # Read images
        for file in files:
            img = imageio.imread(file)
            images.append(img)
    
        # Save GIF
        imageio.mimsave(f'{path}/{name}.gif', images,
                        fps = fps, loop = loop)

        # Print the direction of the gif
        print(f"\nSaved {name}.gif at {path}.\n")

        # Display the GIF
        if display:
            
            print(f"Displaying the gif...\n")

            return Image(filename = f'{path}/{name}.gif')

    def image(self, state: np.ndarray = None, path: str = None, shape: tuple =  (5,5), dpi: int = 100,
              cmap: str = "Blues", show: bool = False, save: bool = True):
        """
        Plot the state of the cellular automata.

        Parameters
        ----------
        state: np.ndarray
            The state of the cellular automata.
        path: str
            The path to save the image. Default is None.
        shape: tuple
            The shape of the image. Default is (5, 5).
        dpi: int
            The dots per inch. Default is 100.
        cmap: str
            The colormap to use. Default is "Blues".
        show: bool
            Whether to show the image. Default is False.
        save: bool
            Whether to save the image. Default is True.
        """
        # Validate the state
        if state is None:
            raise ValueError("State cannot be None.")
        
        # Plot
        fig = plt.figure(figsize = shape, frameon = False, dpi = dpi)
        plt.imshow(state, cmap = cmap, aspect = 'equal', interpolation = "none")
        plt.axis("off")
        
        plt.gca().set_position([0, 0, 1, 1])
        plt.margins(0,0)
        
        if show:
            plt.show()
        if save:
            plt.savefig(path, bbox_inches='tight', pad_inches = 0, dpi = dpi)
        plt.close()
    
    def __str__(self):
        """
        To print the rules of the cellular automata.
        """
        return f"Born with {self.birth_rule} and survive with {self.survive_rule} neighbors."