# Libraries
import numpy as np
from multiprocessing import Pool
import matplotlib.pyplot as plt
from IPython.display import Image
import imageio
from pathlib import Path
from tqdm import tqdm


class LifeLikeCA:
    """
    Simulate Life-like cellular automata.
    """

    def __init__(self, rule: int = 5, matrix: np.ndarray = None):
        """
        Initialize the cellular automata.

        Parameters
        ----------
        rule: int
            The rule of the cellular automata.
            It should be between 0 and 2^18 - 1.
        matrix: np.ndarray
            The initial state of the cellular automata. Default is None.
        """
        # Rule validation
        if not 0 <= rule <= 2**18-1:
            raise ValueError(f"Rule must be in [0, {2**18 - 1}].")
        
        # Matrix validation
        if matrix is None or matrix.ndim != 2:
            raise ValueError("Matrix must be a 2D numpy array.")
        
        if not np.isin(matrix, [0, 1]).all():
            raise ValueError("Matrix must contain only 0s and 1s.")

        # Set attributes
        self.rule = rule
        self.matrix = matrix
        self.n, self.m = matrix.shape
        
        # Print the rules
        print(self)

    def get_rules(self):
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

    def moore_neighbors(self, row, col):
        """
        Get the Moore neighbors of a cell.
        It includes both edges and corners.

        Parameters
        ----------
        row: int
            The row of the cell.
        col: int
            The column of the cell.

        Returns
        -------
        np.ndarray
            The Moore neighbors of the cell.
        """
        # Get the indices (the order doesn't matter)
        indices = np.array([
            #[row, col],        

            [(row - 1) % self.n, col],            # up
            [(row + 1) % self.n, col],            # down
            [row, (col - 1) % self.m],            # left
            [row, (col + 1) % self.m],            # right
            
            [(row - 1) % self.n, (col - 1) % self.m],  # upper left
            [(row - 1) % self.n, (col + 1) % self.m],  # upper right
            [(row + 1) % self.n, (col - 1) % self.m],  # lower left
            [(row + 1) % self.n, (col + 1) % self.m]   # lower right
        ])
        
        return self.matrix[indices[:, 0], indices[:, 1]]
    
    def parallel_count(self, processors = 4):
        """
        Count the neighbors of all cells with parallelization.

        Parameters
        ----------
        processors: int
            The number of processors to use. Default is 4.

        Returns
        -------
        np.ndarray
            The counts of the neighbors.
        """
        # Create all coordinates
        coords = np.array([(i, j) for i in range(self.n) for j in range(self.m)])
        
        # Split coordinates into chunks: one per processor
        coord_chunks = np.array_split(coords, processors)
        
        # Prepare arguments for each process
        args = [(self.matrix, chunk, self.n, self.m) for chunk in coord_chunks]
        
        # Parallelization
        with Pool(processors) as pool:

            # Call the count for each chunk of coordinates
            results = pool.map(self.count, args)
        
        # Combine them
        return np.concatenate(results).reshape(self.n, self.m)

    def count(self, args):
        """
        Count the neighbors in a chunk of coordinates.

        Parameters
        ----------
        args: tuple
            The arguments to be unpacked.

        Returns
        -------
        np.ndarray
            The counts of the neighbors.
        """
        # Unpack the arguments
        matrix, coords, self.n, self.m = args

        # Initialize the counts
        counts = np.zeros(len(coords), dtype = int)
        
        # Loop over the coordinates: enumerate gives the index and the value
        for i, (row, col) in enumerate(coords):

            # Get the neighbors
            neighbors = self.moore_neighbors(row, col)

            # Count them
            counts[i] = np.sum(neighbors)
        
        return counts
    
    def images(self, generations: int, processors: int, path: str = None,
               shape: tuple = (6, 6), cmap: str = "Blues"):
        """
        Generate images of the cellular automata.

        Parameters
        ----------
        generations: int
            The number of generations to simulate.
        processors: int
            The number of processors to use.
        path: str
            The path to save the images. Default is None.
        shape: tuple
            The shape of the images. Default is (6, 6).
        cmap: str
            The colormap to use. Default is "Blues".
        """
        # Create the path
        Path(path).mkdir(exist_ok = True)
        
        # Get the rules
        birth, survive = self.get_rules()

        # Save the initial state
        self.image(state = self.matrix, path = path + "0000.png",
                   shape = shape, cmap = cmap)
        
        # Create an empty matrix
        new_matrix = np.zeros_like(self.matrix, dtype = int)

        # Loop over generations
        for m in range(generations):

            # Get the neighbors
            all_neighbours = self.parallel_count(processors)

            # Apply the rules
            # 1. Birth
            birth_mask = (self.matrix == 0) & np.isin(all_neighbours, birth)

            # 2. Survival
            survive_mask = (self.matrix == 1) & np.isin(all_neighbours, survive)

            # Update the matrix
            new_matrix[:] = np.where(birth_mask | survive_mask, 1, 0)
            
            # Update state
            self.matrix = new_matrix.copy()

            # Save the image
            self.image(state = self.matrix, path = f"{path}{str(m+1).zfill(4)}.png",
                       shape = shape, cmap = cmap)
        
        # Print a message
        print(f"All images have been generated.\n")

    def gif(self, path: str = None, name: str = "evolution", fps: int = 5,
            display: bool = False, loop: int = 2):
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

        # Read images with progress bar
        for f in tqdm(files, desc = "Reading images"):
            images.append(imageio.imread(f))
    
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
        return f"Born with {self.get_rules()[0]} and survive with {self.get_rules()[1]} neighbors. Otherwise, die."