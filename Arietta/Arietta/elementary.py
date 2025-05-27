# Library
import numpy as np

class ElementaryCA:
    """
    Given a rule as an integer, number of iterations and an initial
    configuration, it returns the image produced by the evolution of the rule.

    Parameters
    ----------
    rule : int
        The rule to be applied.
    iterations : int
        The number of iterations.
    initial_config : np.ndarray
        The starting point.

    Author: MAY.
    """
    def __init__(self, rule: int, initial_config: np.ndarray):
        
        # Attributes
        self.rule = rule
        self.array = initial_config

    # Obtain the table
    def get_table(self):
        """
        Convert a decimal number to binary.
        """
        # Get the binary representation
        binary = bin(self.rule)[2:]

        # And the 8-bit representation
        binary = binary.zfill(8)

        # Now construct the table
        the_rule = {(1,1,1): int(binary[0]),
                    (1,1,0): int(binary[1]),
                    (1,0,1): int(binary[2]),
                    (1,0,0): int(binary[3]),
                    (0,1,1): int(binary[4]),
                    (0,1,0): int(binary[5]),
                    (0,0,1): int(binary[6]),
                    (0,0,0): int(binary[7])}

        return the_rule

    # For updating the array
    def update_array(self):

        # Get the lenght
        lenght = len(self.array)

        # Create a clean array
        new_array = np.zeros(lenght)

        # Loop over each element
        for m in range(lenght):

            # Get the left, center and right cells
            left = self.array[(m - 1) % lenght]
            center = self.array[m]
            right = self.array[(m + 1) % lenght]

            # Update the center cell
            new_array[m] = self.get_table()[(left, center, right)]
        
        return new_array
                       
    # Iterate
    def get_evolution(self, iterations: int):

        # Store the iterations
        self.iterations = iterations

        # Create an empty list
        image = []

        # Append the initial array
        image.append(self.array)

        # Loop for updating the state
        for i in range(self.iterations):

            # Get the next array
            next_array = self.update_array()

            # Store it
            image.append(next_array)

            # Reassing to continue
            self.array = next_array

        return np.array(image)