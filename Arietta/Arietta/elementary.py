# Library
import numpy as np

# Class
class ElementaryCA:
    """
    Given a rule as an integer, number of iterations and an initial
    configuration, it returns the evolution of the rule.

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
    def __init__(self, 
                 rule: int,
                 length: int, 
                 ics: np.ndarray = None):
        
        # Attributes
        self.rule = rule

        # If no ics are provided
        if ics is None:
            self.array = np.zeros(length, dtype = int)
            self.array[length // 2] = 1

        else:
            self.array = ics

        self._rule_table = self._table()

    def _table(self) -> np.ndarray:
        """
        Convert a decimal number to binary.

        Returns
        -------
        np.ndarray
            Array representing the rule in binary form.
        """
        # Binary representation
        binary = bin(self.rule)[2:].zfill(8)

        # Get the rule
        rule = np.array([int(b) for b in binary])

        return rule
    
    def _update(self) -> np.ndarray:
        """
        Apply the rule to the current state.

        Returns
        -------
        np.ndarray
            The updated state of the cellular automaton.
        """
        # Length
        length = len(self.array)

        # New array
        new_array = np.zeros(length, dtype = int)
        
        # Periodic boundary conditions
        left = np.roll(self.array, 1)
        right = np.roll(self.array, -1)
        
        # Update array based on rule
        indices = (left * 4 + self.array * 2 + right).astype(int)
        new_array = self._rule_table[7 - indices]
        
        return new_array
                       
    def evolution(self, 
                  iterations: int,
                  save: bool = False,
                  name: str = None) -> np.ndarray:
        """
        Get the evolution of the cellular automaton.

        Parameters
        ----------
        iterations : int
            The number of iterations to perform.

        Returns
        -------
        np.ndarray
            An array containing the evolution of the cellular automaton.
        """
        # Initialize the image array
        image = np.zeros((iterations + 1, len(self.array)), dtype = int)
        image[0] = self.array
        
        # Update
        for i in range(iterations):
            self.array = self._update()
            image[i + 1] = self.array

        # Save data if required
        if save:
            np.save(name, image)

        return image