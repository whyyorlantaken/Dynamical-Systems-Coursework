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
                 ics: np.ndarray):
        
        # Attributes
        self.rule = rule
        self.array = ics
        self._rule_table = self._table()

    def _table(self) -> np.ndarray:
        """
        Convert a decimal number to binary.

        Returns
        -------
        dict
            A dictionary mapping the triplet of cells to their new state.
        """
        # Binary representation
        binary = bin(self.rule)[2:].zfill(8)

        rule_array = np.array([int(b) for b in binary])
        return rule_array
    
    def _update(self) -> np.ndarray:
        """
        Apply the rule to the current state.

        Returns
        -------
        np.ndarray
            The updated state of the cellular automaton.
        """
        # Get needed stuff
        length = len(self.array)

        # New array
        new_array = np.zeros(length, dtype=int)
        
        left = np.roll(self.array, 1)
        right = np.roll(self.array, -1)
        
        indices = (left * 4 + self.array * 2 + right).astype(int)
        new_array = self._rule_table[7 - indices]
        
        return new_array
                       
    def evolution(self, 
                  iterations: int) -> np.ndarray:
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
        # Create an empty list
        image = np.zeros((iterations + 1, len(self.array)), dtype=int)
        image[0] = self.array
        
        for i in range(iterations):
            self.array = self._update()
            image[i + 1] = self.array

        return image