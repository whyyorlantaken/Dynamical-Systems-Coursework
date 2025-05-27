"""
Code to generate bifurcation diagrams using different maps.
"""
# Imports
import h5py
import numpy as np
from numba import jit
import matplotlib.pyplot as plt

################################################
#                     MAPS                     #
################################################

# Sine map
@jit(nopython=True)
def sine_map(x, r):
    return np.sin(r * np.arcsin(np.sqrt(x))) ** 2

# Hyperbolic tan map
@jit(nopython=True)
def tanh_map(x, r):
    return np.abs(np.tanh(1.3 * (x - r)))

# Power map
@jit(nopython=True)
def power_map(x, r):
    return (1-r**(x * (1-x)))/(1-r**(1/4))

################################################
#                   FUNCTION                   #
################################################

def bifurcation(map: callable,
                r_min: float = 1.0,
                r_max: float = 4.0,
                max_x: float = 1.0,
                min_x: float = 0.0,
                numtoplot: int = 1000,
                transient: int = 1000,
                initial_x: float = 0.5,
                width: int = 800,
                height: int = 600) -> None:
    """
    Function to generate a bifurcation diagram using a given map function.

    Parameters
    ----------
    map : callable
        The map function to be used.
    initial_x : float
        The initial condition.
    width : int
        The width of the histogram.
    height : int
        The height of the histogram.
    """
    # Create a 2D histogram
    hist = np.zeros((height, width), dtype = np.int32)
    
    # Calculate r values for each column
    r_values = np.linspace(r_min, r_max, width)
    
    print("---------------------------------------------------")
    print("Bifurcation diagram data...")
    print("---------------------------------------------------")

    # For each parameter value
    for i, r in enumerate(r_values):

        # Info
        if i % 500 == 0:
            print(f"Processing r = {r:.4f}, ({i}/{width})")
        
        # Set initial condition
        x = initial_x
        
        # Run transient iterations
        for _ in range(transient):
            x = map(x, r)
        
        # After transient
        for _ in range(numtoplot):
            x = map(x, r)
            
            # Map x to a row in the histogram
            if x >= min_x and x <= max_x:
                row = int((1 - x) * (height - 1))
                hist[row, i] += 1

    # End
    print("---------------------------------------------------")
    print("Done!")
    print("---------------------------------------------------")
    
    # Filename
    filename = f"results/{map.__name__}_r{r_min:.1f}_{r_max:.1f}_w{int(width)}_h{int(height)}.h5"

    # Save both
    with h5py.File(filename, "w") as f:
        f.create_dataset("histogram", data = hist)
        f.create_dataset("r_values", data = r_values)
    print(f"Saved to {filename}")

    return hist, r_values

if __name__ == "__main__":

    # Usage
    # bifurcation(sine_map,
    #             r_min = 1.0,
    #             r_max = 4.0,
    #             min_x = 0.0,
    #             max_x = 1.0,
    #             numtoplot = 10000,
    #             transient = 2000,
    #             initial_x = 0.5,
    #             width = 10000,
    #             height = 10000//3)
    
    # hist, r_values = bifurcation(tanh_map,
    #                             r_min = 0.0,
    #                             r_max = 1.0,
    #                             min_x = 0.0,
    #                             max_x = 1.0,
    #                             numtoplot = 80000,
    #                             transient = 2000,
    #                             initial_x = 0.5,
    #                             width = 3072,
    #                             height = 3072)
    
    # # Plot
    # plt.figure(figsize = (7, 7))
    # plt.imshow(np.log10(hist +1), 
    #            extent = [0, 1, 0, 1],
    #            aspect = "auto",
    #            cmap = "Reds",
    #            interpolation = "nearest")
    
    # # Labels
    # plt.title("Bifurcation diagram of "+ r"$x_{n+1} = |\tanh(1.3(x_n - c))|$", fontsize = 14)
    # plt.xlabel("c", fontsize = 13)
    # plt.ylabel(r"$x_n$", fontsize = 13)
    # plt.tick_params(axis='both', labelsize=9)
    # plt.tight_layout()
    # plt.savefig("results/bifurcation_diagram5.pdf", dpi = 300)
    # plt.close()

    # hist, r_values = bifurcation(power_map,
    #                             r_min = 0.0,
    #                             r_max = 1.0,
    #                             min_x = 0.0,
    #                             max_x = 1.0,
    #                             numtoplot = 80000,
    #                             transient = 2000,
    #                             initial_x = 0.5,
    #                             width = 3072,
    #                             height = 3072)
    
    # # Plot
    # plt.figure(figsize = (7, 7))
    # plt.imshow(np.log10(hist +1), 
    #            extent = [0, 1, 0, 1],
    #            aspect = "auto",
    #            cmap = "Purples",
    #            interpolation = "nearest")
    
    # # Labels
    # plt.title("Bifurcation diagram of "+ r"$x_{n+1} = \frac{1 - b^{x_n(1-x_n)}}{1 - b^{1/4}}$", fontsize = 14)
    # plt.xlabel("b", fontsize = 13)
    # plt.ylabel(r"$x_n$", fontsize = 13)
    # plt.tick_params(axis='both', labelsize=9)
    # plt.tight_layout()
    # plt.savefig("results/bifurcation_diagram6.pdf", dpi = 300)
    # plt.close()

    x_array = np.linspace(0, 1, 500)

    # Feed the power map
    y_array = power_map(x_array, 0.5)

    # Plot
    plt.figure(figsize = (7, 7))
    plt.plot(x_array, y_array, color = "purple", lw = 0.8)
    plt.title("Power map", fontsize = 14)
    plt.xlabel(r"$x_n$", fontsize = 13)
    plt.ylabel(r"$f(x)$", fontsize = 13)
    plt.tick_params(axis='both', labelsize=9)
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.grid()
    plt.tight_layout()
    plt.show()
    plt.close()
