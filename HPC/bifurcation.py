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

@jit(nopython=True)
def sine_map(x, r):
    return np.sin(r * np.arcsin(np.sqrt(x))) ** 2

@jit(nopython=True)
def tanh_map(x, r):
    return np.abs(np.tanh(1.3 * (x - r)))

@jit(nopython=True)
def power_map(x, r):
    return (1-r**(x * (1-x)))/(1-r**(1/4))

@jit(nopython=True)
def circle_map(x, r):
    return (x + r - (1/(2 * np.pi)) * np.sin(2 * np.pi * x)) % 1.0

@jit(nopython=True)
def map_2d(state, b, a=1.5):
    x, y = state
    x_new = 1 - a * np.abs(x) + y
    y_new = b * x
    return x_new, y_new

def two_map(x, r):
    global y_state
    if not hasattr(two_map, 'y_state'):
        two_map.y_state = 0.0
    
    # Apply the 2D map
    x_new, y_new = map_2d((x, two_map.y_state), r, a=1.5)
    
    # Update
    two_map.y_state = y_new
    
    return x_new


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
            row = int((max_x - x) / (max_x - min_x) * (height - 1))
            row = max(0, min(row, height - 1))
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
    #hist, values = bifurcation(sine_map,
    #            r_min = 1.0,
    #            r_max = 4.0,
    #            min_x = 0.0,
    #            max_x = 1.0,
    #            numtoplot = 250000,
    #            transient = 2000,
    #            initial_x = 0.5,
    #            width = 2048,
    #            height = int(2048*3))
    
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

    hist, r_values = bifurcation(two_map,
                                 r_min = -0.7,
                                 r_max = +0.7,
                                 min_x = -1.0,
                                 max_x = +1.3,
                                 numtoplot = 30000, #80000
                                 transient = 2000,
                                 initial_x = 0.5,
                                 width = int(512*2),
                                 height = 512)
    
    # Plot
    plt.figure(figsize = (7, 5))
    plt.imshow(np.log(hist+1), 
               extent = [-0.7, 0.7, -1, 1.3],
               aspect = "auto",
               cmap = "Reds",
               interpolation = "nearest")
    
    # Labels
    # plt.title("Bifurcation diagram of "+ r"$x_{n+1} = \sin^2(r\,\arcsin(\sqrt{x_n}))$", fontsize=14)
	# plt.title("Bifurcation diagram of " + r"$x_{n+1}=\frac{1-b^{x_n(1-x_n)}}{1-b^{1/4}}$")
    # plt.title("Bifurcation diagram of " + r"$x_{n+1} = x_n + \omega - \frac{1}{2\pi}\sin(2\pi x_n) \text{ mod } 1$", fontsize=14)
    plt.title("Bifurcation diagram of " + r"$x_{n+1} = 1 - 1.5|x_n| + y_n, \, y_{n+1} = b x_n$", fontsize=14)
    plt.xlabel(r"$b$", fontsize = 13)
    plt.ylabel(r"$x_n$", fontsize = 13)
    plt.tick_params(axis='both', labelsize=9)
    plt.tight_layout()
    #plt.ylim(-0.1,1)
    plt.savefig("results/TWO-MAP-REDS.pdf", dpi = 300)
    plt.close()
