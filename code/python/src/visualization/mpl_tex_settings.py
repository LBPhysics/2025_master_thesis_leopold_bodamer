import matplotlib.pyplot as plt
import matplotlib as mpl

# Matplotlib settings according to LaTeX caption formatting
plt.rcParams.update(
    {
        "text.usetex": True,  # Enable LaTeX for text rendering
        "font.family": "serif",  # Use a serif font family
        "font.serif": "Palatino",  # or [], Set Palatino or standard latex font
        "text.latex.preamble": r"\usepackage{amsmath}",
        "font.size": 20,  # Font size for general text
        "axes.titlesize": 20,  # Font size for axis titles
        "axes.labelsize": 20,  # Font size for axis labels
        "xtick.labelsize": 20,  # Font size for x-axis tick labels
        "ytick.labelsize": 20,  # Font size for y-axis tick labels
        "legend.fontsize": 20,  # Font size for legends
        "figure.figsize": [8, 6],  # Size of the plot (width x height)
        "figure.autolayout": True,  # Automatic layout adjustment
        "savefig.format": "svg",  # Default format for saving figures
        "figure.facecolor": "none",  # Make the figure face color transparent
        "axes.facecolor": "none",  # Make the axes face color transparent
        "savefig.transparent": True,  # Save figures with transparent background
    }
)

# mpl.use("Agg")  # Use a non-interactive backend / SAVE figures to svg files
mpl.use(
    "module://matplotlib_inline.backend_inline"
)  # Use inline backend for Jupyter notebooks
# mpl.use('TkAgg')  # open each plot in interactive window
