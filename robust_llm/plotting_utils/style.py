import matplotlib.pyplot as plt


def set_plot_style(style: str) -> None:
    match style:
        case "paper":
            plt.rcParams.update(
                {
                    "font.family": "serif",  # Use a serif font
                    "font.size": 9,  # General font size
                    "axes.titlesize": 10,  # Title font size
                    "axes.labelsize": 9,  # X and Y label font size
                    "xtick.labelsize": 8,  # X tick label font size
                    "ytick.labelsize": 8,  # Y tick label font size
                    "legend.fontsize": 8,  # Legend font size
                }
            )
        case "poster":
            COLOR = "1e4041"
            plt.rcParams.update(
                {
                    # "text.usetex": True,
                    "font.family": "sans-serif",  # Use a sans-serif font
                    "font.sans-serif": ["Arial"],
                    "font.size": 12,  # Poster font size
                    "axes.titlesize": 16,  # Title font size
                    "axes.labelsize": 15,  # X and Y label font size
                    "xtick.labelsize": 10,  # X tick label font size
                    "ytick.labelsize": 10,  # Y tick label font size
                    "legend.fontsize": 10,  # Legend font size
                    "text.color": COLOR,
                    "axes.labelcolor": COLOR,
                    "xtick.color": "black",
                    "ytick.color": "black",
                }
            )
        case "blog":
            plt.rcParams.update(
                {
                    "font.family": "sans-serif",  # Use a sans-serif font
                    "font.sans-serif": ["Arial"],
                    "font.size": 9,  # General font size
                    "axes.titlesize": 10,  # Title font size
                    "axes.labelsize": 9,  # X and Y label font size
                    "xtick.labelsize": 8,  # X tick label font size
                    "ytick.labelsize": 8,  # Y tick label font size
                    "legend.fontsize": 8,  # Legend font size
                }
            )
        case _:
            raise ValueError(f"Unknown style: {style}")
