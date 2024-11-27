import matplotlib.pyplot as plt


def name_to_model(name: str) -> str:
    if "pythia" in name:
        return "Pythia"
    elif "wen" in name:
        return "Qwen2.5"
    else:
        raise ValueError(f"Unknown model name: {name}")


def name_to_attack(name: str) -> str:
    if "gcg" in name and "rt" not in name:
        return "GCG"
    elif "rt" in name and "gcg" not in name:
        return "RandomToken"
    elif "gcg" in name and "rt" in name:
        return "RandomToken"
    else:
        raise ValueError(f"Unknown attack name: {name}")


def name_to_dataset(name: str) -> str:
    if "imdb" in name:
        return "IMDB"
    elif "spam" in name:
        return "Spam"
    elif "pm" in name:
        return "PasswordMatch"
    elif "wl" in name:
        return "WordLength"
    elif "helpful" in name:
        return "Helpful"
    elif "harmless" in name:
        return "Harmless"
    elif "strongreject" in name:
        return "StrongREJECT"
    else:
        raise ValueError(f"Unknown dataset name: {name}")


def set_style(style: str) -> None:
    match style:
        # `paper` was used here for ICLR 2024
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
        case "arxiv":
            plt.rcParams.update(
                {
                    "font.family": "sans-serif",  # Use a sans-serif font
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
