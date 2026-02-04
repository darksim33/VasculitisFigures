import overview as ov
from pathlib import Path
import matplotlib.colors as mcolors
import numpy as np

PARENT = Path(r"D:\home\Thomas\Sciebo\projects\08_Vasculitis\fig_rois")
N_SCLICE = 26
SQUARE_SIZE = (192, 192)

def create_two_value_colormap(value1, color1, value2, color2):
    """
    Create a colormap that assigns specific colors to two values and transparent to all others.
    
    Args:
        value1: First value to color (e.g., 1)
        color1: Color for first value (e.g., 'red', '#FF0000', or (1,0,0))
        value2: Second value to color (e.g., 2)
        color2: Color for second value (e.g., 'blue', '#0000FF', or (0,0,1))
    
    Returns:
        matplotlib.colors.ListedColormap
    """
    # Determine the range of values
    max_value = max(value1, value2)
    n_colors = max_value + 1
    
    # Create array of RGBA colors (all transparent by default)
    colors = np.zeros((n_colors, 4))
    
    # Set the two specific values to their colors (with alpha=1)
    colors[value1] = mcolors.to_rgba(color1)
    colors[value2] = mcolors.to_rgba(color2)
    
    return mcolors.ListedColormap(colors)

def main():
    img = ov.load_img(PARENT / "rpgn_haste.nii.gz")
    segm = ov.load_seg(PARENT / "rpgn_haste_seg_right.nii.gz")
    rois = ov.load_seg(PARENT / "rpgn_rois.nii.gz")
    center_of_mass = ov.find_center_of_mass(segm[:,:, N_SCLICE])
    
    # Create a colormap for ROIs - assign distinct colors to values 1 and 2
    roi_cmap = create_two_value_colormap(1, 'red', 2, 'lightgreen')
    
    fig, ax = ov.create_overlay_figure(
                background_img=img[:, :, N_SCLICE],
                overlay_img=rois[:, :, N_SCLICE],
                center_point=center_of_mass,
                square_size=SQUARE_SIZE,
                overlay_cmap=roi_cmap,
                # title=f"{group} {key} at slice {N_SCLICES[group]}",
                show_cbar=False,
            )
    fig.show()
    fig.savefig(PARENT / "out" / f"rois.svg", bbox_inches="tight")
    fig.savefig(PARENT / "out" / f"rois.png", bbox_inches="tight")

if __name__ == "__main__":
    main()