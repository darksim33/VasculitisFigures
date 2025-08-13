import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import ndimage
from radimgarray import RadImgArray, SegImgArray

PARENT = Path(r"/home/darksim/Sciebo/projects/Kidney/Ln_Vaskulitis/fig_overview_2.0")
N_SCLICES = {"healthy": 22, "vasc": 26, "rpgn": 27}
SQUARE_SIZE = (384, 320)
COLORMAPS = {"adc": "gray", "fa": "jet", "t2star": "jet", "asl": "jet"}


def load_img(path: Path):
    return np.fliplr(np.rot90(RadImgArray(path), 3))


def load_seg(path: Path):
    return np.fliplr(np.rot90(SegImgArray(path), 3))


def load_t2star(path: Path):
    """Remove spikes from t2star maps"""
    img = load_img(path)
    img[img > 80] = 80
    return img.copy()


def load_asl(path):
    """Remove artifacts from asl maps"""
    img = load_img(path)
    img[img < 0] = 0  # Remove negative values
    img[img > 1000] = 1000  # Cap values at 100
    return img.copy()


def load_images(parent):
    anats = {
        "healthy": load_img(parent / "healthy_haste.nii.gz"),
        "vasc": load_img(parent / "vasc_haste.nii.gz"),
        "rpgn": load_img(parent / "rpgn_haste.nii.gz"),
    }
    segs = {
        "healthy": load_seg(parent / "healthy_haste_seg.nii.gz"),
        "vasc": load_seg(parent / "vasc_haste_seg.nii.gz"),
        "rpgn": load_seg(parent / "rpgn_haste_seg.nii.gz"),
    }
    maps = {
        "healthy_adc": load_img(parent / "healthy_adc_rs.nii.gz"),
        "healthy_fa": load_img(parent / "healthy_fa_rs.nii.gz"),
        "healthy_t2star": load_t2star(parent / "healthy_t2star_rs.nii.gz"),
        "healthy_asl": load_asl(parent / "healthy_asl_rs.nii.gz"),
        "vasc_adc": load_img(parent / "vasc_adc_rs.nii.gz"),
        "vasc_fa": load_img(parent / "vasc_fa_rs.nii.gz"),
        "vasc_t2star": load_t2star(parent / "vasc_t2star_rs.nii.gz"),
        "vasc_asl": load_asl(parent / "vasc_asl_rs.nii.gz"),
        "rpgn_adc": load_img(parent / "rpgn_adc_rs.nii.gz"),
        "rpgn_fa": load_img(parent / "rpgn_fa_rs.nii.gz"),
        "rpgn_t2star": load_t2star(parent / "rpgn_t2star_rs.nii.gz"),
        "rpgn_asl": load_asl(parent / "rpgn_asl_rs.nii.gz"),
    }
    return anats, segs, maps


def mask_map(seg, img):
    """
    Apply segmentation mask to image array.

    Args:
        seg: Segmentation image array
        img: Image array to be masked

    Returns:
        Masked image array
    """
    if seg.shape != img.shape:
        raise ValueError("Segmentation and image arrays must have the same shape")
    img[seg == 0] = np.nan  # Set pixels to zero where segmentation is zero
    return img


def apply_masking(maps, segs):
    """
    Apply segmentation masks to anatomical and map images.

    Args:
        maps: Dictionary of map images
        segs: Dictionary of segmentation images
    Returns:
        Dictionary of masked images
    """
    masked_maps = {}
    for key, img in maps.items():
        group = key.split("_")[0]
        masked_maps[key] = mask_map(segs[group], img)
    return masked_maps


def find_center_of_mass(seg):
    """
    Find the center of mass of a segmentation mask.

    Args:
        seg: Segmentation image array

    Returns:
        Tuple of (x, y, z) coordinates of the center of mass
    """
    # if seg.ndim != 3:
    #     raise ValueError("Segmentation array must be 3D")

    # Compute the center of mass using ndimage
    center_of_mass = ndimage.center_of_mass(seg)
    return center_of_mass


def create_overlay_figure(
    background_img,
    overlay_img,
    center_point,
    square_size,
    bg_cmap="gray",
    overlay_cmap="jet",
    alpha=0.7,
    figsize=(10, 8),
    title=None,
    show_cbar=True,
):
    """
    Create a figure showing an overlay image with specified colormap on a background image.

    Args:
        background_img: 2D numpy array (background image)
        overlay_img: 2D numpy array (image to overlay with jet colormap)
        center_point: tuple (y, x) coordinates of the center point
        square_size: int, size of the square area
        bg_cmap: colormap for background image (default: 'gray')
        overlay_cmap: colormap for overlay (default: 'jet')
        alpha: transparency of overlay (default: 0.7)
        figsize: tuple, figure size (default: (10, 8))
        title: string, figure title (optional)
        show_cbar: show colorbar for figure (optional)
        ylabels: show y-axis labels for figure (optional)

    Returns:
        tuple: (figure, axis)
    """

    fig, ax = plt.subplots(figsize=figsize)

    # Show background image
    ax.imshow(background_img, cmap=bg_cmap)

    # Create square area around center point
    center_y, center_x = center_point
    if isinstance(square_size, int):
        half_size = (square_size // 2, square_size // 2)
    elif isinstance(square_size, (tuple, list)) and len(square_size) == 2:
        half_size = (square_size[0] // 2, square_size[1] // 2)

    # Calculate square boundaries
    top = int(center_y - half_size[1])
    bottom = int(center_y + half_size[1])
    left = int(center_x - half_size[0])
    right = int(center_x + half_size[0])

    # Ensure boundaries are within image limits
    height, width = background_img.shape
    top = max(0, top)
    left = max(0, left)
    bottom = min(height, bottom)
    right = min(width, right)

    # Extract the square region from overlay
    overlay_region = overlay_img[top:bottom, left:right].copy()

    # Create masked array - NaN values will be transparent
    masked_overlay = np.array(
        np.ma.masked_where(np.isnan(overlay_region), overlay_region)
    )

    # Create extent for the overlay
    extent = [left, right, bottom, top]

    # Show overlay with jet colormap - masked areas (NaN) will be transparent
    im = ax.imshow(masked_overlay, cmap=overlay_cmap, alpha=alpha, extent=extent)

    # Add colorbar for overlay
    if show_cbar:
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=45)

    if title:
        ax.set_title(title)

    ax.axis("off")

    return fig, ax


if __name__ == "__main__":
    anats, segs, maps = load_images(PARENT)
    maps_masked = apply_masking(maps, segs)
    for group in ["healthy", "vasc", "rpgn"]:
        for key in ["t2star", "asl", "fa", "adc"]:
            center_of_mass = find_center_of_mass(segs[group][:, :, N_SCLICES[group]])
            fig, ax = create_overlay_figure(
                background_img=anats[group][:, :, N_SCLICES[group]],
                overlay_img=maps_masked[group + "_" + key][:, :, N_SCLICES[group]],
                center_point=center_of_mass,
                square_size=SQUARE_SIZE,
                overlay_cmap=COLORMAPS[key],
                # title=f"{group} {key} at slice {N_SCLICES[group]}",
                show_cbar=True if group == "vasc" else False,
            )
            if group == "vasc":
                fig.show()
            fig.savefig(PARENT / "out" / f"{group}_{key}.svg", bbox_inches="tight")
            fig.savefig(PARENT / "out" / f"{group}_{key}.png", bbox_inches="tight")
    pass
