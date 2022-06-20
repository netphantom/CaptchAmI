import base64
import io as py_io

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage import io, util, measure
from skimage.color import rgb2gray
from skimage.filters import threshold_minimum
from skimage.measure import label
from numpy import pad
from skimage.transform import resize


def elaborate_stars(img: str) -> int:
    """
    Elaborate the stars image, returning the number of stars found

    Args:
        img: The image to elaborate

    Returns:
        The number representing how many stars are in the image
    """
    img = preprocess(img)
    distance = ndi.distance_transform_edt(img)
    stars = (distance > 0.8 * distance.max(initial=0))

    return np.max(label(stars))


def elaborate_numbers(img: str) -> list:
    """
    This function transforms the image into three parts, ready to be classified

    Args:
        img: The input image

    Returns:
        A list of ordered elements, corresponding to the first element, the operand and the second element
    """
    img = preprocess(img)
    values, items = ndi.label(img)
    regions = measure.regionprops(values)
    if items < 3:
        values, items = find_third(regions, img)
    elif items > 3:
        values, items = delete_one(regions, img)
    regions = measure.regionprops(values)

    col_order = []
    img_crop = []
    for i in range(items):
        rmin, cmin, rmax, cmax = regions[i].bbox
        boxed_img = img[rmin:rmax, cmin:cmax]
        boxed_img = pad(boxed_img, 5)
        boxed_img = resize(boxed_img, (32, 32))
        img_crop.append(boxed_img)
        col_order.append(cmin)
    img_order = sorted(zip(col_order, img_crop))

    return img_order


def preprocess(img: str) -> np.ndarray:
    """
    Perform the pre-process on the image:
        1) Conversion to gray
        2) Color inversion
        3) Thresholding

    Args:
        img: The image to elaborate

    Returns:
        The preprocessed image
    """
    img = io.imread(img)
    img = rgb2gray(img)
    img = util.invert(img)
    img = img < threshold_minimum(img)
    return img


def find_third(regions: list, img: np.ndarray) -> ndi.label:
    """
    If only two regions have been found, this function split the biggest region into two parts.

    Args:
        regions: The two regions found
        img: the original image

    Returns:
        The new labels of the three regions
    """
    # Let's take the biggest region
    if regions[0].area > regions[1].area:
        big_region = regions[0]
    else:
        big_region = regions[1]
    rmin, cmin, rmax, cmax = big_region.bbox

    # Once we find the biggest image, we just cut 6 px before its end.
    col_cut = cmax - 6
    img[:, col_cut] = False

    return ndi.label(img)


def delete_one(regions: list, img: np.ndarray) -> ndi.label:
    """
    If more than 3 regions have been found, this function remove the smallest

    Args:
        regions: The regions found
        img: the original image

    Returns:
        The new labels of the region
    """
    # Let's find the smallest area
    smallest_area = regions[0]
    for r in regions:
        if r.area < smallest_area.area:
            smallest_area = r
    rmin, cmin, rmax, cmax = smallest_area.bbox

    # Let's remove it
    img[rmin:rmax, cmin:cmax] = False

    return ndi.label(img)


def squarify(img):
    """
    This function make a square out of the image and add some padding to reach the size of 32x32

    Args:
        img: The image to squarify and pad

    Returns:
        The squared and padded image
    """

    # Initially let's make a square
    (a, b) = img.shape
    padding = ((int((32 - a) / 2), int((32 - a) / 2)), (int((32 - b) / 2), int((32 - b) / 2)))
    img = np.pad(img, padding, mode='constant')

    # Let's be sure to have a 32x32 image
    (a, b) = img.shape
    if a < 32:
        padding = ((0, 1), (0, 0))
        img = np.pad(img, padding, mode='constant')
    elif b < 32:
        padding = ((0, 0), (0, 1))
        img = np.pad(img, padding, mode='constant')

    return img


def base64_to_img(base_64: str, path: str) -> None:
    """
    Convert and save a string of a base64 image to a file

    Args:
        base_64: the string containing the converted image
        path: the path to save the converted image

    Returns:
        None
    """
    msg = base64.b64decode(base_64)
    buf = py_io.BytesIO(msg)
    img = Image.open(buf)
    img = img.convert("RGB")
    img.save(fp=path, format="PNG")
