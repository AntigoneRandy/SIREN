import math
import torchvision.transforms.functional as TF
import torch
from torch import Tensor
from PIL import Image

_ycbcr_conversions = {
    'rec_601': (0.299, 0.587, 0.114),
    'rec_709': (0.2126, 0.7152, 0.0722),
    'rec_2020': (0.2627, 0.678, 0.0593),
    'smpte_240m': (0.212, 0.701, 0.087),
}


def rgb_to_ycbcr(input: Tensor, standard: str = 'rec_2020'):
    kr, kg, kb = _ycbcr_conversions[standard]
    conversion_matrix = torch.tensor([[kr, kg, kb],
                                      [-0.5 * kr / (1 - kb), -
                                       0.5 * kg / (1 - kb), 0.5],
                                      [0.5, -0.5 * kg / (1 - kr), -0.5 * kb / (1 - kr)]], device=input.device)
    return torch.einsum('mc,nchw->nmhw', conversion_matrix, input)


def ycbcr_to_rgb(input: Tensor, standard: str = 'rec_2020'):
    kr, kg, kb = _ycbcr_conversions[standard]
    conversion_matrix = torch.tensor([[1, 0, 2 - 2 * kr],
                                      [1, -kb / kg *
                                          (2 - 2 * kb), -kr / kg * (2 - 2 * kr)],
                                      [1, 2 - 2 * kb, 0]], device=input.device)
    return torch.einsum('mc,nchw->nmhw', conversion_matrix, input)


_xyz_conversions = {
    'CIE_RGB': ((0.4887180, 0.3106803, 0.2006017),
                (0.1762044, 0.8129847, 0.0108109),
                (0.0000000, 0.0102048, 0.9897952)),
    'sRGB': ((0.4124564, 0.3575761, 0.1804375),
             (0.2126729, 0.7151522, 0.0721750),
             (0.0193339, 0.1191920, 0.9503041))
}


def rgb_to_xyz(input: Tensor, rgb_space: str = 'sRGB'):
    conversion_matrix = torch.tensor(
        _xyz_conversions[rgb_space], device=input.device)
    # Inverse sRGB companding
    v = torch.where(input <= 0.04045, input / 12.92,
                    ((input + 0.055) / 1.055) ** 2.4)
    return torch.einsum('mc,nchw->nmhw', conversion_matrix, v)


_delta = 6 / 29


def cielab_func(input: Tensor) -> Tensor:
    # torch.where produces NaNs in backward if one of the choice produces NaNs or infs in backward (here .pow(1/3))
    return torch.where(input > _delta ** 3, input.clamp(min=_delta ** 3).pow(1 / 3), input / (3 * _delta ** 2) + 4 / 29)


def cielab_inverse_func(input: Tensor) -> Tensor:
    return torch.where(input > _delta, input.pow(3), 3 * _delta ** 2 * (input - 4 / 29))


_cielab_conversions = {
    'illuminant_d50': (96.4212, 100, 82.5188),
    'illuminant_d65': (95.0489, 100, 108.884),
}


def rgb_to_cielab(input: Tensor, standard: str = 'illuminant_d65') -> Tensor:
    # Convert to XYZ
    XYZ_input = rgb_to_xyz(input=input)

    Xn, Yn, Zn = _cielab_conversions[standard]
    L_star = 116 * cielab_func(XYZ_input.narrow(1, 1, 1) / Yn) - 16
    a_star = 500 * (cielab_func(XYZ_input.narrow(1, 0, 1) / Xn) -
                    cielab_func(XYZ_input.narrow(1, 1, 1) / Yn))
    b_star = 200 * (cielab_func(XYZ_input.narrow(1, 1, 1) / Yn) -
                    cielab_func(XYZ_input.narrow(1, 2, 1) / Zn))
    return torch.cat((L_star, a_star, b_star), 1)


def ciede2000_color_difference(Lab_1: Tensor, Lab_2: Tensor, k_L: float = 1, k_C: float = 1, k_H: float = 1,
                               squared: bool = False, ε: float = 0.0001) -> Tensor:
    """
    Inputs should be L*, a*, b*. Primes from formulas in
    http://www2.ece.rochester.edu/~gsharma/ciede2000/ciede2000noteCRNA.pdf are omitted for conciseness.
    This version is based on the matlab implementation from Gaurav Sharma
    http://www2.ece.rochester.edu/~gsharma/ciede2000/dataNprograms/deltaE2000.m modified to have non NaN gradients.

    Parameters
    ----------
    Lab_1 : Tensor
        First image in L*a*b* space. First image is intended to be the reference image.
    Lab_2 : Tensor
        Second image in L*a*b* space. Second image is intended to be the modified one.
    k_L : float
        Weighting factor for S_L.
    k_C : float
        Weighting factor for S_C.
    k_H : float
        Weighting factor for S_H.
    squared : bool
        Return the squared ΔE_00.
    ε : float
        Small value for numerical stability when computing gradients. Default to 0 for most accurate evaluation.

    Returns
    -------
    ΔE_00 : Tensor
        The CIEDE2000 color difference for each pixel.

    """
    assert Lab_1.size(1) == 3 and Lab_2.size(1) == 3
    assert Lab_1.dtype == Lab_2.dtype
    dtype = Lab_1.dtype
    π = torch.tensor(math.pi, dtype=dtype, device=Lab_1.device)
    π_compare = π if dtype == torch.float64 else torch.tensor(
        math.pi, dtype=torch.float64, device=Lab_1.device)

    L_star_1, a_star_1, b_star_1 = Lab_1.unbind(dim=1)
    L_star_2, a_star_2, b_star_2 = Lab_2.unbind(dim=1)

    C_star_1 = torch.norm(torch.stack((a_star_1, b_star_1), dim=1), p=2, dim=1)
    C_star_2 = torch.norm(torch.stack((a_star_2, b_star_2), dim=1), p=2, dim=1)
    C_star_bar = (C_star_1 + C_star_2) / 2
    C7 = C_star_bar ** 7
    G = 0.5 * (1 - (C7 / (C7 + 25 ** 7)).clamp(min=ε).sqrt())

    scale = 1 + G
    a_1 = scale * a_star_1
    a_2 = scale * a_star_2
    C_1 = torch.norm(torch.stack((a_1, b_star_1), dim=1), p=2, dim=1)
    C_2 = torch.norm(torch.stack((a_2, b_star_2), dim=1), p=2, dim=1)
    C_1_C_2_zero = (C_1 == 0) | (C_2 == 0)
    h_1 = torch.atan2(b_star_1, a_1 + ε * (a_1 == 0))
    h_2 = torch.atan2(b_star_2, a_2 + ε * (a_2 == 0))

    # required to match the test data
    h_abs_diff_compare = (torch.atan2(b_star_1.to(dtype=torch.float64),
                                      a_1.to(dtype=torch.float64)).remainder(2 * π_compare) -
                          torch.atan2(b_star_2.to(dtype=torch.float64),
                                      a_2.to(dtype=torch.float64)).remainder(2 * π_compare)).abs() <= π_compare

    h_1 = h_1.remainder(2 * π)
    h_2 = h_2.remainder(2 * π)
    h_diff = h_2 - h_1
    h_sum = h_1 + h_2

    ΔL = L_star_2 - L_star_1
    ΔC = C_2 - C_1
    Δh = torch.where(C_1_C_2_zero, torch.zeros_like(h_1),
                     torch.where(h_abs_diff_compare, h_diff,
                                 torch.where(h_diff > π, h_diff - 2 * π, h_diff + 2 * π)))

    ΔH = 2 * (C_1 * C_2).clamp(min=ε).sqrt() * torch.sin(Δh / 2)
    ΔH_squared = 4 * C_1 * C_2 * torch.sin(Δh / 2) ** 2

    L_bar = (L_star_1 + L_star_2) / 2
    C_bar = (C_1 + C_2) / 2

    h_bar = torch.where(C_1_C_2_zero, h_sum,
                        torch.where(h_abs_diff_compare, h_sum / 2,
                                    torch.where(h_sum < 2 * π, h_sum / 2 + π, h_sum / 2 - π)))

    T = 1 - 0.17 * (h_bar - π / 6).cos() + 0.24 * (2 * h_bar).cos() + \
        0.32 * (3 * h_bar + π / 30).cos() - 0.20 * \
        (4 * h_bar - 63 * π / 180).cos()

    Δθ = π / 6 * (torch.exp(-((180 / π * h_bar - 275) / 25) ** 2))
    C7 = C_bar ** 7
    R_C = 2 * (C7 / (C7 + 25 ** 7)).clamp(min=ε).sqrt()
    S_L = 1 + 0.015 * (L_bar - 50) ** 2 / torch.sqrt(20 + (L_bar - 50) ** 2)
    S_C = 1 + 0.045 * C_bar
    S_H = 1 + 0.015 * C_bar * T
    R_T = -torch.sin(2 * Δθ) * R_C

    ΔE_00 = (ΔL / (k_L * S_L)) ** 2 + (ΔC / (k_C * S_C)) ** 2 + ΔH_squared / (k_H * S_H) ** 2 + \
        R_T * (ΔC / (k_C * S_C)) * (ΔH / (k_H * S_H))
    if squared:
        return ΔE_00
    return ΔE_00.clamp(min=ε).sqrt()


def rgb_ciede2000_color_difference(input: Tensor, target: Tensor, **kwargs) -> Tensor:
    """Computes the CIEDE2000 Color-Difference from RGB inputs."""
    return ciede2000_color_difference(*map(rgb_to_cielab, (input, target)), **kwargs)


def ciede2000_loss(x1: Tensor, x2: Tensor, squared: bool = False, **kwargs) -> Tensor:
    """
    Computes the L2-norm over all pixels of the CIEDE2000 Color-Difference for two RGB inputs.

    Parameters
    ----------
    x1 : Tensor:
        First input.
    x2 : Tensor:
        Second input (of size matching x1).
    squared : bool
        Returns the squared L2-norm.

    Returns
    -------
    ΔE_00_l2 : Tensor
        The L2-norm over all pixels of the CIEDE2000 Color-Difference.

    """
    ΔE_00 = rgb_ciede2000_color_difference(
        x1, x2, squared=True, **kwargs).flatten(1)
    ε = kwargs.get('ε', 0.0001)
    if squared:
        return ΔE_00.mean(1)
    return ΔE_00.mean(1).clamp(min=ε)


def load_image(file_path):
    img = Image.open(file_path)
    img_tensor = TF.to_tensor(img)
    img_tensor = img_tensor.unsqueeze(0)
    return img_tensor


# torch.set_printoptions(precision=15, sci_mode=True)

# # x = int(input())
# img1_tensor = load_image(f'/Users/randy/Downloads/pokemon/15_o.png')
# img2_tensor = load_image(f'/Users/randy/Downloads/pokemon/15_h.png')
# print(ciede2000_loss(img1_tensor, img2_tensor) * 1e-3,
#       torch.nn.MSELoss()(img1_tensor, img2_tensor))
