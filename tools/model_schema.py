"""Static tensor schemas for checkpoint validation.

No model evaluation lives here. All initialization and tensor arithmetic are
written as ordinary Camel programs and remain visible to autodiff/Tiles.
"""
import math

WIDTH = 128
TEXT_LENGTH = 128
HEADS = 4


def weight(*shape, std=None):
    return {"shape": shape, "init": "normal", "std": std}


def norm():
    return {"shape": (WIDTH,), "init": "ones"}


TYPES = {
    "Att": {key: weight(WIDTH, WIDTH) for key in ("q", "k", "v", "o")},
    "FF": {"up": weight(WIDTH, WIDTH * 4), "gate_w": weight(WIDTH, WIDTH * 4), "down": weight(WIDTH * 4, WIDTH)},
    "Encoder": {"att": "Att", "ff": "FF", "n1": norm(), "n2": norm()},
    "Denoiser": {"att": "Att", "cross": "Att", "ff": "FF", "n1": norm(), "n2": norm(), "n3": norm(), "time_w": weight(16, WIDTH, std=0.02)},
    "AE": {"input_w": weight(192, WIDTH), "encoder0": "Encoder", "encoder1": "Encoder",
        "pos_e": weight(64, WIDTH, std=0.02), "mu": weight(WIDTH, 4, std=0.02), "logvar": weight(WIDTH, 4, std=0.02),
        "decoder_w": weight(4, WIDTH), "pos_d": weight(64, WIDTH, std=0.02),
        "decoder0": "Encoder", "decoder1": "Encoder", "output_w": weight(WIDTH, 192, std=0.02),
        "output_b": {"shape": (192,), "init": "zeros"}},
    "LD": {"token": weight(259, WIDTH, std=0.02), "pos_t": weight(TEXT_LENGTH, WIDTH, std=0.02),
        "text0": "Encoder", "text1": "Encoder", "text_norm": norm(),
        "image_w": weight(4, WIDTH), "pos_i": weight(64, WIDTH, std=0.02), "time_w": weight(16, WIDTH, std=0.02),
        "block0": "Denoiser", "block1": "Denoiser", "block2": "Denoiser", "block3": "Denoiser",
        "output_w": weight(WIDTH, 4, std=0.02), "output_b": {"shape": (4,), "init": "zeros"}},
}


def shapes(kind, prefix=""):
    result = {}
    for name, value in TYPES[kind].items():
        path = prefix + name
        if isinstance(value, str):
            result.update(shapes(value, path + "."))
        else:
            result[path] = tuple(value["shape"])
    return result


def count(kind):
    return sum(math.prod(shape) for shape in shapes(kind).values())
