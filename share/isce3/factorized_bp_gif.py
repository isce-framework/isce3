#!/usr/bin/env python3
import argparse
import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("factors", help="HDF5 file containing FBP sub-images")
parser.add_argument("-r", "--looks-range", type=int, default=0)
parser.add_argument("-a", "--looks-azimuth", type=int, default=0)
parser.add_argument("-o", help="output animation", default="factors.gif")
parser.add_argument("--stage", type=int, default=0)
parser.add_argument("--cw", type=float, default=0.5)
parser.add_argument("--exp", type=float, default=1.0)
parser.add_argument("--duration", type=int, default=10)
args = parser.parse_args()

def multilook(z, ny=1, nx=1, f=lambda z: z):
    m, n = z.shape
    mout, nout = m // ny, n // nx
    x = f(z[:(mout * ny), :(nout * nx)])
    x.shape = mout, ny, nout, nx
    return x.mean(axis=(1, 3))

def powlooks(z, ny=1, nx=1):
    return multilook(z, ny, nx, f = lambda z: z.real**2 + z.imag**2)

h5 = h5py.File(args.factors, mode="r")
group_name = f"stage_{args.stage:02d}"
group = h5[group_name]
blocks = sorted([key for key in group if key.startswith("block_")])

nr, na = args.looks_range, args.looks_azimuth
if args.looks_range == 0 or args.looks_azimuth == 0:
    grid = group[blocks[0]]["polar_grid"]
    origin = grid["origin"][:]
    # law of cosines
    o = np.linalg.norm(origin)
    r = grid["range/first"][()]
    a = 6378137.
    look = np.arccos((o**2 + r**2 - a**2) / (2 * o * r))
    print("look angle (deg) =", np.rad2deg(look))
    dg = grid["range/spacing"][()] / np.sin(look)
    ds = grid["sin_squint/spacing"] * r
    if ds > dg:
        na = 1
        nr = round(ds / dg)
    else:
        nr = 1
        na = round(dg / ds)
    print("looks range =", nr)
    print("looks azimuth =", na)

# Figure out max image size to use for all frames.
image_shapes = []
for block_name in blocks:
    image_shapes.append(group[block_name]["image"].shape)
rows, cols = np.max(np.array(image_shapes), axis=0)
buf = np.zeros((rows, cols), dtype="c8")

power_images = []
for block_name in tqdm(blocks, "reading"):
    buf[:] = np.nan
    slc = group[block_name]["image"][:]
    # Assume we should center each image in the frame.
    i0 = (rows - slc.shape[0]) // 2
    i1 = i0 + slc.shape[0]
    j0 = (cols - slc.shape[1]) // 2
    j1 = j0 + slc.shape[1]
    buf[i0:i1, j0:j1] = slc
    power_images.append(powlooks(buf, na, nr))

mean = np.nanmean(power_images)
cw_scale = 1.0
if mean > 0:
    cw_scale = 0.7 * args.cw / mean

bitmaps = []
for zpp in tqdm(power_images, "scaling"):
    x = cw_scale * zpp
    x **= args.exp
    np.clip(x, 0, 1, x)
    arr = (255 * np.nan_to_num(x)).astype(np.uint8)
    img = Image.fromarray(arr, mode="L")
    bitmaps.append(img)

bitmaps[0].save(args.o, save_all=True, append_images=bitmaps[1:],
    optimize=False, duration=args.duration, loop=0)
