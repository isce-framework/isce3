#!/usr/bin/env python3
import argparse, os
from imagesets import imagesets, projsrcdir

def run(*, steps, imgset, **kwargs):
    imgsetclass = imagesets[imgset](**kwargs)

    mainsteps = [
        "setup",
        "configure",
        "build",
        "test",
        "makepkg",
        "makedistrib",
        "fetchdata",
        "rslctest",
        "doppler_test",
        "el_edge_test",
        "el_null_test",
        "gslctest",
        "gcovtest",
        "insartest",
        "end2endtest",
    ]

    nisarsteps = mainsteps + [
        "makedistrib_nisar",
        "noisesttest",
        "ptatest",
        "soilmtest",
        "rslcqa",
        "gslcqa",
        "gcovqa",
        "insarqa",
        "end2endqa",
    ]

    # you can say "all" or "main" for a sequence similar to our CI pipeline
    if steps == ["all"] or steps == ["main"]:
        steps = mainsteps
    elif steps == ["nisar"]:
        steps = nisarsteps

    # extra helper steps that don't fall under the main  pipeline
    everything = nisarsteps + [
        "dropin",
        "docsbuild",
        "prdocs",
        "fetchmindata",
        "mintests",
        "minqa",
        "tartests",
        "push",
    ]

    # map step names to the imgset's methods
    mapsteps = {step: getattr(imgsetclass, step) for step in everything}

    # validate the provided steps
    for s in steps:
        if s not in mapsteps:
            raise ValueError(f"Unrecognized step: {s}")

    # run all the steps in the order provided
    for s in steps:
        mapsteps[s]()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--imgset", default="oracle8conda")
    parser.add_argument("-B", "--projblddir", default=f"{projsrcdir}/build-docker")
    parser.add_argument("-p", "--printlog", action='store_true')
    parser.add_argument("-t", "--imgtag", default=None)
    parser.add_argument("steps", nargs="+")
    run(**vars(parser.parse_args()))
