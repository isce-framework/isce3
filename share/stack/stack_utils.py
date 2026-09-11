import os
from pathlib import Path
from nisar.workflows.resample_slc_v2 import (
    resample_secondary_rslc_onto_reference
)
import shutil
from nisar.products.insar import ROFFWriter
import yaml
from argparse import Namespace
from ScanList import *
from nisar.workflows.runconfig import RunConfig

MODULE_DIR = Path(__file__).resolve().parent
template = f'{MODULE_DIR}/insar.yaml'
with open(template) as f:
    template_yaml = yaml.safe_load(f)


def load_config(template=template):
    args = Namespace(
        run_config_path=template,
        log_file=False,
        restart=True,
    )

    insar_runcfg = RunConfig(args, workflow_name="insar")
    insar_runcfg.load_yaml_to_dict()
    return insar_runcfg


def set_nested(d, keys, value):
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value


def resample(
    out_dir,
    offset_dir,
    ref_path,
    secondary_path,
    out_tag="coarse",
    freq="A",
    pols=["HH"],
    block_size_az=1024,
    block_size_rg=0,
    with_gpu=True,
):
    os.makedirs(out_dir, exist_ok=True)
    resample_secondary_rslc_onto_reference(
        ref_file_path=ref_path,
        sec_file_path=secondary_path,
        out_path=out_dir,
        az_off_file=offset_dir / "azimuth.off",
        rg_off_file=offset_dir / "range.off",
        freq=freq,
        pols=pols,
        block_size_az=block_size_az,
        block_size_rg=block_size_rg,
        with_gpu=with_gpu,
    )
    shutil.move(
        out_dir /
        Path('HH/coregistered_secondary.hdr'),
        out_dir /
        Path(f'{out_tag}.hdr'))
    shutil.move(
        out_dir /
        Path('HH/coregistered_secondary.slc'),
        out_dir /
        Path(f'{out_tag}.slc'))
    shutil.rmtree(out_dir / 'HH')


def create_roff_h5(output_hdf5, cfg):
    with ROFFWriter(name=output_hdf5, mode='w',
                    runconfig_dict=cfg,
                    runconfig_path="None") as roff:
        roff.save_to_hdf5()


def relative_symlink_dir(target_dir, link_dir):
    target_dir = Path(target_dir).resolve()
    link_dir = Path(link_dir)

    link_dir.parent.mkdir(parents=True, exist_ok=True)

    if link_dir.is_symlink():
        link_dir.unlink()
    elif link_dir.exists():
        raise FileExistsError(
            f"{link_dir} already exists and is not a symlink")

    relative_target = os.path.relpath(
        target_dir,
        start=link_dir.parent.resolve()
    )

    link_dir.symlink_to(
        relative_target,
        target_is_directory=True
    )

    return link_dir


def relative_symlink_contents(target_dir, link_dir):
    """
    Symlink all top-level contents of target_dir into link_dir.

    Existing destination files/directories/symlinks with the same
    names are overwritten.

    Symlinks are relative.
    """

    target_dir = Path(target_dir).resolve()
    link_dir = Path(link_dir)

    if not target_dir.is_dir():
        raise NotADirectoryError(target_dir)

    # If link_dir itself is currently a symlink or file,
    # remove it and create a real directory.
    if link_dir.is_symlink() or link_dir.is_file():
        link_dir.unlink()

    link_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for src in target_dir.iterdir():

        dst = link_dir / src.name

        # ----------------------------------------
        # Remove existing destination
        # ----------------------------------------

        if dst.is_symlink() or dst.is_file():
            dst.unlink()

        elif dst.is_dir():
            shutil.rmtree(dst)

        # ----------------------------------------
        # Create relative symlink
        # ----------------------------------------

        relative_target = os.path.relpath(
            src,
            start=link_dir.resolve(),
        )

        dst.symlink_to(
            relative_target,
            target_is_directory=src.is_dir(),
        )

    return link_dir
