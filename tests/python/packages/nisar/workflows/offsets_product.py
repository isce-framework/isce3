#!/usr/bin/env python3

import journal
import pytest

from nisar.workflows.offsets_product import get_start_pixels


def make_cfg(windows, start=None):
    '''Minimal offsets_product cfg with square layer windows'''
    cfg = {'margin': 50, 'gross_offset_range': 0, 'gross_offset_azimuth': 0,
           'start_pixel_range': start, 'start_pixel_azimuth': start}
    for i, win in enumerate(windows):
        cfg[f'layer{i + 1}'] = {'window_range': win, 'window_azimuth': win,
                                'half_search_range': 20,
                                'half_search_azimuth': 20}
    return cfg


@pytest.mark.parametrize("windows", [(32, 64, 128), (64, 96, 196),
                                     (33, 64, 127)])
@pytest.mark.parametrize("start", [None, 314])
def test_layer_windows_centered_on_grid(windows, start):
    cfg = make_cfg(windows, start)

    # Common start pixel (margin + half search if not set), and grid center
    # as in helpers.get_offset_radar_grid
    start0 = 50 + 20 if start is None else start
    center = start0 + min(windows) // 2

    for win in windows:
        az_start, rg_start = get_start_pixels(cfg, win, win)
        assert rg_start + win // 2 == center
        assert az_start + win // 2 == center

    # Smallest window keeps the common start pixel
    assert get_start_pixels(cfg, min(windows), min(windows)) == (start0, start0)


@pytest.mark.parametrize("window_azimuth, window_range",
                         [(None, 64), (64, None), (None, None)])
def test_missing_window_size(window_azimuth, window_range):
    with pytest.raises(journal.ApplicationError):
        get_start_pixels(make_cfg((32, 64, 128)), window_azimuth,
                         window_range)
