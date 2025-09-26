from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
import journal


def to_bytes(s: str | ArrayLike) -> np.ndarray:
    """
    Convert a Unicode string or array of strings into bytes.

    Parameters
    ----------
    s : str or array_like
        A Unicode string or array of Unicode strings.

    Returns
    -------
    numpy.ndarray
        The input string(s) converted to bytestrings with 'utf-8' encoding.
    """
    return np.char.encode(s, encoding="utf-8")


def get_static_layers_data_access(
        static_layers_data_access_template: str | None,
        granule_id: str | None) -> str:
    """
    Read the static layers data access template and replace the placeholder
     "{granule_id}" with the granule ID, if provided.

    Parameters
    ----------
    static_layers_data_access_template: str or None
        Template string for static layers data access. If the string contains
        the substring "{granule_id}", that substring will be replaced with
        the contents of `granule_id`.
        If set to `None` or an empty string, the function returns
        "(NOT SPECIFIED)"
        If the string is valid, i.e., it is not `None` or an empty string,
        and it does not contain the substring "{granule_id}", then
        `granule_id` is ignored.
    granule_id: str or None
        The granule ID, which will be used to replace the substring
        "{granule_id}" in `static_layers_data_access_template`.
        If the template string contains "{granule_id}", but `granule_id` is
        `None`, empty string, or "(NOT SPECIFIED)", the function raises an
        error.

    Returns
    -------
    static_layers_data_access: str
        The static layers data access string with "{granule_id}" placeholder
        replaced. Returns "(NOT SPECIFIED)" if
        `static_layers_data_access_template` is `None` or an empty
        string.
    """

    if not static_layers_data_access_template:
        return '(NOT SPECIFIED)'

    static_layers_data_access = static_layers_data_access_template

    if '{granule_id}' in static_layers_data_access_template:
        if not granule_id or granule_id == '(NOT SPECIFIED)':
            error_msg = ('The placeholder "{granule_id}" is included in'
                         ' the static layers data access template,'
                         ' but the `granule_id` was not provided'
                         " or is invalid")
            error_channel = journal.error('get_static_layers_data_access')
            error_channel.log(error_msg)
            raise ValueError(error_msg)

        static_layers_data_access = \
            static_layers_data_access_template.replace('{granule_id}',
                                                       granule_id)

    return static_layers_data_access
