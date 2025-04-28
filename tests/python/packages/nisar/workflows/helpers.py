#!/usr/bin/env python3

import pytest

from nisar.workflows.helpers import deep_update


class TestDeepUpdate:

    @pytest.mark.parametrize("flag_none_is_valid", [True, False])
    def test_case_1a(self, flag_none_is_valid):

        # =================================================================
        # Test case 1
        # -----------------------------------------------------------------
        # The user doesn't provide a parameter (i.e., a field is completely
        # omitted in the runconfig).
        # Test it with `flag_none_is_valid` equal to `True` and `False`
        # -----------------------------------------------------------------
        # Expected result:
        # Assign the default value (`value_a`) to the parameter
        # (`parameter_a`)
        # -----------------------------------------------------------------

        user_dict = {}
        default_dict = {'parameter_a': 'value_a'}
        output_dict = deep_update(
            default_dict, user_dict,
            flag_none_is_valid=flag_none_is_valid)
        assert output_dict['parameter_a'] == 'value_a'

    def test_case_2a(self):
        # =================================================================
        # Test case 2.a
        # -----------------------------------------------------------------
        # The user provides a parameter but he doesn't provide a value, AND
        # `flag_none_is_valid`` is `True`
        # -----------------------------------------------------------------
        # Expected result:
        # Assign the user value `None` (that is considered valid) to the
        # parameter (`parameter_a`)
        # -----------------------------------------------------------------

        user_dict = {'parameter_a': None}
        default_dict = {'parameter_a': 'value_a'}
        flag_none_is_valid = True
        output_dict = deep_update(
            default_dict, user_dict,
            flag_none_is_valid=flag_none_is_valid)
        assert output_dict['parameter_a'] is None

    def test_case_2b(self):
        # =================================================================
        # Test case 2.b
        # -----------------------------------------------------------------
        # The user provides a parameter but doesn't provide a value, AND
        # `flag_none_is_valid`` is `False`
        # -----------------------------------------------------------------
        # Expected result:
        # Assign the default value (`value_a`) to the parameter
        # (`parameter_a`) since the user value `None` is considered
        # invalid (`flag_none_is_valid`` is `False`)
        # -----------------------------------------------------------------

        user_dict = {'parameter_a': None}
        default_dict = {'parameter_a': 'value_a'}
        flag_none_is_valid = False
        output_dict = deep_update(
            default_dict, user_dict,
            flag_none_is_valid=flag_none_is_valid)
        assert output_dict['parameter_a'] == 'value_a'

    def test_case_3a(self):
        # =================================================================
        # Test case 3.a
        # -----------------------------------------------------------------
        # The user doesn't provide a sub-branch (e.g., `list_of_frequencies`)
        # but that sub-branch exists in the default runconfig, AND
        # `flag_none_is_valid`` is `True`.
        # -----------------------------------------------------------------
        # Expected result:
        # Assign the user value `None` (which is now considered valid) to the
        # parameter (`list_of_frequencies`) and ignore the default
        # sub-branch ({'A': 'HH', 'B': None})
        # -----------------------------------------------------------------

        user_dict = {'list_of_frequencies': None}
        default_dict = {'list_of_frequencies': {'A': 'HH', 'B': None}}

        flag_none_is_valid = True
        output_dict = deep_update(
            default_dict, user_dict,
            flag_none_is_valid=flag_none_is_valid)

        assert output_dict['list_of_frequencies'] is None

    def test_case_3b(self):
        # =================================================================
        # Test case 3.b
        # -----------------------------------------------------------------
        # The user doesn't provide a sub-branch (e.g., `list_of_frequencies`)
        # but the sub-branch exists in the default runconfig, AND
        # `flag_none_is_valid`` is `False`
        # -----------------------------------------------------------------
        # Expected result:
        # Assign the default sub-branch (`{'A': 'HH', 'B': None}`)
        # to the parameter (`list_of_frequencies`), since the user value `None`
        # is considered invalid (`flag_none_is_valid`` is `False`)
        # -----------------------------------------------------------------

        user_dict = {'list_of_frequencies': None}
        default_dict = {'list_of_frequencies': {'A': 'HH', 'B': None}}

        flag_none_is_valid = False
        output_dict = deep_update(
            default_dict, user_dict,
            flag_none_is_valid=flag_none_is_valid)
        assert output_dict['list_of_frequencies'] == \
            default_dict['list_of_frequencies']

    @pytest.mark.parametrize("flag_none_is_valid", [True, False])
    def test_case_4(self, flag_none_is_valid):
        # =================================================================
        # Test case 4
        # -----------------------------------------------------------------
        # The user provides a sub-branch (e.g., `list_of_frequencies`), but
        # the default runconfig doesn't provide it.
        # Test it with `flag_none_is_valid = True`
        # -----------------------------------------------------------------
        # Expected result:
        # Assign the sub-branch provided by the user
        # (`{'A': 'HH', 'B': None}`) to the parameter (`list_of_frequencies`)
        # -----------------------------------------------------------------

        user_dict = {'list_of_frequencies': {'A': 'HH', 'B': None}}
        default_dict = {'list_of_frequencies': None}

        output_dict = deep_update(
            default_dict, user_dict,
            flag_none_is_valid=flag_none_is_valid)

        assert output_dict['list_of_frequencies'] == \
            user_dict['list_of_frequencies']

    def test_recursion(self):
        # flag_none_is_valid should propagate to nested dicts
        default_dict = {"foo": {"x":1, "y":2}}
        user_dict = {"foo": {"x": None, "y": None}}
        output_dict = deep_update(default_dict, user_dict, False)
        assert output_dict == default_dict
