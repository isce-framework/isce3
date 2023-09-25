#!/usr/bin/env python3

from nisar.workflows.helpers import deep_update


class TestDeepUpdate:
    def test_case_1(self):

        # =================================================================
        # Test case 1
        # -----------------------------------------------------------------
        # The user doesn't provide a parameter (i.e., a field is completely
        # ommited in the runconfig).
        # -----------------------------------------------------------------
        # Expected result:
        # Assign the default value (`value_a`) to the parameter
        # (`parameter_a`)
        # -----------------------------------------------------------------

        user_dict = {}
        default_dict = {'parameter_a': 'value_a'}
        output_dict = deep_update(default_dict, user_dict)
        assert output_dict['parameter_a'] == 'value_a'

    def test_case_2(self):
        # =================================================================
        # Test case 2
        # -----------------------------------------------------------------
        # The user provides a parameter but he doesn't provide a value
        # -----------------------------------------------------------------
        # Expected result:
        # Assign the user value `None` (that is considered valid) to the
        # parameter (`parameter_a`)
        # -----------------------------------------------------------------

        user_dict = {'parameter_a': None}
        default_dict = {'parameter_a': 'value_a'}
        output_dict = deep_update(default_dict, user_dict)
        assert output_dict['parameter_a'] is None

    def test_case_3(self):
        # =================================================================
        # Test case 3
        # -----------------------------------------------------------------
        # The user doesn't provide a sub-branch (e.g., `list_of_frequencies`)
        # but that sub-branch exists in the default runconfig
        # -----------------------------------------------------------------
        # Expected result:
        # Assign the user value `None` (thta is considered valid) to the
        # parameter (`list_of_frequencies`) and ignore the default
        # sub-branch ({'A': 'HH', 'B': None})
        # -----------------------------------------------------------------

        user_dict = {'list_of_frequencies': None}
        default_dict = {'list_of_frequencies': {'A': 'HH', 'B': None}}

        output_dict = deep_update(default_dict, user_dict)

        assert output_dict['list_of_frequencies'] is None

    def test_case_4(self):
        # =================================================================
        # Test case 4
        # -----------------------------------------------------------------
        # The user provides a sub-branch (e.g., `list_of_frequencies`), but
        # the default runconfig doesn't provide it.
        # -----------------------------------------------------------------
        # Expected result:
        # Assign the sub-branch provided by the user
        # (`{'A': 'HH', 'B': None}`) to the parameter (`list_of_frequencies`)
        # -----------------------------------------------------------------

        user_dict = {'list_of_frequencies': {'A': 'HH', 'B': None}}
        default_dict = {'list_of_frequencies': None}

        output_dict = deep_update(default_dict, user_dict)
        assert 'A' in output_dict['list_of_frequencies'].keys()
        assert 'B' in output_dict['list_of_frequencies'].keys()

        assert output_dict['list_of_frequencies']['A'] == 'HH'
        assert output_dict['list_of_frequencies']['B'] is None
