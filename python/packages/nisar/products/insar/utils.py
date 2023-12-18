import re
from datetime import datetime
from typing import Optional


def number_to_ordinal(number):
    """
    Convert an unsigned integer to its ordinal representation.

    Parameters
    ----------
    number : int
        The non-negative integer to be converted to its ordinal form.

    Returns
    -------
    str
        The ordinal representation of the input number.

    Notes
    -----
    The function appends the appropriate suffix ('st', 'nd', 'rd', or 'th')
    to the input number based on common English ordinal representations.
    Exceptions are made for numbers ending in 11, 12, and 13, which use 'th'.

    Examples
    --------
    >>> number_to_ordinal(1)
    '1st'

    >>> number_to_ordinal(22)
    '22nd'

    >>> number_to_ordinal(33)
    '33rd'

    >>> number_to_ordinal(104)
    '104th'
    """
    if 10 <= number % 100 <= 20:
        suffix = 'th'
    else:
        suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(number % 10, 'th')
    return f"{number}{suffix}"


def extract_datetime_from_string(date_string,
                                 prefix: Optional[str] = ''):
    """
    Extracts a datetime object from a string.

    Parameters
    ----------
    date_string : str
        The input string containing the datetime information.

    prefix : str, optional
        The prefix of the datatime. Defaults to ''.

    Returns
    -------
    string or None
        A string with format YYYY-mm-ddTHH:MM:SS if successful,
        or None if there was an error.

    Notes
    -----
    This function uses a regular expression to extract a datetime string
    from the input string and then converts it to a string
    with format YYYY-mm-ddTHH:MM:SS.

    Examples
    --------
    >>> date_string = "Some text here 2023-12-10 14:30:00 and more text"
    >>> result = extract_datetime_from_string(date_string)
    >>> print(result)
    2023-12-10T14:30:00

    """
    # Define a regular expression pattern for the datetime format
    pattern = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})"

    # Search for the pattern in the string
    match = re.search(pattern, date_string)

    if match:
        # Extract the matched datetime string
        datetime_string = match.group(1)

        # Convert the datetime string to a datetime object
        try:
            datetime_object = \
                datetime.strptime(datetime_string, "%Y-%m-%d %H:%M:%S")
            return f'{prefix}{datetime_object.strftime("%Y-%m-%dT%H:%M:%S")}'
        except ValueError:
            return None
    else:
        return None