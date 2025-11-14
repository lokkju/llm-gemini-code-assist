import pytest


def _assert_recursive_contains(actual, expected, path=""):
    """
    Internal helper to recursively assert 'actual' contains 'expected'.
    Ignores extra keys in 'actual' dicts.
    Requires exact matches for lists and simple values.
    """

    # Case 1: Expected is a dictionary
    if isinstance(expected, dict):
        # Check that 'actual' is also a dict
        if not isinstance(actual, dict):
            pytest.fail(
                f"Type mismatch at path '{path}': Expected dict, got {type(actual).__name__}"
            )

        # Check all keys from 'expected' exist in 'actual'
        for key, expected_value in expected.items():
            current_path = f"{path}.{key}" if path else key
            if key not in actual:
                pytest.fail(f"Missing key '{key}' in actual response at path: {path}")

            # Recurse to check the value
            _assert_recursive_contains(actual[key], expected_value, path=current_path)

    # Case 2: Expected is a list
    elif isinstance(expected, list):
        # Check that 'actual' is also a list
        if not isinstance(actual, list):
            pytest.fail(
                f"Type mismatch at path '{path}': Expected list, got {type(actual).__name__}"
            )

        # Check for exact length match (adjust if partial lists are OK)
        if len(actual) != len(expected):
            pytest.fail(
                f"List length mismatch for key: '{path}'. "
                f"Expected {len(expected)}, got {len(actual)}"
            )

        # Recurse for each item in the list
        for i, expected_item in enumerate(expected):
            item_path = f"{path}[{i}]"
            _assert_recursive_contains(actual[i], expected_item, path=item_path)

    # Case 3: Simple value (str, int, bool, etc.)
    else:
        # Simple value comparison
        if actual != expected:
            pytest.fail(f"Value mismatch for key: '{path}'. Expected '{expected}', got '{actual}'")


def assert_dict_contains(actual, expected, path=""):
    """
    Recursively asserts that the 'actual' dict contains all keys and values
    from the 'expected' dict, ignoring extra keys in 'actual'.

    This is a wrapper for the fully recursive helper.
    """
    # This entry-point function expects the top level to be a dict.
    if not isinstance(actual, dict):
        pytest.fail(f"Expected 'actual' to be a dict at path: {path or 'root'}")
    if not isinstance(expected, dict):
        pytest.fail("Expected 'expected' to be a dict.")

    # Call the helper that can handle any nested type
    _assert_recursive_contains(actual, expected, path)


def assert_structure_matches(actual, schema, path=""):
    """
    Recursively asserts that the 'actual' data matches the 'schema' structure.

    - 'actual' is the JSON data you received.
    - 'schema' is a dict/list/type defining the expected structure.

    - Ignores extra keys in 'actual' dictionaries.
    - Validates types, not specific values.
    """

    # Case 1: Schema is a dictionary
    if isinstance(schema, dict):
        if not isinstance(actual, dict):
            pytest.fail(
                f"Type mismatch at path '{path}': Expected dict, got {type(actual).__name__}"
            )

        # Iterate through the keys defined in the SCHEMA
        for key, sub_schema in schema.items():
            current_path = f"{path}.{key}" if path else key

            # Check that the key exists in the actual response
            if key not in actual:
                pytest.fail(f"Missing key at path '{path}': Expected key '{key}'")

            # Recurse to check the structure of the sub-element
            assert_structure_matches(actual[key], sub_schema, path=current_path)

    # Case 2: Schema is a list (defines the structure for all items)
    elif isinstance(schema, list):
        if not isinstance(actual, list):
            pytest.fail(
                f"Type mismatch at path '{path}': Expected list, got {type(actual).__name__}"
            )

        # If the schema list is empty (schema=[]), it matches any list (including empty).
        if not schema:
            return

        # Use the first element of the schema as the template for ALL items
        item_schema = schema[0]
        for index, actual_item in enumerate(actual):
            current_path = f"{path}[{index}]"
            assert_structure_matches(actual_item, item_schema, path=current_path)

    # Case 3: Schema is a type (str, int, bool, float, etc.)
    elif isinstance(schema, type):
        if not isinstance(actual, schema):
            pytest.fail(
                f"Type mismatch at path '{path}': Expected type {schema.__name__}, "
                f"got {type(actual).__name__}"
            )

    # Case 4: Invalid schema definition
    else:
        raise TypeError(
            f"Invalid schema at path '{path}'. Schema must be a dict, list, or type (like str). "
            f"Got {type(schema)}"
        )


def assert_gemini_2_5_flash_lite_response(response):
    expected_response = {
        "candidates": [
            {
                "finishReason": "STOP",
            }
        ],
        "modelVersion": "gemini-2.5-flash-lite",
    }
    assert_dict_contains(response.response_json, expected_response)


def assert_gemini_2_5_flash_response(response):
    expected_response = {
        "candidates": [
            {
                "finishReason": "STOP",
                "safetyRatings": [
                    {
                        "category": "HARM_CATEGORY_HATE_SPEECH",
                        "probability": "NEGLIGIBLE",
                    },
                    {
                        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                        "probability": "NEGLIGIBLE",
                    },
                    {
                        "category": "HARM_CATEGORY_HARASSMENT",
                        "probability": "NEGLIGIBLE",
                    },
                    {
                        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "probability": "NEGLIGIBLE",
                    },
                ],
            }
        ],
        "modelVersion": "gemini-2.5-flash",
    }
    assert_dict_contains(response.response_json, expected_response)
