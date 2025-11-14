import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import json

def before_record_request(request):
    if request.body:
        try:
            body = json.loads(request.body)
            if 'user_prompt_id' in body:
                body['user_prompt_id'] = 'CENSORED-USER-PROMPT-ID'
            request.body = json.dumps(body).encode('utf-8')
        except (json.JSONDecodeError, AttributeError):
            pass
    return request

@pytest.fixture(scope="module")
def vcr_config():
    return {
        # 1. This prevents your secret token from being saved in the YAML file
        "filter_headers": [
            ("authorization", "CENSORED-OAUTH-TOKEN")
        ],
        "before_record_request": before_record_request,

        # 2. This tells VCR to ignore the auth header when matching
        #    This is the key to making playback work!
        "match_on": [
            "method",
            "scheme",
            "host",
            "port",
            "path",
            "query",
            "body",
        ],
    }

@pytest.fixture(scope="module")
def shared_tmpdir(tmp_path_factory):
    """
    Creates a temporary directory shared across all tests in a module.

    The directory is created by the tmp_path_factory and will be unique
    for each module. It's automatically cleaned up by pytest after
    all tests in the module have run.
    """
    # Create a base temporary directory for the module
    module_tmp_dir = tmp_path_factory.mktemp("shared_module_dir")

    # Yield the path to the tests
    yield module_tmp_dir

    # No cleanup code is needed here; pytest's tmp_path_factory
    # handles the removal of the directory and its contents.

@pytest.fixture
def mock_llm_user_path(shared_tmpdir, monkeypatch):
    """
    Creates a dedicated 'llm.datasette.io' user directory inside a
    temporary folder and sets the LLM_USER_PATH environment variable to it.
    """
    # tmp_path is the modern (pathlib.Path) version of tmpdir
    user_dir = shared_tmpdir / "llm.datasette.io"
    user_dir.mkdir()

    # monkeypatch automatically reverts this after the test
    monkeypatch.setenv("LLM_USER_PATH", str(user_dir))

    # We yield the path in case the test wants to use it
    yield user_dir

    # Cleanup is handled automatically by pytest:
    # - monkeypatch reverts the environment variable.
    # - tmp_path fixture removes the temporary directory.
