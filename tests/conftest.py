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
