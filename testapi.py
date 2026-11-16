"""Manual smoke test: python testapi.py --document https://host/policy.pdf"""

import argparse
import json
import os
import sys
import time

import requests
from dotenv import load_dotenv


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--document", required=True, help="Public PDF URL")
    parser.add_argument("--query", action="append", help="Repeat for multiple questions")
    args = parser.parse_args()
    token = os.getenv("API_KEY")
    if not token or token.startswith("your_"):
        parser.error("Set API_KEY in the environment or .env")
    url = os.getenv("API_URL", "http://127.0.0.1:8000/api/run")
    payload = {"document": args.document, "queries": args.query or ["What is the waiting period?"]}
    started = time.monotonic()
    try:
        response = requests.post(
            url, headers={"Authorization": f"Bearer {token}"}, json=payload, timeout=(10, 300)
        )
        response.raise_for_status()
        data = response.json()
        answers = data.get("answers")
        if (
            not isinstance(answers, list)
            or len(answers) != len(payload["queries"])
            or any(not isinstance(answer, str) or not answer.strip() for answer in answers)
        ):
            raise ValueError("Unexpected API response shape")
        print(json.dumps(data, indent=2, ensure_ascii=False))
        print(f"Completed in {time.monotonic() - started:.2f}s")
        return 0
    except (requests.RequestException, ValueError) as exc:
        # Avoid printing request URLs or upstream response bodies containing signed tokens.
        print(
            f"Smoke test failed ({type(exc).__name__}). Check the API status and logs.",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
