"""
Generate synthetic Hungarian training examples per Article 9 category
using Azure OpenAI. Output is JSONL written to data/synthetic/.

Usage:
    export AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com/
    export AZURE_OPENAI_API_KEY=<your-key>
    export AZURE_OPENAI_DEPLOYMENT=gpt-4o   # or your deployment name
    python -m spd.synth --category health --count 100 --out data/synthetic/health.jsonl
"""
import argparse
import json
import os
import sys
import uuid
from pathlib import Path

from openai import AzureOpenAI

from spd.categories import Article9Category, CATEGORIES

_SYSTEM = (
    "Te egy adatgeneráló asszisztens vagy. Kizárólag válaszolj egy JSON tömbbel, "
    "semmi más magyarázattal."
)

_PROMPTS: dict[str, str] = {
    "ethnicity": (
        "Generálj {n} rövid, természetes hangvételű magyar bankos chat üzenetet (1-3 mondat), "
        "amelyek tartalmazzák a feladó faji vagy etnikai hátterét. "
        "Légy változatos: közvetlen kijelentés, harmadik személyre utalás, implicit utalás. "
        "Válasz: JSON tömb stringekkel."
    ),
    "political_opinion": (
        "Generálj {n} rövid magyar bankos chat üzenetet, amelyek tartalmazzák a feladó "
        "politikai nézetét vagy párttagságát. JSON tömb."
    ),
    "religion_belief": (
        "Generálj {n} rövid magyar bankos chat üzenetet, amelyek tartalmazzák a feladó "
        "vallási vagy világnézeti hovatartozását (pl. katolikus, református, zsidó, ateista). "
        "JSON tömb."
    ),
    "trade_union": (
        "Generálj {n} rövid magyar bankos chat üzenetet, amelyek tartalmazzák a feladó "
        "szakszervezeti tagságát. JSON tömb."
    ),
    "genetic": (
        "Generálj {n} rövid magyar bankos chat üzenetet, amelyek genetikai adatot tartalmaznak "
        "(pl. örökletes betegség, DNS-teszt). JSON tömb."
    ),
    "health": (
        "Generálj {n} rövid magyar bankos chat üzenetet, amelyek egészségügyi adatot "
        "tartalmaznak (betegség, fogyatékosság, gyógyszer, kezelés). JSON tömb."
    ),
    "sex_life_orientation": (
        "Generálj {n} rövid magyar bankos chat üzenetet, amelyek tartalmazzák a feladó "
        "szexuális életét vagy orientációját. JSON tömb."
    ),
}

assert set(_PROMPTS) == {c.value for c in Article9Category}, (
    f"_PROMPTS keys out of sync with Article9Category"
)


def _make_negative_prompt(n: int) -> str:
    return (
        f"Generálj {n} rövid, természetes hangvételű magyar bankos chat üzenetet (1-3 mondat), "
        "amelyek NEM tartalmaznak GDPR 9. cikk szerinti különleges kategóriájú adatot. "
        "Egyszerű banki kérdések, tranzakciók, számlaegyenleg, hitelkérelmek. JSON tömb."
    )


def generate(category: str, count: int, out_path: str) -> None:
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")

    if not endpoint or not api_key:
        print(
            "ERROR: Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY env vars.",
            file=sys.stderr,
        )
        sys.exit(1)

    api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01")
    client = AzureOpenAI(azure_endpoint=endpoint, api_key=api_key, api_version=api_version)

    is_negative = category == "negative"
    prompt = _make_negative_prompt(count) if is_negative else _PROMPTS[category].format(n=count)

    response = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=0.9,
        response_format={"type": "text"},
    )

    raw = response.choices[0].message.content or ""
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"ERROR: model returned non-JSON: {exc}\nRaw: {raw[:200]}", file=sys.stderr)
        sys.exit(1)
    if isinstance(parsed, list):
        texts = parsed
    elif isinstance(parsed, dict) and parsed:
        candidate = next(iter(parsed.values()))
        if not isinstance(candidate, list):
            print(f"ERROR: unexpected JSON shape: {raw[:200]}", file=sys.stderr)
            sys.exit(1)
        texts = candidate
    else:
        print(f"ERROR: unexpected JSON shape: {raw[:200]}", file=sys.stderr)
        sys.exit(1)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(out_path, "a") as f:
        for text in texts:
            if not isinstance(text, str) or not text.strip():
                continue
            labels = {cat: 0 for cat in CATEGORIES}
            if not is_negative:
                labels[category] = 1
            record = {
                "id": str(uuid.uuid4()),
                "text": text.strip(),
                "lang": "hu",
                "labels": labels,
                "source": "synthetic",
                "annotator": f"azure-{deployment}",
                "notes": "",
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"Wrote {written} examples to {out_path}")
    if written < count:
        print(f"WARNING: requested {count} but only wrote {written} examples.", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", required=True,
                        choices=CATEGORIES + ["negative"],
                        help="Article 9 category or 'negative'")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    generate(args.category, args.count, args.out)


if __name__ == "__main__":
    main()
