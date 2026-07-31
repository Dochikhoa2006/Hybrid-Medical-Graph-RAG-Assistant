# Contributing

Thank you for considering an improvement to SympScan. Contributions should preserve the project's educational scope, make technical claims that the repository can support, and avoid exposing sensitive health information.

## Before opening an issue

- Search existing issues and documentation.
- Remove names, contact details, medical records, tokens, passwords, and raw chat logs.
- Use fictional medical examples only.
- Remember that repository issues are for software behavior, not medical guidance.

Security-sensitive reports should follow [SECURITY.md](SECURITY.md) instead of using a public issue.

## Development setup

Follow [Setup and reproducibility](docs/SETUP.md). The full application requires locally generated artifacts and external services, so explain which environment and artifacts you used when reporting a result.

## Change guidelines

1. Create a focused branch from `main`.
2. Keep each pull request limited to one coherent change.
3. Update documentation when behavior, configuration, artifacts, or limitations change.
4. Add tests when a change introduces testable behavior.
5. Do not commit `.env`, logs, raw data, virtual environments, model caches, or unreviewed serialized artifacts.
6. Do not describe dataset-derived output as clinically verified or production-ready.

The existing root module names and artifact paths are compatibility-sensitive. Serialized objects can depend on Python module/class resolution, and scripts use root-relative paths. A packaging or filename change must include a migration plan and runtime verification rather than a cosmetic move.

## Local checks

Run the checks relevant to your change:

```bash
python -m compileall -q *.py
docker compose config --quiet
git diff --check
```

For runtime changes, also test at least one retrieval request and one chitchat request in a dedicated local environment. Never use real patient data in test prompts or fixtures.

## Pull-request checklist

- The change has a clear purpose and limited scope.
- Existing application behavior is preserved unless the change explicitly documents otherwise.
- Source and setup documentation are updated.
- No secrets, personal data, raw chat logs, or unsafe artifacts are included.
- Medical and performance claims are supported by evidence.
- New dependencies are justified and version compatibility has been checked.

## Generated artifacts

Do not add or replace model, pickle, FAISS, Parquet, or Neo4j artifacts without documenting:

- The source commit and dataset version.
- The exact generation command and environment.
- Cryptographic checksums.
- Artifact size and distribution location.
- Trust and compatibility considerations.

See [Data and artifacts](docs/DATA_AND_ARTIFACTS.md) for the current policy and inventory.
