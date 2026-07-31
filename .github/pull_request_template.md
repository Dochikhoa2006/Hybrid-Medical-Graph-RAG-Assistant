## Summary

Describe the problem, the focused change, and why it belongs in this project.

## Validation

List the commands and manual checks you ran.

```text
python -m compileall -q *.py
docker compose config --quiet
```

## Compatibility and safety checklist

- [ ] I kept the change focused and documented any behavior change.
- [ ] I considered root-level import, path, and serialized-artifact compatibility.
- [ ] I added or updated tests where practical.
- [ ] I updated relevant setup, architecture, artifact, or limitation documentation.
- [ ] I included no credentials, `.env` values, raw logs, patient data, or identifying health information.
- [ ] I used fictional medical examples only.
- [ ] I made no unsupported claims about clinical validity, accuracy, confidence, or production readiness.
- [ ] Any new artifact has documented provenance, generation steps, checksum, and trust requirements.

## Screenshots or logs

Include only sanitized material when it is necessary to review the change. Never attach `Chat_History.log`.
