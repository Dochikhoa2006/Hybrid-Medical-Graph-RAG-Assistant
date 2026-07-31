# Security policy

## Reporting a vulnerability

Please report suspected vulnerabilities privately to [dochikhoa2006@gmail.com](mailto:dochikhoa2006@gmail.com). Include a concise description, affected files or versions, reproduction steps, impact, and any suggested mitigation.

Do not include real patient information, credentials, private artifact URLs, or exploit data that is unnecessary to reproduce the issue. Please do not open a public GitHub issue for an unresolved vulnerability.

## Supported scope

This is an educational portfolio project rather than a supported production service. Security review is focused on the current `main` branch. No response-time or remediation-time service level is promised.

## Important trust boundaries

### Serialized artifacts

The application loads joblib/pickle-compatible data and enables dangerous deserialization for the LangChain FAISS docstore. A malicious artifact may execute code during loading. Only use artifacts built locally or obtained from a trusted, checksum-verified release channel.

### Medical and personal data

The application writes raw prompts, retrieved context, generated responses, and heuristic scores to a plaintext local log. Do not enter or process personal health information. Do not publish `Chat_History.log` or screenshots containing sensitive queries.

### Credentials and network exposure

- `.env` is ignored by Git and excluded from Docker build context.
- `.env.example` contains placeholders only.
- The current Compose configuration uses a demonstration Neo4j credential; replace it before any networked deployment.
- The Compose development service bind-mounts the repository read-write at runtime, so `.env`, `.git`, logs, and other workspace files remain visible inside that container despite `.dockerignore`.
- Do not expose Streamlit, Neo4j Browser, or Bolt ports to an untrusted network without authentication, access controls, and transport security.

### Neo4j data deletion

Retriever initialization deletes all nodes and drops existing indexes and constraints in the configured Neo4j database before attempting to reload the repository snapshot. Use only a dedicated disposable database. Treat connection changes as destructive.

## Not a healthcare security claim

The repository has not been assessed for HIPAA, GDPR, medical-device, hospital, or other regulated-environment requirements. Local execution alone does not make the system compliant or suitable for patient data.

Review [Safety and limitations](docs/SAFETY_AND_LIMITATIONS.md) before deployment or demonstration.
