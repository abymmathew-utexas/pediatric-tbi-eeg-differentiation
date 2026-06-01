# Data Governance

## Principles
- Use only de-identified data.
- Never store PHI/PII in this repository.
- Restrict data access to approved team members.
- Log data lineage and transformations.

## Allowed in Repository
- Code
- Synthetic or toy sample data
- Metadata schemas
- Derived, non-identifying summary statistics

## Not Allowed in Repository
- Raw identifiable EEG files
- Dates of birth, MRNs, names, addresses, or any direct identifiers
- Exported reports containing re-identification risk

## Data Handling Workflow
1. Access data from approved secure environment.
2. Run preprocessing and modeling in approved compute environment.
3. Export only approved non-identifying artifacts to this repository.
4. Peer review outputs before publication or sharing.

## Compliance
This project should adhere to all applicable IRB, HIPAA, and institutional compliance requirements.
