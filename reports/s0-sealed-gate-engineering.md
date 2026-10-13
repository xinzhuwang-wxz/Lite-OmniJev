# S0 Sealed Gate Engineering Status

2026-10-14: implemented canary-only sealed summary tooling. The code accepts
redacted annotation exports, computes pre-frozen Cohen kappa from raw double
labels paired by manifest-frozen annotator identities, rejects payload fields and
malformed double labels, enforces family separation between dev and holdout
pools, and writes aggregate-only summaries.

Follow-up hardening on the same branch catches malformed manifest/annotation
inputs as category-only failures, rejects undeclared modalities/pools and empty
or non-positive expected counts, and requires non-canary manifests to declare
four modalities with 500 dev and 1,000 holdout items each. Tests include a
malicious sentinel check to verify labels, item ids, and payload fields do not
appear in stdout, stderr, or the report.

This is not a real S0 sealed-set pass. The active development environment has
root SSH access to the instance and local workspace, so it cannot honestly prove
that the agent cannot read sealed payloads. Real acceptance remains blocked
until the gate runs under an externally controlled identity or host and receives
actual double-human annotations for the required per-modality counts.
