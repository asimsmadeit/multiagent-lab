# Refactoring Audit

This directory records the compatibility, correctness, and architecture audit of the negotiation and interpretability systems. The spelling `refactopring` is retained from the requested artifact name.

## Documents

1. [Concordia Compatibility Audit](01_CONCORDIA_COMPATIBILITY_AUDIT.md) - pinned upstream versions, breaking interfaces, and migration decisions.
2. [Negotiation Module Audit](02_NEGOTIATION_MODULE_AUDIT.md) - component-by-component behavior, defects, fixes, and residual risks.
3. [Interpretability Audit](03_INTERPRETABILITY_AUDIT.md) - activation, labels, splits, serialization, causal evaluation, and data-quality findings.
4. [Verification Matrix](04_VERIFICATION_MATRIX.md) - offline contract tests, model-backed tests, commands, and results.
5. [Ideal Refactor Plan](05_IDEAL_REFACTOR_PLAN.md) - target architecture, phases, migration gates, and external justification.

## Status Vocabulary

- **Fixed:** implementation and regression test are present.
- **Verified:** existing implementation was exercised and behaved as specified.
- **Deferred:** valid issue intentionally left for the planned architecture change.
- **Blocked:** verification requires unavailable data, credentials, hardware, or a policy decision.
- **Invalidated:** an earlier concern was disproved by source inspection or testing.

The audit distinguishes verifiable behavior from inferred intent and preserves the repository's scientific invariants: acting-agent labels are not agent perception labels, only negotiation turns train primary probes, and trials/dyads remain grouped across splits.
