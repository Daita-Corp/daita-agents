## Summary

Brief description of what this PR does.

## Changes

-
-

## Type of Change

- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing

Describe how you tested this change. If you added tests, point to them.

## Checklist

- [ ] Focused tests and the default deterministic suite pass (`python -m pytest`)
- [ ] New tests identify an owner, current contract, defect, and observable failure
- [ ] Live/slow requirements and any unexecuted checks are disclosed
- [ ] Model-facing surface changed? Live smoke recorded on subscription
      (`scripts/live_smoke.py run`; see docs/TESTING.md, Model-surface gate)
- [ ] Pre-commit checks pass (`pre-commit run --all-files`)
- [ ] Public API changes are documented
- [ ] Durable agent-home compatibility was checked; persistence changes include
      the candidate migration, golden fixture, and refreshed release snapshot
