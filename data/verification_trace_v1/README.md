# Verification trace then verdict v1

This representation preserves the frozen 1,000/50 high-quality Verification
examples and adds supervised execution traces before the strict final verdict.
It is the Chain-of-Thought or structured-state-tracking improvement described
in the Project 6 specification.

The inference contract remains strict: downstream code parses only the final
explicitly labelled `Verdict:` line and returns either `valid` or
`invalid | step=<integer> | reason=<error_type>`.

Rebuild with:

```bash
python scripts/build_trace_verification_data.py
```
