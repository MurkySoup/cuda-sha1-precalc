# Contributing

Thank you for your interest in this project.

Contributions are welcome — provided they align with the project’s goals and constraints.

---

## Guiding Principles

This repository values:

- Clear, readable code
- Explicit design decisions
- Documentation alongside implementation
- Performance improvements that can be explained and defended
- Respect for future maintainers

---

## What Makes a Good Contribution

Good contributions typically:

- Solve a clearly stated problem
- Include a short explanation of *why* the change exists
- Avoid unnecessary abstraction
- Do not silently change performance characteristics
- Preserve or improve code readability

Documentation improvements are always welcome.

---

## What Is Unlikely to Be Accepted

Changes are unlikely to be accepted if they:

- Add significant complexity without clear benefit
- Obscure GPU behavior behind generic frameworks
- Introduce auto-magic tuning logic without transparency
- Optimize for a single GPU at the expense of portability (without justification)
- Remove warnings, checks, or explanatory comments

---

## Performance Claims

If a change claims performance improvements:

- Describe the test setup
- State the GPU architecture used
- Note whether the change is architecture-specific

Benchmarks without context are not persuasive.

---

## Tone and Conduct

Be curious. Be precise. Be kind.

This project is intended to be educational as well as performant.
