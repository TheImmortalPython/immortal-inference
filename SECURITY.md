# Security Policy

## Supported Versions

Only the latest release on the `main` branch receives active security updates and patch releases.

| Version | Supported          |
| ------- | ------------------ |
| main    | :white_check_mark: |
| < 0.1   | :x:                |

## Reporting a Vulnerability

**Do not report security vulnerabilities through public GitHub issues, discussions, or pull requests.**

If you discover a potential security vulnerability, memory leak, tensor-level constraint exploit, or execution failure within `immortal-inference`, please submit a report privately via email: theimmortalpythonlabs@proton.me

* **Response Expectation:** Initial acknowledgement within 48 hours, followed by patch status updates.

### What to Include in Your Report

To help us investigate and resolve the issue quickly, please include:
1. A clear description of the vulnerability or boundary bypass.
2. Minimal reproduction code or environment details (PyTorch version, model architecture, logit processor configuration).
3. Any potential impact or proof-of-concept payload.

## Scope & Architectural Disclaimers

`immortal-inference` enforces mathematical containment and logit-level restrictions during inference execution. 

* **Out of Scope:** Stochastic model hallucinatory outputs where logit masks are explicitly bypassed or unconfigured by the user application.
* **In Scope:** Token-masking leaks, runtime execution escapes, illegal state transitions past active Pāṇinian constraint boundaries, and memory/buffer corruption within the core C++/Python binding layer.

Thank you for helping keep non-stochastic inference safe and reliable for everyone.
