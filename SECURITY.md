# Security Policy

## Reporting a Vulnerability

**Do NOT open a public issue for security vulnerabilities.**

Email **zhhlbaw2011@gmail.com** with:

1. Description of the vulnerability
2. Steps to reproduce
3. Potential impact
4. Suggested fix (if any)

You will receive a response within 72 hours.

## Scope

In scope:

- **Strategy engine** — pattern detection logic, risk-management bypass
- **IBKR integration** — credential leakage, unsafe order construction, sandbox -> live boundary leaks
- **Replay / backtesting** — deterministic-replay tampering, data corruption
- **Configuration** — secrets exposure, unsafe defaults

## Out of Scope

- **Trading losses or strategy performance** — this is a research framework, not financial advice
- Issues in third-party brokers (IBKR / TWS)
- Issues requiring physical access to the user's machine
- Social engineering attacks

## Disclosure Policy

Coordinated disclosure. Once a fix is released, the reporter is credited (unless anonymous is preferred) in the release notes.

**Note**: This framework ships demo strategies only. Users supply their own alpha. We do not warrant correctness of any included signal logic.
