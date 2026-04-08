# Security Policy

## Reporting Vulnerabilities

If you discover a security vulnerability in miniml, please report it responsibly.

**Do not** open a public GitHub issue for security vulnerabilities.

Instead, please email: info@chatmangpt.com

Include the following in your report:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Any suggested fixes (optional)

## Response Timeline

- Acknowledgment within 48 hours
- Initial assessment within 7 days
- Fix timeline communicated within 14 days
- CVE requested if applicable

## Supported Versions

| Version | Supported |
|---------|-----------|
| 26.x   | Yes       |
| < 26    | No        |

## Scope

This security policy covers:
- The `@seanchatmangpt/wminml` npm package
- The `wminml` Rust crate
- The WASM binary (`wminml_bg.wasm`)

## Known Limitations

- WASM runs in the browser sandbox — no filesystem or network access from WASM code
- No user data is collected or transmitted by the library
- The library has no runtime dependencies (zero-dep)
