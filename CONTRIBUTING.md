# Contributing to miniml

## Development Setup

```bash
# Prerequisites: Rust 1.75+, Node.js 20+, pnpm 9+
git clone https://github.com/seanchatmangpt/miniml.git
cd miniml
pnpm install
pnpm build:wasm    # Build Rust → WASM
pnpm build         # Build TypeScript bundle
pnpm test          # Run tests
```

## Project Structure

```
miniml/
├── crates/miniml-core/    # Rust WASM core
│   └── src/               # ML algorithm implementations
├── packages/miniml/       # TypeScript npm package
│   ├── src/               # TypeScript wrapper + types
│   ├── wasm/              # Built WASM artifacts
│   └── dist/              # Published package output
└── docs/                  # Documentation
```

## Development Workflow

1. **Modify Rust code** in `crates/miniml-core/src/`
2. **Rebuild WASM**: `pnpm build:wasm`
3. **Update TypeScript wrapper** in `packages/miniml/src/index.ts` if API changed
4. **Rebuild bundle**: `pnpm build`
5. **Run tests**: `pnpm test`
6. **Type check**: `pnpm typecheck`

## Adding a New Algorithm

1. Implement in Rust with `#[wasm_bindgen]` in the appropriate module
2. Register in `crates/miniml-core/src/lib.rs`
3. Rebuild WASM: `pnpm build:wasm`
4. Add TypeScript wrapper in `packages/miniml/src/index.ts`
5. Add types in `packages/miniml/src/types.ts`
6. Add tests in `packages/miniml/src/index.test.ts`
7. Run full verification: `pnpm build && pnpm test && pnpm typecheck`

## Code Standards

- **Rust**: No panics in WASM code. Use `Result` types for error handling. Run `cargo fmt` and `cargo clippy`.
- **TypeScript**: Strict mode. No `any` types. All functions must have type signatures.
- **Tests**: All tests must pass. Use `vitest`. Tests must be deterministic (seeded random).

## Commit Convention

```
type(scope): description

Types: feat, fix, docs, refactor, test, chore
```

## Pull Request Process

1. Create a feature branch from `main`
2. Ensure all tests pass (`pnpm test`)
3. Ensure typecheck passes (`pnpm typecheck`)
4. Ensure build succeeds (`pnpm build`)
5. Keep PRs focused on a single concern

## License

Contributions are accepted under the BSL-1.1 license. By contributing, you agree that your contributions will be licensed under the same terms.
