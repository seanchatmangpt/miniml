# miniml — Claude Code Context

## Quick Start

```bash
pnpm install
pnpm build        # Build WASM + TypeScript bundle
pnpm test         # Run tests
pnpm bench        # Run benchmarks
```

## Architecture

**Monorepo structure:**
- `crates/miniml-core/` — Rust WASM core (ML algorithms)
- `packages/miniml/` — TypeScript npm package wrapper
- `docs/` — Markdown documentation (AutoML guide, algorithms, optimization, performance, examples)

**Build pipeline:**
1. Rust → WASM via `wasm-pack` (`build:wasm`)
2. WASM → `packages/miniml/wasm/`
3. TypeScript → `dist/` via `tsup` (`build`)

**Critical:** Rebuild WASM after any Rust code changes.

## Code Patterns

### Rust (WASM Core)
- Use `#[wasm_bindgen]` for exported functions
- Return `Result` types for error handling
- No panics in WASM code
- **AutoML:** `automl.rs` — Genetic algorithm feature selection + PSO hyperparameter optimization
- **SIMD:** `matrix.rs` — WASM SIMD v128 intrinsics for vectorized operations

### TypeScript
- Always `await init()` before ML functions
- Use `createWorker()` for non-blocking operations
- Strict mode: no `any` types

## Testing

- Framework: Vitest (Node environment, 30s timeout)
- Tests: `src/*.test.ts`
- Run: `pnpm test` or `pnpm test:watch`

## Development Workflow

1. **Modify Rust code:** Edit files in `crates/miniml-core/src/`
2. **Rebuild WASM:** `pnpm build:wasm`
3. **Rebundle:** `pnpm build`
4. **Test:** `pnpm test`
5. **Type check:** `pnpm typecheck`

## Gotchas

1. WASM build is slow (30-60s first time)
2. `wasm-opt = false` in Cargo.toml for faster builds
3. Always run `build:wasm` after Rust changes
4. Run `pnpm` commands from repository root
5. SIMD requires `--features simd` flag for wasm-pack

---

## Documentation

| Document | Purpose |
|----------|---------|
| **README.md** | Root: Overview, quick start, installation |
| **docs/automl.md** | AutoML comprehensive guide |
| **docs/algorithms.md** | Complete algorithm reference |
| **docs/optimization.md** | Metaheuristic optimization suite |
| **docs/performance.md** | Benchmarks and SIMD |
| **docs/examples.md** | Real-world usage examples |
| **packages/miniml/README.md** | npm package usage, API reference |
