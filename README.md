# HAMST — Hash Array Mapped Shareable Trie

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE-MIT)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE-APACHE)

A persistent [Hash Array Mapped Trie](https://en.wikipedia.org/wiki/Hash_array_mapped_trie)
(HAMT) in Rust whose internal nodes are immutable and cheaply shareable between
copies — including across threads.

Cloning a `Hamt` is O(1): it bumps a reference to the root. Inserting, removing,
updating, or replacing a key only allocates fresh nodes along the path from the
root to the affected leaf; every other node is shared with the previous version.
When the last copy of a version is dropped, the nodes unique to it are freed
automatically.

## Features

- **Persistent / structurally shared.** Every `Hamt` is immutable; mutations
  return a new `Hamt` that shares untouched nodes with its source.
- **Cheap clones.** `Hamt::clone` does not walk the tree.
- **Thaw / freeze mutation.** For bulk edits, thaw into a `HamtMut`, stage many
  changes in place, then freeze back into an immutable `Hamt`. During the
  mutation, unchanged sub-trees remain shared with the original.
- **Pluggable hasher.** Defaults to `std::collections::hash_map::DefaultHasher`;
  any type implementing `std::hash::Hasher + Default` can be used instead.
- **Collision handling.** Full-hash collisions are resolved by per-bucket
  collision lists rather than by unbounded tree depth.
- **Typed errors.** Each fallible operation has a dedicated error enum
  (`InsertError`, `RemoveError`, `UpdateError`, `ReplaceError`).
- **`hamt!` macro.** Hash-map-like literal syntax for building a `Hamt`.
- **No runtime dependencies.** `smoke` / `smoke-macros` are dev-only.

## Installation

```toml
[dependencies]
hamst = "0.1"
```

Requires Rust 2024 edition or newer.

## Quick start

```rust
use hamst::Hamt;

let h0: Hamt<String, u32> = Hamt::new();

// Each mutate_freeze call returns a brand-new Hamt. h0 is untouched.
let h1 = h0.mutate_freeze(|h| {
    h.insert("one".to_string(), 1)?;
    h.insert("two".to_string(), 2)
}).unwrap();

assert_eq!(h1.lookup(&"one".to_string()), Some(&1));
assert_eq!(h1.size(), 2);
assert!(h0.is_empty());

// h1 and h2 share all the nodes they can.
let h2 = h1.mutate_freeze(|h| h.insert("three".to_string(), 3)).unwrap();
assert_eq!(h2.size(), 3);
assert_eq!(h1.size(), 2); // still 2 — h1 is persistent
```

## The two types

### `Hamt<K, V, H>` — immutable, shareable

The public, read-only view. Safe to clone freely and to share between threads
(given `K: Send + Sync`, `V: Send + Sync`).

| Method                | Purpose                                      |
| --------------------- | -------------------------------------------- |
| `new()`               | Empty HAMT                                   |
| `is_empty()`          | `true` iff the root has no entries           |
| `size()`              | Total number of key/value pairs              |
| `lookup(&k)`          | `Option<&V>` for key `k`                     |
| `contains_key(&k)`    | Shortcut for `lookup(&k).is_some()`          |
| `iter()`              | Iterator yielding `(&K, &V)`                 |
| `thaw()`              | Convert into a `HamtMut` for mutation        |
| `mutate_freeze(f)`    | Thaw → apply `f` → freeze, all in one call   |
| `mutate_freeze_ret(f)`| Same, but also returns a value from `f`      |

`Hamt` implements `Clone`, `Default`, `PartialEq`, `Eq`, and
`FromIterator<(K, V)>`.

### `HamtMut<K, V, H>` — staging area for edits

Produced by `Hamt::thaw` (or built from scratch with `HamtMut::new`). All
modifying operations live here; calling `freeze()` returns an immutable
`Hamt` again.

| Method                                  | Purpose                                   |
| --------------------------------------- | ----------------------------------------- |
| `insert(k, v)`                          | Fails with `InsertError::EntryExists` if `k` already present |
| `remove(&k)`                            | Fails with `RemoveError::KeyNotFound`     |
| `remove_match(&k, &v)`                  | Remove only if the stored value equals `v` |
| `replace(&k, v)`                        | Replace and return the previous value      |
| `replace_with(&k, &#124;old&#124; new)` | Replace via closure on the current value  |
| `update(&k, &#124;old&#124; Result<Option<V>, E>)` | Update or delete (return `None`) |
| `insert_or_update(k, v, f)`             | Insert if absent, else apply `f`          |
| `insert_or_update_simple(k, v, f)`      | Same, with an infallible closure          |
| `freeze()`                              | Back to an immutable `Hamt`               |

## The `hamt!` macro

```rust
#[macro_use]
extern crate hamst;

use hamst::Hamt;

let h: Hamt<&str, u32> = hamt!{
    "one"   => 1,
    "two"   => 2,
    "three" => 3,
};
```

Duplicate keys keep the last value given.

## Using a custom hasher

```rust
use hamst::Hamt;
use std::collections::hash_map::DefaultHasher;

// Hasher is a type parameter (defaults to DefaultHasher).
let h: Hamt<u32, u32, DefaultHasher> = Hamt::new();
```

Any `H: std::hash::Hasher + Default` will do, so you can plug in `fxhash`,
`ahash`, `siphash`, or a deterministic hasher for reproducible tests.

## How it works

A HAMT is a trie indexed by chunks of a key's hash. Each internal node holds a
bitmap of which hash-prefixes are populated plus a compact array of child
pointers — so a node with only a handful of children stores only those children,
not a full 32- or 64-wide array.

`hamst` adds two properties on top of the classic structure:

1. **Structural sharing.** `Arc`-like reference counting on every node means
   that two HAMTs that only differ in a few leaves share the rest of their
   tree. A write allocates one new node per level on the modified path —
   typically `O(log n)` nodes, not `O(n)`.
2. **Thaw / freeze.** When you are about to perform many mutations, thawing
   gives you a `HamtMut` that edits nodes in place *where it owns them
   uniquely* and falls back to copy-on-write where it does not. Freezing
   converts the staging tree back into an immutable, shareable one without
   further copying.

This design is well-suited for version histories, snapshots, speculative
execution, undo stacks, and passing "before" and "after" copies between
threads without locks.

## Error types

All fallible operations return a dedicated error enum so the call site can
distinguish cases without string matching:

- `InsertError::EntryExists`
- `RemoveError::{KeyNotFound, ValueNotMatching}`
- `UpdateError::{KeyNotFound, ValueCallbackError(E)}`
- `ReplaceError::KeyNotFound`

## Testing

Property-based tests live in `src/lib.rs` and use
[`smoke`](https://crates.io/crates/smoke). They build a random sequence of
operations (`Insert`, `DeleteOne`, `DeleteOneMatching`, `Update`,
`UpdateRemoval`, `Replace`, `ReplaceWith`), apply it to both a `Hamt` and a
reference `BTreeMap`, and check that the two agree on every key — both via
point lookups and via iteration.

```sh
cargo test
```

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.
