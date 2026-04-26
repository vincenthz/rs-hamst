//! HAMST - Hash Array Mapped Shareable Tries
//!
//! Each key is hashed and store related to the hash value.
//!
//! # HAMT — what and why
//!
//! A **Hash Array Mapped Trie** stores each `(key, value)` at the
//! position dictated by the key's 64-bit hash: the hash is sliced
//! into 5-bit chunks, each chunk indexes into a 32-slot sparse
//! array at one level of a trie. Lookups are `O(log₃₂ n)` — on a
//! 1-million-entry map that's at most four indirections. Hash
//! collisions are handled by a small linear bucket at the bottom
//! level; they do not degrade the structure.
//!
//! # Thaw / freeze — how mutation works
//!
//! The top-level [`Hamt`] is strictly immutable. To change anything,
//! call [`Hamt::thaw`] (or the all-in-one [`Hamt::mutate_freeze`])
//! to obtain a [`HamtMut`] — a mutable workspace whose initial
//! state shares every node with the original `Hamt`. Mutations are
//! applied in place on the paths they touch; unmodified paths stay
//! shared. When you are done, [`HamtMut::freeze`] converts the
//! workspace back into an immutable [`Hamt`] and makes the new
//! snapshot shareable again. The original `Hamt` value is never
//! disturbed — both snapshots remain valid after the freeze.
//!
//! # Structural sharing — why clones are cheap
//!
//! Each internal trie node is wrapped in an `Arc`. Cloning a
//! [`Hamt`] is therefore a handful of reference-count bumps, not a
//! deep copy — independent of how many entries the map holds. Two
//! clones of the same map can diverge through independent
//! `mutate_freeze` calls; each freeze produces a new root that
//! shares every unchanged path with the predecessor. This is the
//! "persistent" in "persistent data structure": prior versions
//! stay reachable at negligible cost.
//!
//! # Thread safety
//!
//! An immutable [`Hamt`] is freely shareable across threads — all
//! internal nodes are `Arc`-backed, and a shared `&Hamt` exposes
//! only lookup-style methods. Independent threads may each hold
//! their own clone of the same `Hamt` and read concurrently
//! without synchronisation.
//!
//! The mutable [`HamtMut`] workspace, on the other hand, is **not
//! `Send`**: a thawed tree carries interior-mutability state that
//! is only valid for the thread that thawed it. Stage your updates
//! on one thread, call [`HamtMut::freeze`] to return to the
//! immutable world, and then hand the fresh [`Hamt`] off wherever
//! it needs to go.
//!
//! # When to use this (and when not to)
//!
//! Reach for `hamst` when you need cheap snapshots of a map — for
//! example, undo history, speculative state, or sharing a view of
//! a map across threads without locking. Pick `std::collections::HashMap`
//! instead when you have a single owner and never need multiple
//! versions to coexist: `HashMap` will always win on raw
//! insert/lookup throughput for that workload, because it has no
//! persistence tax to pay.
//!
//! # Examples
//!
//! Build a map with the [`hamt!`] macro literal and look a key up:
//!
//! ```
//! use hamst::{hamt, Hamt};
//! let m: Hamt<u32, &str> = hamt!{ 1 => "a", 2 => "b" };
//! assert_eq!(m.lookup(&1), Some(&"a"));
//! assert_eq!(m.lookup(&3), None);
//! ```
//!
//! Stage a batch of inserts through the thaw / freeze workflow.
//! [`Hamt::mutate_freeze`] is the common shorthand that does both
//! sides for you — thaw, run the closure, freeze:
//!
//! ```
//! use hamst::Hamt;
//! let empty: Hamt<u32, u32> = Hamt::new();
//! let populated = empty.mutate_freeze(|m| {
//!     m.insert(1, 10)?;
//!     m.insert(2, 20)
//! }).unwrap();
//! assert_eq!(populated.lookup(&1), Some(&10));
//! assert_eq!(populated.lookup(&2), Some(&20));
//! // The original `empty` is untouched.
//! assert_eq!(empty.lookup(&1), None);
//! ```
//!
//! # A note for doctest authors
//!
//! The default hasher's iteration order is not stable across Rust
//! versions. If you write a doctest that asserts on the contents
//! produced by [`Hamt::iter`], collect into a [`std::collections::HashSet`]
//! rather than a [`Vec`] — otherwise the test will pass locally
//! and break randomly on a compiler upgrade.

#![deny(missing_docs)]
#![allow(dead_code)]

mod bitmap;
mod collision;
mod hamt;
mod hash;
mod immutable;
mod mutable;
mod node;

pub use hamt::*;

#[cfg(test)]
mod tests {
    use super::*;

    use smoke::Generator;
    use smoke::generator::{self, BoxGenerator};
    use smoke_macros::smoketest;

    //use quickcheck::{Arbitrary, Gen};

    use std::collections::BTreeMap;
    use std::hash::Hash;

    #[derive(Debug, Clone)]
    enum PlanOperation<K, V> {
        Insert(K, V),
        DeleteOneMatching(usize),
        DeleteOne(usize),
        Update(usize),
        UpdateRemoval(usize),
        Replace(usize, V),
        ReplaceWith(usize),
    }

    #[derive(Debug, Clone)]
    struct Plan<K, V>(Vec<PlanOperation<K, V>>);

    const SIZE_LIMIT: usize = 5120;

    fn ascii_string() -> impl Generator<Item = String> {
        let gen_ascii_char = generator::range(30..0x7fu32).map(|x| std::char::from_u32(x).unwrap());
        generator::vector(generator::range(1..8usize), gen_ascii_char)
            .map(|s| s.into_iter().collect::<String>())
    }

    fn gen_plan() -> impl Generator<Item = Plan<String, u32>> {
        let sz = generator::range(1000..1000 + SIZE_LIMIT);
        let g0 = generator::product2(ascii_string(), generator::num(), |g1, g2| {
            PlanOperation::Insert(g1, g2)
        });
        let g1 = generator::num().map(PlanOperation::DeleteOne);
        let g2 = generator::num().map(PlanOperation::DeleteOneMatching);
        let g3 = generator::num().map(PlanOperation::Update);
        let g4 = generator::num().map(PlanOperation::UpdateRemoval);
        let g5 = generator::product2(generator::num(), generator::num(), |g1, g2| {
            PlanOperation::Replace(g1, g2)
        });
        let g6 = generator::num().map(PlanOperation::ReplaceWith);
        let el = generator::choose(vec![
            Box::new(g0.into_boxed()),
            Box::new(g1.into_boxed()),
            Box::new(g2.into_boxed()),
            Box::new(g3.into_boxed()),
            Box::new(g4.into_boxed()),
            Box::new(g5.into_boxed()),
            Box::new(g6.into_boxed()),
        ]);
        generator::vector(sz, el).map(|v| Plan(v))
    }

    #[test]
    fn insert_lookup() {
        let h: Hamt<String, u32> = Hamt::new();

        let k1 = "ABC".to_string();
        let v1 = 12u32;

        let k2 = "DEF".to_string();
        let v2 = 24u32;

        let k3 = "XYZ".to_string();
        let v3 = 42u32;

        let h1 = h.mutate_freeze(|hm| hm.insert(k1.clone(), v1)).unwrap();
        let h2 = h.mutate_freeze(|hm| hm.insert(k2.clone(), v2)).unwrap();

        assert_eq!(h1.lookup(&k1), Some(&v1));
        assert_eq!(h2.lookup(&k2), Some(&v2));
        assert_eq!(h1.lookup(&k2), None);
        assert_eq!(h2.lookup(&k1), None);

        let h3 = h1.mutate_freeze(|hm| hm.insert(k3.clone(), v3)).unwrap();

        assert_eq!(h1.lookup(&k3), None);
        assert_eq!(h2.lookup(&k3), None);

        assert_eq!(h3.lookup(&k1), Some(&v1));
        assert_eq!(h3.lookup(&k2), None);
        assert_eq!(h3.lookup(&k3), Some(&v3));

        let (h4, oldk1) = h3.mutate_freeze_ret(|hm| hm.replace(&k1, v3)).unwrap();
        assert_eq!(oldk1, v1);
        assert_eq!(h4.lookup(&k1), Some(&v3));
    }

    #[test]
    fn dup_insert() {
        let mut h: Hamt<&String, u32> = Hamt::new();
        let dkey = "A".to_string();
        h = h.mutate_freeze(|hm| hm.insert(&dkey, 1)).unwrap();
        assert_eq!(
            h.mutate_freeze(|hm| hm.insert(&dkey, 2)).and(Ok(())),
            Err(InsertError::EntryExists)
        )
    }

    #[test]
    fn empty_size() {
        let h: Hamt<&String, u32> = Hamt::new();
        assert_eq!(h.size(), 0)
    }

    #[test]
    fn delete_key_not_exist() {
        let mut h: Hamt<&String, u32> = Hamt::new();
        let dkey = "A".to_string();
        h = h.mutate_freeze(|hm| hm.insert(&dkey, 1)).unwrap();
        assert_eq!(
            h.mutate_freeze(|hm| hm.remove_match(&&dkey, &2))
                .and(Ok(())),
            Err(RemoveError::ValueNotMatching)
        )
    }

    #[test]
    fn delete_value_not_match() {
        let mut h: Hamt<&String, u32> = Hamt::new();
        let dkey = "A".to_string();
        h = h.mutate_freeze(|hm| hm.insert(&dkey, 1)).unwrap();
        assert_eq!(
            h.mutate_freeze(|hm| hm.remove_match(&&dkey, &2))
                .and(Ok(())),
            Err(RemoveError::ValueNotMatching)
        )
    }

    #[allow(clippy::trivially_copy_pass_by_ref)]
    fn next_u32(x: &u32) -> Result<Option<u32>, ()> {
        Ok(Some(*x + 1))
    }

    #[test]
    fn delete() {
        let mut h: Hamt<String, u32> = Hamt::new();

        let keys = [
            ("KEY1", 10000u32),
            ("KEY2", 20000),
            ("KEY3", 30000),
            ("KEY4", 40000),
            ("KEY5", 50000),
            ("KEY6", 60000),
            ("KEY7", 70000),
            ("KEY8", 80000),
            ("KEY9", 10000),
            ("KEY10", 20000),
            ("KEY11", 30000),
            ("KEY12", 40000),
            ("KEY13", 50000),
            ("KEY14", 60000),
            ("KEY15", 70000),
            ("KEY16", 80000),
        ];

        let k1 = "ABC".to_string();
        let v1 = 12u32;

        let k2 = "DEF".to_string();
        let v2 = 24u32;

        let k3 = "XYZ".to_string();
        let v3 = 42u32;

        h = h
            .mutate_freeze::<InsertError, _>(|hm| {
                for (k, v) in keys.iter() {
                    hm.insert((*k).to_owned(), *v)?;
                }
                Ok(())
            })
            .unwrap();

        h = h
            .mutate_freeze(|hm| {
                hm.insert(k1.clone(), v1)?;
                hm.insert(k2.clone(), v2)?;
                hm.insert(k3.clone(), v3)
            })
            .unwrap();

        let h2 = h
            .mutate_freeze(|hm| hm.remove_match(&k1, &v1))
            .expect("cannot remove from already inserted");

        assert_eq!(h.size(), keys.len() + 3);
        assert_eq!(h2.size(), keys.len() + 2);

        assert_eq!(h.lookup(&k1), Some(&v1));
        assert_eq!(h2.lookup(&k1), None);

        h = h
            .mutate_freeze(|hm| {
                hm.remove_match(&k2, &v2)?;
                hm.remove_match(&k3, &v3)
            })
            .unwrap();

        assert_eq!(
            h.mutate_freeze(|hm| hm.remove_match(&k3, &v3)).and(Ok(())),
            Err(RemoveError::KeyNotFound),
        );
        assert_eq!(
            h.mutate_freeze(|hm| hm.remove_match(&k1, &v2)).and(Ok(())),
            Err(RemoveError::ValueNotMatching),
        );
        assert_eq!(
            h2.mutate_freeze(|hm| hm.insert(k2, v3)).and(Ok(())),
            Err(InsertError::EntryExists)
        );

        assert_eq!(
            h2.mutate_freeze(|hm| hm.update(&"ZZZ".to_string(), next_u32))
                .and(Ok(())),
            Err(UpdateError::KeyNotFound)
        );

        assert_eq!(h.size(), keys.len() + 1);
        assert_eq!(h2.size(), keys.len() + 2);
    }

    //use hash::HashedKey;
    //use std::marker::PhantomData;

    /* commented -- as this doesn't do what it says on the tin.
    it doesn't test for h collision, but node splitting

    #[test]
    fn collision() {
        let k0 = "keyx".to_string();
        let h1 = HashedKey::compute(PhantomData::<DefaultHasher>, &k0);
        let l = h1.level_index(0);
        let mut found = None;
        for i in 0..10000 {
            let x = format!("key{}", i);
            let h2 = HashedKey::compute(PhantomData::<DefaultHasher>, &x);
            if h1 == h2 {
            //if h2.level_index(0) == l {
                found = Some(x.clone());
                break;
            }
        }

        match found {
            None => assert!(false),
            Some(x) => {
                let mut h: Hamt<DefaultHasher, String, u32> = Hamt::new();
                println!("k0: {}", k0);
                h = h.insert(k0.clone(), 1u32).unwrap();
                println!("x: {}", x);
                h = h.insert(x.clone(), 2u32).unwrap();
                assert_eq!(h.size(), 2);
                assert_eq!(h.lookup(&k0), Some(&1u32));
                assert_eq!(h.lookup(&x), Some(&2u32));

                let h2 = h.remove_match(&x, &2u32).unwrap();
                assert_eq!(h2.lookup(&k0), Some(&1u32));
                assert_eq!(h2.size(), 1);

                let h3 = h.remove_match(&k0, &1u32).unwrap();
                assert_eq!(h3.lookup(&x), Some(&2u32));
                assert_eq!(h3.size(), 1);
            }
        }
    }

    fn xproperty_btreemap_eq<A: Eq + Ord + Hash, B: PartialEq>(
        reference: &BTreeMap<A, B>,
        h: &Hamt<A, B>,
    ) -> bool {
        // using the btreemap reference as starting point
        for (k, v) in reference.iter() {
            if h.lookup(k) != Some(v) {
                return false;
            }
        }
        // then asking the hamt for any spurious values
        for (k, v) in h.iter() {
            if reference.get(k) != Some(v) {
                return false;
            }
        }
        true
    }
    */

    fn property_btreemap_eq<A: Eq + Ord + Hash, B: PartialEq>(
        reference: &BTreeMap<A, B>,
        h: &Hamt<A, B>,
    ) -> impl smoke::property::Property + use<A, B> {
        let mut same = true;
        // using the btreemap reference as starting point
        for (k, v) in reference.iter() {
            if h.lookup(k) != Some(v) {
                same = false;
            }
        }
        // then asking the hamt for any spurious values
        for (k, v) in h.iter() {
            if reference.get(k) != Some(v) {
                same = false;
            }
        }
        smoke::property::equal(same, true)
    }

    fn vec_string_int() -> BoxGenerator<Vec<(String, u32)>> {
        let gen_ascii_char = generator::range(30..0x7fu32).map(|x| std::char::from_u32(x).unwrap());
        let string = generator::vector(generator::range(1..8usize), gen_ascii_char)
            .map(|s| s.into_iter().collect::<String>());
        let gen_t = generator::product2(string, generator::num(), |g1, g2| (g1, g2));
        generator::vector(generator::range(1..43), gen_t).into_boxed()
    }

    #[smoketest{xs:vec_string_int()}]
    fn insert_equivalent(xs: Vec<(String, u32)>) {
        let mut reference = BTreeMap::new();
        let mut h: HamtMut<String, u32> = HamtMut::new();
        for (k, v) in xs.iter() {
            if reference.get(k).is_some() {
                continue;
            }
            reference.insert(k.clone(), *v);
            h.insert(k.clone(), *v).expect("insert error");
        }
        property_btreemap_eq(&reference, &h.freeze())
    }

    fn get_key_nth<K: Clone, V>(b: &BTreeMap<K, V>, n: usize) -> Option<K> {
        let keys_nb = b.len();
        if keys_nb == 0 {
            return None;
        }
        let mut keys = b.keys();
        Some(keys.nth(n % keys_nb).unwrap().clone())
    }

    fn arbitrary_hamt_and_btree<K, V, F, G>(
        xs: Plan<K, V>,
        update_f: F,
        replace_with_f: G,
    ) -> (Hamt<K, V>, BTreeMap<K, V>)
    where
        K: Hash + Clone + Eq + Ord + Sync,
        V: Clone + PartialEq + Sync,
        F: Fn(&V) -> Result<Option<V>, ()> + Copy,
        G: Fn(&V) -> V + Copy,
    {
        let mut reference = BTreeMap::new();
        let mut h: HamtMut<K, V> = HamtMut::new();
        //println!("plan {} operations", xs.0.len());
        for op in xs.0.iter() {
            match op {
                PlanOperation::Insert(k, v) => {
                    if reference.get(k).is_some() {
                        continue;
                    }
                    reference.insert(k.clone(), v.clone());
                    h.insert(k.clone(), v.clone()).expect("insert error")
                }
                PlanOperation::DeleteOne(r) => match get_key_nth(&reference, *r) {
                    None => continue,
                    Some(k) => {
                        reference.remove(&k);
                        h.remove(&k).expect("remove error");
                    }
                },
                PlanOperation::DeleteOneMatching(r) => match get_key_nth(&reference, *r) {
                    None => continue,
                    Some(k) => {
                        let v = reference.get(&k).unwrap().clone();
                        reference.remove(&k);
                        h.remove_match(&k, &v).expect("remove match error");
                    }
                },
                PlanOperation::Replace(r, newv) => match get_key_nth(&reference, *r) {
                    None => continue,
                    Some(k) => {
                        let v = reference.get_mut(&k).unwrap();
                        *v = newv.clone();

                        h.replace(&k, newv.clone()).expect("replace error");
                    }
                },
                PlanOperation::ReplaceWith(r) => match get_key_nth(&reference, *r) {
                    None => continue,
                    Some(k) => {
                        let v = reference.get_mut(&k).unwrap();
                        *v = replace_with_f(v);

                        h.replace_with(&k, replace_with_f)
                            .expect("replace with error");
                    }
                },
                PlanOperation::Update(r) => match get_key_nth(&reference, *r) {
                    None => continue,
                    Some(k) => {
                        let v = reference.get_mut(&k).unwrap();
                        match update_f(v).unwrap() {
                            None => {
                                reference.remove(&k);
                            }
                            Some(newv) => *v = newv,
                        }

                        h.update(&k, update_f).expect("update error");
                    }
                },
                PlanOperation::UpdateRemoval(r) => match get_key_nth(&reference, *r) {
                    None => continue,
                    Some(k) => {
                        reference.remove(&k);
                        h.update::<_, ()>(&k, |_| Ok(None))
                            .expect("update removal error");
                    }
                },
            }
        }
        (h.freeze(), reference)
    }

    #[smoketest{xs: gen_plan()}]
    fn plan_equivalent(xs: Plan<String, u32>) -> bool {
        let (h, reference) = arbitrary_hamt_and_btree(xs, next_u32, |v| v.wrapping_mul(2));
        property_btreemap_eq(&reference, &h)
    }

    #[smoketest{xs: gen_plan()}]
    fn iter_equivalent(xs: Plan<String, u32>) -> bool {
        use std::iter::FromIterator;
        let (h, reference) = arbitrary_hamt_and_btree(xs, next_u32, |v| v.wrapping_mul(2));
        let after_iter = BTreeMap::from_iter(h.iter().map(|(k, v)| (k.clone(), *v)));
        smoke::property::equal(reference, after_iter)
    }
}
