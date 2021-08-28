//! HAMST - Hash Array Mapped Shareable Tries
//!
//! Each key is hashed and store related to the hash value.
//!
//! When cloning the data structure, the nodes are shared, so that the operation is
//! is really cheap. When modifying the data structure, after an explicit thawing,
//! the mutable structure share the unmodified node and only nodes requiring modification
//! will be re-created from the node onwards to the leaf.

#![allow(dead_code)]
#[cfg(test)]
extern crate quickcheck;
#[cfg(test)]
#[macro_use(quickcheck)]
extern crate quickcheck_macros;

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

    use quickcheck::{Arbitrary, Gen};

    use std::cmp;
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

    impl<K: Arbitrary + Clone + Send, V: Arbitrary + Clone + Send> Arbitrary for Plan<K, V> {
        fn arbitrary<G: Gen>(g: &mut G) -> Plan<K, V> {
            let nb_ops = 1000 + cmp::min(SIZE_LIMIT, Arbitrary::arbitrary(g));
            let mut v = Vec::new();
            for _ in 0..nb_ops {
                let op_nb: u32 = Arbitrary::arbitrary(g);
                let op = match op_nb % 7u32 {
                    0 => PlanOperation::Insert(Arbitrary::arbitrary(g), Arbitrary::arbitrary(g)),
                    1 => PlanOperation::DeleteOne(Arbitrary::arbitrary(g)),
                    2 => PlanOperation::DeleteOneMatching(Arbitrary::arbitrary(g)),
                    3 => PlanOperation::Update(Arbitrary::arbitrary(g)),
                    4 => PlanOperation::UpdateRemoval(Arbitrary::arbitrary(g)),
                    5 => PlanOperation::Replace(Arbitrary::arbitrary(g), Arbitrary::arbitrary(g)),
                    6 => PlanOperation::ReplaceWith(Arbitrary::arbitrary(g)),
                    _ => panic!("test internal error: quickcheck tag code is invalid"),
                };
                v.push(op)
            }
            Plan(v)
        }
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
    */

    fn property_btreemap_eq<A: Eq + Ord + Hash, B: PartialEq>(
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

    #[quickcheck]
    fn insert_equivalent(xs: Vec<(String, u32)>) -> bool {
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

    #[derive(Clone, Debug, PartialEq, Eq)]
    pub struct LargeVec<A>(Vec<A>);

    const LARGE_MIN: usize = 1000;
    const LARGE_DIFF: usize = 1000;

    impl<A: Arbitrary + Clone + PartialEq + Send + 'static> Arbitrary for LargeVec<A> {
        fn arbitrary<G: Gen>(g: &mut G) -> Self {
            let nb = LARGE_MIN + (usize::arbitrary(g) % LARGE_DIFF);
            let mut v = Vec::with_capacity(nb);
            for _ in 0..nb {
                v.push(Arbitrary::arbitrary(g))
            }
            LargeVec(v)
        }
    }

    #[quickcheck]
    fn large_insert_equivalent(xs: LargeVec<(String, u32)>) -> bool {
        let xs = xs.0;
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

    #[quickcheck]
    fn plan_equivalent(xs: Plan<String, u32>) -> bool {
        let (h, reference) = arbitrary_hamt_and_btree(xs, next_u32, |v| v.wrapping_mul(2));
        property_btreemap_eq(&reference, &h)
    }

    #[quickcheck]
    fn iter_equivalent(xs: Plan<String, u32>) -> bool {
        use std::iter::FromIterator;
        let (h, reference) = arbitrary_hamt_and_btree(xs, next_u32, |v| v.wrapping_mul(2));
        let after_iter = BTreeMap::from_iter(h.iter().map(|(k, v)| (k.clone(), *v)));
        reference == after_iter
    }
}
