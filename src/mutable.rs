use super::hash::{Hash, HashedKey, Hasher};
use super::immutable::Hamt;
use super::node::{
    insert_rec, remove_eq_rec, remove_rec, replace_rec, replace_with_rec, update_rec,
};
use std::iter::FromIterator;
use std::marker::PhantomData;

/// Error returned by [`HamtMut::insert`] when the key is already present.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InsertError {
    /// The key already has a value in the map. Use [`HamtMut::replace`]
    /// for overwrite semantics, or [`HamtMut::update`] to transform the
    /// existing value.
    EntryExists,
}

/// Error returned by [`HamtMut::remove`] and [`HamtMut::remove_match`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RemoveError {
    /// The key was not present in the map.
    KeyNotFound,
    /// Only returned by [`HamtMut::remove_match`]: the key was present
    /// but its value did not `PartialEq`-match the expected value.
    ValueNotMatching,
}

/// Error returned by [`HamtMut::update`]. The type parameter `T` is the
/// caller's closure-error type, surfaced via [`UpdateError::ValueCallbackError`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpdateError<T> {
    /// The key was not present in the map. [`HamtMut::insert_or_update`]
    /// turns this case into an insert.
    KeyNotFound,
    /// The update closure returned an error — propagated verbatim.
    ValueCallbackError(T),
}

/// Error returned by [`HamtMut::replace`] and [`HamtMut::replace_with`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplaceError {
    /// The key was not present in the map.
    KeyNotFound,
}

/// Mutable HAMT data structure — staging workspace for in-place mutations.
///
/// Created by thawing a [`Hamt`] (via [`Hamt::thaw`] or
/// [`HamtMut::from`]). Right after creation, all the nodes are still
/// shared with the immutable [`Hamt`]; structural sharing is preserved
/// until the first write.
///
/// Once modification happens, each node from root to leaf along the
/// modified path is copied and kept in an efficient mutable format until
/// freezing. Call [`HamtMut::freeze`] to return to the immutable world
/// — the resulting [`Hamt`] shares all unmodified subtrees with the
/// original.
///
/// # Thread safety
///
/// `HamtMut` is **not `Send`**. A thawed tree holds in-place mutation
/// state that is only valid on the thread that thawed it. Stage your
/// updates on one thread, call [`HamtMut::freeze`] to return to the
/// immutable world, and then hand the fresh [`Hamt`] off to another
/// thread.
pub struct HamtMut<K, V, H = std::collections::hash_map::DefaultHasher> {
    root: Hamt<K, V, H>,
}

/// O(1) clone of the staging workspace — the root [`Hamt`] is cloned, and
/// internal nodes are [`std::sync::Arc`]-shared.
impl<H, K, V> Clone for HamtMut<K, V, H> {
    fn clone(&self) -> Self {
        Self {
            root: self.root.clone(),
        }
    }
}

/// Thaw a [`Hamt`] into a [`HamtMut`] without mutation. Equivalent to
/// [`Hamt::thaw`].
impl<'a, H, K, V> From<&'a Hamt<K, V, H>> for HamtMut<K, V, H> {
    fn from(t: &'a Hamt<K, V, H>) -> Self {
        HamtMut { root: t.clone() }
    }
}

impl<H: Hasher + Default, K: Clone + Eq + Hash, V: Clone> HamtMut<K, V, H> {
    /// Create a new empty mutable HAMT.
    pub fn new() -> Self {
        HamtMut { root: Hamt::new() }
    }
}

impl<H, K, V> HamtMut<K, V, H> {
    /// Freeze this mutable HAMT back into an immutable [`Hamt`].
    ///
    /// Recursively freezes all the in-place mutable nodes along the
    /// modified path. The returned [`Hamt`] shares all unmodified
    /// subtrees with the [`Hamt`] this `HamtMut` was thawed from.
    ///
    /// # Examples
    ///
    /// ```
    /// # use hamst::{Hamt, HamtMut};
    /// let mut staging: HamtMut<u32, u32> = HamtMut::new();
    /// staging.insert(1, 10).unwrap();
    /// let m: Hamt<u32, u32> = staging.freeze();
    /// assert_eq!(m.lookup(&1), Some(&10));
    /// ```
    pub fn freeze(self) -> Hamt<K, V, H> {
        self.root
    }
}

impl<H: Hasher + Default, K: Clone + Eq + Hash, V: Clone> HamtMut<K, V, H> {
    /// Insert a new key/value pair into the HAMT.
    ///
    /// Insert is strict: if the key already exists, this returns
    /// [`InsertError::EntryExists`] rather than silently overwriting.
    /// Use [`HamtMut::replace`] for overwrite semantics, or
    /// [`HamtMut::insert_or_update`] to atomically insert-or-transform.
    ///
    /// # Examples
    ///
    /// ```
    /// # use hamst::{Hamt, HamtMut};
    /// let mut staging: HamtMut<u32, &str> = HamtMut::new();
    /// staging.insert(1, "a").unwrap();
    /// let m = staging.freeze();
    /// assert_eq!(m.lookup(&1), Some(&"a"));
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`InsertError::EntryExists`] when the key is already
    /// present in the map.
    pub fn insert(&mut self, k: K, v: V) -> Result<(), InsertError> {
        let h = HashedKey::compute(self.root.hasher, &k);
        let newroot = insert_rec(&self.root.root, h, 0, k, v)?;
        self.root = Hamt {
            root: newroot,
            hasher: PhantomData,
        };
        Ok(())
    }
}

impl<H: Hasher + Default, K: Eq + Hash + Clone, V: PartialEq + Clone> HamtMut<K, V, H> {
    /// Remove a key from the HAMT only if its current value `PartialEq`-matches
    /// the supplied `v`.
    ///
    /// Useful for conditional removal — e.g., compare-and-delete patterns
    /// where the caller wants to ensure no concurrent update slipped in
    /// between the read and the remove. (Concurrency here is logical, not
    /// thread-level — `HamtMut` is `!Send`.)
    ///
    /// # Examples
    ///
    /// ```
    /// # use hamst::{Hamt, HamtMut};
    /// let mut staging: HamtMut<u32, &str> = HamtMut::new();
    /// staging.insert(1, "a").unwrap();
    /// staging.remove_match(&1, &"a").unwrap();
    /// let m = staging.freeze();
    /// assert_eq!(m.lookup(&1), None);
    /// ```
    ///
    /// # Errors
    ///
    /// - Returns [`RemoveError::KeyNotFound`] when the key is not present.
    /// - Returns [`RemoveError::ValueNotMatching`] when the key is present
    ///   but its current value does not equal `v`.
    pub fn remove_match(&mut self, k: &K, v: &V) -> Result<(), RemoveError> {
        let h = HashedKey::compute(self.root.hasher, &k);
        let newroot = remove_eq_rec(&self.root.root, h, 0, k, v)?;
        match newroot {
            None => self.root = Hamt::new(),
            Some(r) => {
                self.root = Hamt {
                    root: r,
                    hasher: PhantomData,
                }
            }
        };
        Ok(())
    }
}

impl<H: Hasher + Default, K: Clone + Eq + Hash, V: Clone> HamtMut<K, V, H> {
    /// Remove a key from the HAMT.
    ///
    /// # Examples
    ///
    /// ```
    /// # use hamst::{Hamt, HamtMut};
    /// let mut staging: HamtMut<u32, &str> = HamtMut::new();
    /// staging.insert(1, "a").unwrap();
    /// staging.remove(&1).unwrap();
    /// let m = staging.freeze();
    /// assert_eq!(m.lookup(&1), None);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`RemoveError::KeyNotFound`] when the key is not present.
    pub fn remove(&mut self, k: &K) -> Result<(), RemoveError> {
        let h = HashedKey::compute(self.root.hasher, &k);
        let newroot = remove_rec(&self.root.root, h, 0, k)?;
        match newroot {
            None => self.root = Hamt::new(),
            Some(r) => {
                self.root = Hamt {
                    root: r,
                    hasher: PhantomData,
                }
            }
        }
        Ok(())
    }
}

impl<H: Hasher + Default, K: Eq + Hash + Clone, V: Clone> HamtMut<K, V, H> {
    /// Replace the value at `k` and return the previous value.
    ///
    /// Unlike [`HamtMut::insert`], `replace` requires the key to be
    /// already present — it will not insert. Use it when overwrite
    /// semantics are intended.
    ///
    /// # Examples
    ///
    /// ```
    /// # use hamst::{Hamt, HamtMut};
    /// let mut staging: HamtMut<u32, u32> = HamtMut::new();
    /// staging.insert(1, 10).unwrap();
    /// let old = staging.replace(&1, 99).unwrap();
    /// assert_eq!(old, 10);
    /// let m = staging.freeze();
    /// assert_eq!(m.lookup(&1), Some(&99));
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`ReplaceError::KeyNotFound`] when the key is not present.
    pub fn replace(&mut self, k: &K, v: V) -> Result<V, ReplaceError> {
        let h = HashedKey::compute(self.root.hasher, &k);
        let (newroot, oldv) = replace_rec(&self.root.root, h, 0, k, v)?;
        self.root = Hamt {
            root: newroot,
            hasher: PhantomData,
        };
        Ok(oldv)
    }

    /// Replace the value at `k` with the result of `f` applied to the
    /// existing value.
    ///
    /// Like [`HamtMut::replace`] but accepts a transform closure instead
    /// of a fresh value. The closure is invoked exactly once if the key
    /// exists; the new value replaces the old.
    ///
    /// # Examples
    ///
    /// ```
    /// # use hamst::{Hamt, HamtMut};
    /// let mut staging: HamtMut<u32, u32> = HamtMut::new();
    /// staging.insert(1, 10).unwrap();
    /// staging.replace_with(&1, |old| old + 5).unwrap();
    /// let m = staging.freeze();
    /// assert_eq!(m.lookup(&1), Some(&15));
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`ReplaceError::KeyNotFound`] when the key is not present;
    /// the closure `f` is not invoked.
    pub fn replace_with<F>(&mut self, k: &K, f: F) -> Result<(), ReplaceError>
    where
        F: FnOnce(&V) -> V,
    {
        let h = HashedKey::compute(self.root.hasher, &k);
        let newroot = replace_with_rec(&self.root.root, h, 0, k, f)?;
        self.root = Hamt {
            root: newroot,
            hasher: PhantomData,
        };
        Ok(())
    }
}

impl<H: Hasher + Default, K: Eq + Hash + Clone, V: Clone> HamtMut<K, V, H> {
    /// Update the value at `k` by applying the closure `f` to the existing
    /// value.
    ///
    /// If `f` returns `Ok(None)`, the key is removed. If it returns
    /// `Ok(Some(new_v))`, the value is replaced. If it returns `Err(e)`,
    /// the error is wrapped in [`UpdateError::ValueCallbackError`] and
    /// propagated.
    ///
    /// `update` cannot insert a missing key — use
    /// [`HamtMut::insert_or_update`] for that pattern.
    ///
    /// # Examples
    ///
    /// ```
    /// # use hamst::{Hamt, HamtMut};
    /// let mut staging: HamtMut<u32, u32> = HamtMut::new();
    /// staging.insert(1, 10).unwrap();
    /// staging.update::<_, ()>(&1, |old| Ok(Some(old + 1))).unwrap();
    /// let m = staging.freeze();
    /// assert_eq!(m.lookup(&1), Some(&11));
    /// ```
    ///
    /// # Errors
    ///
    /// - Returns [`UpdateError::KeyNotFound`] when the key is not present.
    /// - Returns [`UpdateError::ValueCallbackError`] wrapping any error
    ///   `U` returned by the closure `f`.
    pub fn update<F, U>(&mut self, k: &K, f: F) -> Result<(), UpdateError<U>>
    where
        F: FnOnce(&V) -> Result<Option<V>, U>,
    {
        let h = HashedKey::compute(self.root.hasher, &k);
        let newroot = update_rec(&self.root.root, h, 0, k, f)?;
        match newroot {
            None => self.root = Hamt::new(),
            Some(r) => {
                self.root = Hamt {
                    root: r,
                    hasher: PhantomData,
                }
            }
        };
        Ok(())
    }

    /// Insert a new entry, or update the existing one if the key is
    /// already present.
    ///
    /// If the key is absent, `v` is inserted. If the key is present, `f`
    /// is applied to the existing value; an `Ok(Some(new))` result
    /// replaces the value, and `Ok(None)` removes the key. Closure errors
    /// are propagated as `E` directly — neither [`UpdateError`] nor
    /// [`InsertError`] surface to the caller.
    ///
    /// # Examples
    ///
    /// ```
    /// # use hamst::{Hamt, HamtMut};
    /// let mut staging: HamtMut<u32, u32> = HamtMut::new();
    /// // First call: key absent → insert 10.
    /// staging.insert_or_update::<_, ()>(1, 10, |_| Ok(Some(0))).unwrap();
    /// // Second call: key present → transform via closure.
    /// staging.insert_or_update::<_, ()>(1, 0, |old| Ok(Some(old + 5))).unwrap();
    /// let m = staging.freeze();
    /// assert_eq!(m.lookup(&1), Some(&15));
    /// ```
    ///
    /// # Errors
    ///
    /// Propagates any error `E` returned by the closure `f`. Neither
    /// [`UpdateError::KeyNotFound`] nor [`InsertError::EntryExists`]
    /// surface to the caller — the implementation handles both.
    pub fn insert_or_update<F, E>(&mut self, k: K, v: V, f: F) -> Result<(), E>
    where
        F: FnOnce(&V) -> Result<Option<V>, E>,
        V: Clone,
    {
        match self.update(&k, f) {
            Ok(new_self) => Ok(new_self),
            Err(UpdateError::KeyNotFound) =>
            // unwrap is safe: only error than can be raised is an EntryExist which is fundamentally impossible in this error case handling
            {
                Ok(self.insert(k, v).unwrap())
            }
            Err(UpdateError::ValueCallbackError(x)) => Err(x),
        }
    }

    /// Infallible variant of [`HamtMut::insert_or_update`].
    ///
    /// The closure `f` returns `Option<V>` directly (no `Result`), so
    /// nothing can fail and the function returns `()`. This is the
    /// machinery behind the [`crate::hamt!`] macro.
    ///
    /// # Examples
    ///
    /// ```
    /// # use hamst::{Hamt, HamtMut};
    /// let mut staging: HamtMut<u32, u32> = HamtMut::new();
    /// staging.insert_or_update_simple(1, 10, |_| Some(0));
    /// staging.insert_or_update_simple(1, 0, |old| Some(old + 5));
    /// let m = staging.freeze();
    /// assert_eq!(m.lookup(&1), Some(&15));
    /// ```
    pub fn insert_or_update_simple<F>(&mut self, k: K, v: V, f: F) -> ()
    where
        F: for<'a> FnOnce(&'a V) -> Option<V>,
        V: Clone,
    {
        match self.update(&k, |x| Ok(f(x))) {
            Ok(new_self) => new_self,
            Err(UpdateError::ValueCallbackError(())) => unreachable!(), // callback always wrapped in Ok
            Err(UpdateError::KeyNotFound) => {
                // unwrap is safe: only error than can be raised is an EntryExist which is fundamentally impossible in this error case handling
                self.insert(k, v).unwrap()
            }
        }
    }
}

/// Build a [`HamtMut`] from an iterator of `(K, V)` pairs. Duplicate keys:
/// first insert wins (subsequent inserts of the same key surface
/// [`InsertError::EntryExists`] internally and are silently dropped).
impl<H: Default + Hasher, K: Eq + Hash + Clone, V: Clone> FromIterator<(K, V)>
    for HamtMut<K, V, H>
{
    fn from_iter<I: IntoIterator<Item = (K, V)>>(iter: I) -> Self {
        let mut h = HamtMut::new();
        for (k, v) in iter {
            match h.insert(k, v) {
                Err(_) => {}
                Ok(()) => (),
            }
        }
        h
    }
}
