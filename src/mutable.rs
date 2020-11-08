use super::hash::{Hash, HashedKey, Hasher};
use super::immutable::Hamt;
use super::node::{
    insert_rec, remove_eq_rec, remove_rec, replace_rec, replace_with_rec, update_rec,
};
use std::iter::FromIterator;
use std::marker::PhantomData;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InsertError {
    EntryExists,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RemoveError {
    KeyNotFound,
    ValueNotMatching,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpdateError<T> {
    KeyNotFound,
    ValueCallbackError(T),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplaceError {
    KeyNotFound,
}

/// Mutable HAMT data structure
///
/// This is created after a thaw of an immutable HAMT,
/// and right after the creation, all the node are
/// still shared with the immutable HAMT.
///
/// Once modification happens, then each nodes from
/// root to leaf will be modified and kept in an
/// efficient mutable format until freezing.
pub struct HamtMut<K, V, H> {
    root: Hamt<K, V, H>,
}

impl<H, K, V> Clone for HamtMut<K, V, H> {
    fn clone(&self) -> Self {
        Self {
            root: self.root.clone(),
        }
    }
}

impl<'a, H, K, V> From<&'a Hamt<K, V, H>> for HamtMut<K, V, H> {
    fn from(t: &'a Hamt<K, V, H>) -> Self {
        HamtMut { root: t.clone() }
    }
}

impl<H: Hasher + Default, K: Clone + Eq + Hash, V: Clone> HamtMut<K, V, H> {
    /// Create a new empty mutable HAMT
    pub fn new() -> Self {
        HamtMut { root: Hamt::new() }
    }
}

impl<H, K, V> HamtMut<K, V, H> {
    /// Freeze the mutable HAMT back into an immutable HAMT
    ///
    /// This recursively freeze all the mutable nodes
    pub fn freeze(self) -> Hamt<K, V, H> {
        self.root
    }
}

impl<H: Hasher + Default, K: Clone + Eq + Hash, V: Clone> HamtMut<K, V, H> {
    /// Insert a new key into the HAMT
    ///
    /// If the key already exists, then an InsertError is returned.
    ///
    /// To simulaneously manipulate a key, either to insert or update, use 'insert_or_update'
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
    /// Remove a key from the HAMT, if it also matching the value V
    ///
    /// If the key doesn't exist, then RemoveError::KeyNotFound will be returned
    /// and otherwise if the key exists but the value doesn't match, RemoveError::ValueNotMatching
    /// will be returned.
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
    /// Remove a key from the HAMT
    ///
    /// If the key doesn't exist, then RemoveError::KeyNotFound will be returned
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
    /// Replace the element at the key by the v and return the new tree
    /// and the old value.
    pub fn replace(&mut self, k: &K, v: V) -> Result<V, ReplaceError> {
        let h = HashedKey::compute(self.root.hasher, &k);
        let (newroot, oldv) = replace_rec(&self.root.root, h, 0, k, v)?;
        self.root = Hamt {
            root: newroot,
            hasher: PhantomData,
        };
        Ok(oldv)
    }

    /// Replace the element at the key by the v and return the new tree
    /// and the old value.
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
    /// Update the element at the key K.
    ///
    /// If the closure F in parameter returns None, then the key is deleted.
    ///
    /// If the key is not present then UpdateError::KeyNotFound is returned
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

    /// Update or insert the element at the key K
    ///
    /// If the element is not present, then V is added, otherwise the closure F is apply
    /// to the found element. If the closure returns None, then the key is deleted
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

    /// Update or insert the element at the key K
    ///
    /// If the element is not present, then V is added, otherwise the closure F is apply
    /// to the found element. If the closure returns None, then the key is deleted.
    ///
    /// This is similar to 'insert_or_update' except the closure shouldn't be failing
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
