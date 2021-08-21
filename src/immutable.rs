use super::hash::{Hash, HashedKey, Hasher};
use super::mutable::HamtMut;
pub use super::mutable::{InsertError, RemoveError, ReplaceError, UpdateError};
use super::node::{lookup_one, size_rec, Entry, LookupRet, Node, NodeIter};
use std::iter::FromIterator;
use std::marker::PhantomData;
use std::mem::swap;
use std::slice;

/// HAMT with shareable nodes
///
/// The structure is immutable from root to leaves, where
/// each node of this structure is also independently
/// shareable, including on different thread.
pub struct Hamt<K, V, H = std::collections::hash_map::DefaultHasher> {
    pub(crate) root: Node<K, V>,
    pub(crate) hasher: PhantomData<H>,
}

impl<H, K, V> Clone for Hamt<K, V, H> {
    fn clone(&self) -> Self {
        Hamt {
            root: self.root.clone(),
            hasher: self.hasher.clone(),
        }
    }
}

/// HAMT iterator
pub struct HamtIter<'a, K, V> {
    stack: Vec<NodeIter<'a, K, V>>,
    content: Option<slice::Iter<'a, (K, V)>>,
}

impl<H: Hasher + Default, K: Eq + Hash, V> Default for Hamt<K, V, H> {
    fn default() -> Self {
        Self::new()
    }
}

impl<H: Hasher + Default, K: PartialEq + Eq + Hash, V> Hamt<K, V, H> {
    /// Create a new empty HAMT
    pub fn new() -> Self {
        Hamt {
            root: Node::new(),
            hasher: PhantomData,
        }
    }

    /// Check if the HAMT is empty
    pub fn is_empty(&self) -> bool {
        self.root.is_empty()
    }

    /// Return the number of elements in this HAMT
    pub fn size(&self) -> usize {
        size_rec(&self.root)
    }

    /// Thaw this HAMT into a HAMT that can be mutated
    pub fn thaw(&self) -> HamtMut<K, V, H> {
        self.into()
    }

    /// Temporary create a mutable HAMT from an immutable one, and apply
    /// the callback f to the mutable reference, then thaw into the resulting HAMT
    ///
    /// ```
    /// # use hamst::{Hamt, HamtMut};
    /// # use std::collections::hash_map::DefaultHasher;
    /// # fn main() {
    /// let empty : Hamt<u32, u32, DefaultHasher> = Hamt::new();
    /// let result = empty.mutate_freeze(|x| {
    ///     x.insert(10, 20)?;
    ///     x.insert(20, 30)
    /// }).expect("it works");
    /// # }
    /// ```
    pub fn mutate_freeze<E, F>(&self, f: F) -> Result<Self, E>
    where
        F: FnOnce(&mut HamtMut<K, V, H>) -> Result<(), E>,
    {
        let mut x = self.thaw();
        f(&mut x)?;
        Ok(x.freeze())
    }

    /// Similar to 'mutate_freeze' and also returns the result of f too
    pub fn mutate_freeze_ret<E, F, R>(&self, f: F) -> Result<(Self, R), E>
    where
        F: FnOnce(&mut HamtMut<K, V, H>) -> Result<R, E>,
    {
        let mut x = self.thaw();
        let r = f(&mut x)?;
        Ok((x.freeze(), r))
    }
}

impl<H: Hasher + Default, K: Hash + Eq, V> Hamt<K, V, H> {
    /// Try to get the element related to key K
    pub fn lookup(&self, k: &K) -> Option<&V> {
        let h = HashedKey::compute(self.hasher, &k);
        let mut n = &self.root;
        let mut lvl = 0;
        loop {
            match lookup_one(n, &h, lvl, k) {
                LookupRet::NotFound => return None,
                LookupRet::Found(v) => return Some(v),
                LookupRet::ContinueIn(subnode) => {
                    lvl += 1;
                    n = &subnode;
                }
            }
        }
    }
    /// Check if the key is contained into the HAMT
    pub fn contains_key(&self, k: &K) -> bool {
        self.lookup(k).map_or_else(|| false, |_| true)
    }

    /// Create a new iterator for this HAMT
    pub fn iter(&self) -> HamtIter<K, V> {
        HamtIter {
            stack: vec![self.root.iter()],
            content: None,
        }
    }
}

impl<'a, K, V> Iterator for HamtIter<'a, K, V> {
    type Item = (&'a K, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let mut x = None;
            swap(&mut self.content, &mut x);
            match x {
                Some(mut iter) => match iter.next() {
                    None => self.content = None,
                    Some(ref o) => {
                        self.content = Some(iter);
                        return Some((&o.0, &o.1));
                    }
                },
                None => match self.stack.last_mut() {
                    None => return None,
                    Some(last) => match last.next() {
                        None => {
                            self.stack.pop();
                        }
                        Some(next) => match next.as_ref() {
                            Entry::SubNode(ref sub) => self.stack.push(sub.iter()),
                            Entry::Leaf(_, ref k, ref v) => return Some((&k, &v)),
                            Entry::LeafMany(_, ref col) => self.content = Some(col.iter()),
                        },
                    },
                },
            }
        }
    }
}

impl<H: Default + Hasher, K: Eq + Hash + Clone, V: Clone> FromIterator<(K, V)> for Hamt<K, V, H> {
    fn from_iter<I: IntoIterator<Item = (K, V)>>(iter: I) -> Self {
        HamtMut::from_iter(iter).freeze()
    }
}

impl<H: Default + Hasher, K: Eq + Hash, V: PartialEq> PartialEq for Hamt<K, V, H> {
    fn eq(&self, other: &Self) -> bool {
        // optimised the obvious cases first
        if self.is_empty() && other.is_empty() {
            return true;
        }
        if self.is_empty() != other.is_empty() {
            return false;
        }
        if self.root.bitmap != other.root.bitmap {
            return false;
        }
        // then compare key and values
        // TODO : optimise by comparing nodes directly
        for (k, v) in self.iter() {
            if let Some(v2) = other.lookup(k) {
                if v != v2 {
                    return false;
                }
            } else {
                return false;
            }
        }
        true
    }
}

impl<H: Default + Hasher, K: Eq + Hash, V: Eq> Eq for Hamt<K, V, H> {}
