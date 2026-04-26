use super::hash::{Hash, HashedKey, Hasher};
use super::mutable::HamtMut;
use super::node::{Entry, LookupRet, Node, NodeIter, lookup_one, size_rec};
use std::iter::FromIterator;
use std::marker::PhantomData;
use std::mem::swap;
use std::slice;

/// Hash Array Mapped Trie — an immutable map with structural sharing.
///
/// `Hamt<K, V, H>` is a read-only map keyed by `K`, valued by `V`, hashed
/// by `H`. The structure is immutable from root to leaves: every "mutation"
/// is conceptually a function from one [`Hamt`] to another, and the new tree
/// shares all unmodified subtrees with the old one via [`std::sync::Arc`].
///
/// # Structural sharing
///
/// Cloning a [`Hamt`] is **O(1)** — only the root [`std::sync::Arc`] is bumped.
/// All previously-observable [`Hamt`] values remain reachable and unchanged
/// after a "mutation" routes through [`Hamt::thaw`] / [`HamtMut::freeze`] or
/// [`Hamt::mutate_freeze`]; the typical persistent-data-structure pattern.
///
/// # Thread safety
///
/// [`Hamt`] is `Send + Sync` whenever `K`, `V`, and `H` are. Hand a `Hamt`
/// across threads freely. The mutation-staging counterpart [`HamtMut`] is
/// **not `Send`** — see its type-level documentation.
///
/// # Examples
///
/// ```
/// # use hamst::{Hamt, hamt};
/// let m: Hamt<u32, u32> = hamt!{ 1 => 10, 2 => 20 };
/// assert_eq!(m.lookup(&1), Some(&10));
/// assert_eq!(m.size(), 2);
/// ```
pub struct Hamt<K, V, H = std::collections::hash_map::DefaultHasher> {
    pub(crate) root: Node<K, V>,
    pub(crate) hasher: PhantomData<H>,
}

/// Cloning is O(1) — internal nodes are [`std::sync::Arc`]-shared. Prior
/// versions remain reachable.
impl<H, K, V> Clone for Hamt<K, V, H> {
    fn clone(&self) -> Self {
        Hamt {
            root: self.root.clone(),
            hasher: self.hasher,
        }
    }
}

/// Iterator over the `(&K, &V)` pairs of a [`Hamt`].
///
/// Created by [`Hamt::iter`]. Iteration order is **unspecified** — it follows
/// the underlying trie layout, which depends on hash output and is not stable
/// across Rust versions. Collect into [`std::collections::HashSet`] (or
/// compare via [`Hamt::lookup`]) instead of asserting on the iterator's
/// emission order.
pub struct HamtIter<'a, K, V> {
    stack: Vec<NodeIter<'a, K, V>>,
    content: Option<slice::Iter<'a, (K, V)>>,
}

/// Returns an empty [`Hamt`] (equivalent to [`Hamt::new`]).
impl<H: Hasher + Default, K: Eq + Hash, V> Default for Hamt<K, V, H> {
    fn default() -> Self {
        Self::new()
    }
}

impl<H: Hasher + Default, K: PartialEq + Eq + Hash, V> Hamt<K, V, H> {
    /// Create a new empty HAMT.
    pub fn new() -> Self {
        Hamt {
            root: Node::new(),
            hasher: PhantomData,
        }
    }

    /// Return `true` if this HAMT contains no entries.
    pub fn is_empty(&self) -> bool {
        self.root.is_empty()
    }

    /// Return the number of elements in this HAMT.
    ///
    /// This walks the entire trie and is **O(n)** in the number of entries.
    /// It is not a cached count — callers that call `size` in a hot loop
    /// should cache the result in a local variable.
    ///
    /// # Examples
    ///
    /// ```
    /// # use hamst::Hamt;
    /// let m: Hamt<u32, u32> = Hamt::new();
    /// assert_eq!(m.size(), 0);
    /// ```
    pub fn size(&self) -> usize {
        size_rec(&self.root)
    }

    /// Thaw this [`Hamt`] into a [`HamtMut`] mutation-staging workspace.
    ///
    /// The thawed [`HamtMut`] starts as a structural-share clone of `self`
    /// — no nodes are copied up-front. Subsequent mutations on the
    /// [`HamtMut`] copy-on-write only the path from root to modified leaf.
    /// Call [`HamtMut::freeze`] (or use [`Hamt::mutate_freeze`]) to return
    /// to the immutable world.
    ///
    /// # Examples
    ///
    /// ```
    /// # use hamst::{Hamt, HamtMut};
    /// let m: Hamt<u32, &str> = Hamt::new();
    /// let mut staging = m.thaw();
    /// staging.insert(1, "a").unwrap();
    /// let m2: Hamt<u32, &str> = staging.freeze();
    /// assert_eq!(m2.lookup(&1), Some(&"a"));
    /// ```
    pub fn thaw(&self) -> HamtMut<K, V, H> {
        self.into()
    }

    /// Temporarily thaw `self`, apply `f` to the mutable view, and freeze
    /// the result.
    ///
    /// This is the canonical one-shot mutation API — it composes [`Hamt::thaw`],
    /// the closure `f`, and [`HamtMut::freeze`] into a single call. The original
    /// `self` is untouched (this is a persistent data structure); the returned
    /// [`Hamt`] is a new tree that shares unmodified subtrees with `self`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use hamst::{Hamt, HamtMut};
    /// let empty: Hamt<u32, u32> = Hamt::new();
    /// let result = empty.mutate_freeze(|x| {
    ///     x.insert(10, 20)?;
    ///     x.insert(20, 30)
    /// }).expect("inserts succeed");
    /// assert_eq!(result.lookup(&10), Some(&20));
    /// ```
    ///
    /// # Errors
    ///
    /// Propagates any error `E` returned by the closure `f`. In practice `E`
    /// is one of [`crate::InsertError`], [`crate::RemoveError`],
    /// [`crate::UpdateError`], [`crate::ReplaceError`], or any caller-defined
    /// error type that the closure surfaces via `?`.
    pub fn mutate_freeze<E, F>(&self, f: F) -> Result<Self, E>
    where
        F: FnOnce(&mut HamtMut<K, V, H>) -> Result<(), E>,
    {
        let mut x = self.thaw();
        f(&mut x)?;
        Ok(x.freeze())
    }

    /// Like [`Hamt::mutate_freeze`], but also returns a value computed by `f`.
    ///
    /// Useful when the closure needs to return the previously-stored value of
    /// a key (for example, [`HamtMut::replace`] returns the old value). The
    /// first element of the returned tuple is the new [`Hamt`]; the second
    /// is whatever the closure produced.
    ///
    /// # Examples
    ///
    /// ```
    /// # use hamst::{Hamt, HamtMut};
    /// let m: Hamt<u32, u32> = Hamt::new()
    ///     .mutate_freeze(|x| x.insert(1, 10))
    ///     .unwrap();
    /// let (m2, old) = m.mutate_freeze_ret(|x| x.replace(&1, 99)).unwrap();
    /// assert_eq!(old, 10);
    /// assert_eq!(m2.lookup(&1), Some(&99));
    /// ```
    ///
    /// # Errors
    ///
    /// Propagates any error `E` returned by the closure `f`. See
    /// [`Hamt::mutate_freeze`] for the typical error types.
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
    /// Look up the value associated with key `k`, returning `None` if absent.
    ///
    /// Lookup is **O(log_32 n)** in the number of entries — the trie branches
    /// 32 ways at each level, so even very large maps walk at most a handful
    /// of nodes before reaching a leaf.
    ///
    /// # Examples
    ///
    /// ```
    /// # use hamst::{Hamt, hamt};
    /// let m: Hamt<u32, &str> = hamt!{ 1 => "one", 2 => "two" };
    /// assert_eq!(m.lookup(&1), Some(&"one"));
    /// assert_eq!(m.lookup(&3), None);
    /// ```
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
                    n = subnode;
                }
            }
        }
    }

    /// Return `true` if `k` is present. Equivalent to
    /// `self.lookup(k).is_some()`.
    pub fn contains_key(&self, k: &K) -> bool {
        self.lookup(k).map_or_else(|| false, |_| true)
    }

    /// Return an iterator over the `(&K, &V)` pairs in this HAMT.
    ///
    /// Iteration order is **unspecified** and depends on hash output; do not
    /// rely on insertion order or key order.
    ///
    /// # Examples
    ///
    /// ```
    /// # use std::collections::HashSet;
    /// # use hamst::{Hamt, hamt};
    /// let m: Hamt<u32, u32> = hamt!{ 1 => 10, 2 => 20 };
    /// let got: HashSet<(u32, u32)> = m.iter().map(|(k, v)| (*k, *v)).collect();
    /// assert_eq!(got, HashSet::from([(1, 10), (2, 20)]));
    /// ```
    pub fn iter<'a>(&'a self) -> HamtIter<'a, K, V> {
        HamtIter {
            stack: vec![self.root.iter()],
            content: None,
        }
    }
}

/// Iterates `(&K, &V)` pairs in unspecified order. See [`Hamt::iter`].
impl<'a, K, V> Iterator for HamtIter<'a, K, V> {
    type Item = (&'a K, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let mut x = None;
            swap(&mut self.content, &mut x);
            match x {
                Some(mut iter) => match iter.next() {
                    None => self.content = None,
                    Some(o) => {
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
                            Entry::SubNode(sub) => self.stack.push(sub.iter()),
                            Entry::Leaf(_, k, v) => return Some((k, v)),
                            Entry::LeafMany(_, col) => self.content = Some(col.iter()),
                        },
                    },
                },
            }
        }
    }
}

/// Build a [`Hamt`] from an iterator of `(K, V)` pairs.
///
/// Duplicate keys: first insert wins (subsequent inserts of the same key are
/// silently dropped, since the underlying [`HamtMut::insert`] returns
/// [`crate::InsertError::EntryExists`] which `FromIterator` discards).
impl<H: Default + Hasher, K: Eq + Hash + Clone, V: Clone> FromIterator<(K, V)> for Hamt<K, V, H> {
    fn from_iter<I: IntoIterator<Item = (K, V)>>(iter: I) -> Self {
        HamtMut::from_iter(iter).freeze()
    }
}

/// Equality is content-based: two maps are equal iff they contain the same
/// `(K, V)` pairs. The current implementation iterates one side and looks up
/// each key on the other, so worst-case complexity is O(n · log₃₂ n).
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
