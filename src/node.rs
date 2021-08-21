//! Node in the HAMT
//!
//! Node has the following representation:
//! ```text
//!
//!   bitmap (32 bits)   boxed array of Entry (size = number of bit set in bitmap)
//! ,----------------.  ,-----------------------.
//! 0001000..000010000  | Arc | Arc | ... | Arc |
//!                         |     \
//!                         |      `----- SubNode(Node)
//!                         |
//!                       Leaf(HashedKey, K1, V1)
//! ```
//!

use super::bitmap::{ArrayIndex, SmallBitmap};
use super::collision::Collision;
use super::hash::{HashedKey, LevelIndex};
use super::mutable::*;

use std::slice;
use std::sync::Arc;

/// Node of the Hash Array Mapped Trie
///
/// The bitmap is indexed by a part of the Hash
/// and give an entry
pub struct Node<K, V> {
    pub bitmap: SmallBitmap,
    pub children: Box<[Arc<Entry<K, V>>]>,
}

impl<K, V> Clone for Node<K, V> {
    fn clone(&self) -> Self {
        Self {
            bitmap: self.bitmap,
            children: self.children.clone(),
        }
    }
}

pub type NodeIter<'a, K, V> = slice::Iter<'a, Arc<Entry<K, V>>>;

pub enum Entry<K, V> {
    Leaf(HashedKey, K, V),
    LeafMany(HashedKey, Collision<K, V>),
    SubNode(Node<K, V>),
}

impl<K, V> Node<K, V> {
    pub fn new() -> Self {
        Node {
            bitmap: SmallBitmap::new(),
            children: Vec::with_capacity(0).into(),
        }
    }

    pub fn singleton(idx: LevelIndex, child: Arc<Entry<K, V>>) -> Self {
        Node {
            bitmap: SmallBitmap::once(idx),
            children: vec![child].into(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.bitmap.is_empty()
    }

    pub fn number_children(&self) -> usize {
        self.bitmap.present()
    }

    pub fn get_child(&self, at: ArrayIndex) -> &Arc<Entry<K, V>> {
        assert_eq!(at.is_not_found(), false);
        &self.children[at.get_found()]
    }

    pub fn set_at(&self, idx: LevelIndex, child: Arc<Entry<K, V>>) -> Self {
        assert_eq!(self.bitmap.is_set(idx), false);
        let pos = self.bitmap.get_sparse_pos(idx);

        let mut new_array = Vec::with_capacity(self.children.len() + 1);
        new_array.extend_from_slice(&self.children[0..pos.get_found()]);
        new_array.push(child);
        new_array.extend_from_slice(&self.children[pos.get_found()..]);

        Node {
            bitmap: self.bitmap.set_index(idx),
            children: new_array.into(),
        }
    }

    pub fn clear_at(&self, idx: LevelIndex) -> Option<Self> {
        assert_eq!(self.bitmap.is_set(idx), true);
        let new_bitmap = self.bitmap.clear_index(idx);
        if new_bitmap.is_empty() {
            None
        } else {
            // use the old bitmap to locate the element
            let pos = self.bitmap.get_sparse_pos(idx);

            let mut v: Vec<_> = self.children.to_vec();
            v.remove(pos.get_found());

            Some(Node {
                bitmap: new_bitmap,
                children: v.into(),
            })
        }
    }

    pub fn replace_at(&self, idx: ArrayIndex, child: Arc<Entry<K, V>>) -> Self {
        // with the raw index should have:
        // assert_eq!(self.bitmap.is_set(idx), true);

        let mut v = self.children.clone();
        v[idx.get_found()] = child;

        Node {
            bitmap: self.bitmap,
            children: v,
        }
    }

    pub fn clear_or_replace_at(
        &self,
        idx: LevelIndex,
        child: Option<Arc<Entry<K, V>>>,
    ) -> Option<Self> {
        assert_eq!(self.bitmap.is_set(idx), true);
        match child {
            None => self.clear_at(idx),
            Some(v) => {
                let aidx = self.bitmap.get_index_sparse(idx);
                Some(self.replace_at(aidx, v))
            }
        }
    }

    pub fn iter(&self) -> NodeIter<K, V> {
        self.children.iter()
    }
}

// Insert leaf recursively, settings parents node back to cope with the change
//
// this is guaranteed by the trie design not to recurse forever, because at some
// point the hashedkey value being shifted by level_index will match to 0,
// creating Leaf and Collision node instead of Subnode.
pub fn insert_rec<K: Clone + PartialEq, V: Clone>(
    node: &Node<K, V>,
    hash: HashedKey,
    lvl: usize,
    key: K,
    value: V,
) -> Result<Node<K, V>, InsertError> {
    let level_hash = hash.level_index(lvl);
    let idx = node.bitmap.get_index_sparse(level_hash);
    if idx.is_not_found() {
        let e = Arc::new(Entry::Leaf(hash, key, value));
        Ok(node.set_at(level_hash, e))
    } else {
        match &(node.get_child(idx)).as_ref() {
            Entry::Leaf(lh, lk, lv) => {
                // in case of same hash, then we append to the collision type
                // otherwise we create a new subnode
                if *lh == hash {
                    if lk == &key {
                        return Err(InsertError::EntryExists);
                    }
                    let dat = vec![(lk.clone(), lv.clone()), (key, value)];
                    let e = Arc::new(Entry::LeafMany(*lh, Collision::from_vec(dat)));
                    Ok(node.replace_at(idx, e))
                } else {
                    let leaf_idx = lh.level_index(lvl + 1);
                    let entry_next_idx = hash.level_index(lvl + 1);
                    let subnode = Node::singleton(leaf_idx, Arc::clone(node.get_child(idx)));

                    if entry_next_idx != leaf_idx {
                        let subnode =
                            subnode.set_at(entry_next_idx, Arc::new(Entry::Leaf(hash, key, value)));
                        Ok(node.replace_at(idx, Arc::new(Entry::SubNode(subnode))))
                    } else {
                        let r = insert_rec(&subnode, hash, lvl + 1, key, value)?;
                        let e = Arc::new(Entry::SubNode(r));
                        Ok(node.replace_at(idx, e))
                    }
                }
            }
            Entry::LeafMany(lh, col) => {
                assert_eq!(*lh, hash);
                let col = col.insert(key, value)?;
                Ok(node.replace_at(idx, Arc::new(Entry::LeafMany(*lh, col))))
            }
            Entry::SubNode(sub) => {
                if lvl > 13 {
                    // this is to appease the compiler for now, but globally an impossible
                    // state.
                    unreachable!()
                } else {
                    let r = insert_rec(sub, hash, lvl + 1, key, value)?;
                    let e = Arc::new(Entry::SubNode(r));
                    Ok(node.replace_at(idx, e))
                }
            }
        }
    }
}

pub enum LookupRet<'a, K, V> {
    Found(&'a V),
    NotFound,
    ContinueIn(&'a Node<K, V>),
}

pub fn lookup_one<'a, K: PartialEq, V>(
    node: &'a Node<K, V>,
    h: &HashedKey,
    lvl: usize,
    k: &K,
) -> LookupRet<'a, K, V> {
    let level_hash = h.level_index(lvl);
    let idx = node.bitmap.get_index_sparse(level_hash);
    if idx.is_not_found() {
        LookupRet::NotFound
    } else {
        match &(node.get_child(idx)).as_ref() {
            Entry::Leaf(lh, lk, lv) => {
                if lh == h && lk == k {
                    LookupRet::Found(lv)
                } else {
                    LookupRet::NotFound
                }
            }
            Entry::LeafMany(lh, col) => {
                if lh != h {
                    LookupRet::NotFound
                } else {
                    match col.0.iter().find(|(lk, _)| lk == k) {
                        None => LookupRet::NotFound,
                        Some(lkv) => LookupRet::Found(&lkv.1),
                    }
                }
            }
            Entry::SubNode(sub) => LookupRet::ContinueIn(sub),
        }
    }
}

// recursively try to remove a key with an expected equality value v
pub fn remove_eq_rec<K: PartialEq + Clone, V: PartialEq + Clone>(
    node: &Node<K, V>,
    h: HashedKey,
    lvl: usize,
    k: &K,
    v: &V,
) -> Result<Option<Node<K, V>>, RemoveError> {
    let level_hash = h.level_index(lvl);
    let idx = node.bitmap.get_index_sparse(level_hash);
    if idx.is_not_found() {
        Err(RemoveError::KeyNotFound)
    } else {
        match &(node.get_child(idx)).as_ref() {
            Entry::Leaf(lh, lk, lv) => {
                if *lh == h && lk == k {
                    if lv == v {
                        Ok(node.clear_at(level_hash))
                    } else {
                        Err(RemoveError::ValueNotMatching)
                    }
                } else {
                    Err(RemoveError::KeyNotFound)
                }
            }
            Entry::LeafMany(lh, col) => {
                assert_eq!(*lh, h);
                let replacement = col.remove_match(h, k, v)?;
                Ok(Some(node.replace_at(idx, Arc::new(replacement))))
            }
            Entry::SubNode(sub) => match remove_eq_rec(sub, h, lvl + 1, k, v)? {
                None => Ok(node.clear_at(level_hash)),
                Some(newsub) => {
                    let e = Entry::SubNode(newsub);
                    Ok(Some(node.replace_at(idx, Arc::new(e))))
                }
            },
        }
    }
}

// recursively try to remove a key
pub fn remove_rec<K: Clone + PartialEq, V: Clone>(
    node: &Node<K, V>,
    h: HashedKey,
    lvl: usize,
    k: &K,
) -> Result<Option<Node<K, V>>, RemoveError> {
    let level_hash = h.level_index(lvl);
    let idx = node.bitmap.get_index_sparse(level_hash);
    if idx.is_not_found() {
        Err(RemoveError::KeyNotFound)
    } else {
        match &(node.get_child(idx)).as_ref() {
            Entry::Leaf(lh, lk, _) => {
                if *lh == h && lk == k {
                    Ok(node.clear_at(level_hash))
                } else {
                    Err(RemoveError::KeyNotFound)
                }
            }
            Entry::LeafMany(lh, col) => {
                assert_eq!(*lh, h);
                let replacement = col.remove(h, k)?;
                Ok(Some(node.replace_at(idx, Arc::new(replacement))))
            }
            Entry::SubNode(sub) => match remove_rec(sub, h, lvl + 1, k)? {
                None => Ok(node.clear_at(level_hash)),
                Some(newsub) => {
                    let e = Entry::SubNode(newsub);
                    Ok(Some(node.replace_at(idx, Arc::new(e))))
                }
            },
        }
    }
}

// recursively try to update a key.
//
// note, an update cannot create a new value, it can only delete or update an existing value.
pub fn update_rec<K: PartialEq + Clone, V: Clone, F, U>(
    node: &Node<K, V>,
    h: HashedKey,
    lvl: usize,
    k: &K,
    f: F,
) -> Result<Option<Node<K, V>>, UpdateError<U>>
where
    F: FnOnce(&V) -> Result<Option<V>, U>,
{
    let level_hash = h.level_index(lvl);
    let idx = node.bitmap.get_index_sparse(level_hash);
    if idx.is_not_found() {
        Err(UpdateError::KeyNotFound)
    } else {
        match &(node.get_child(idx)).as_ref() {
            Entry::Leaf(lh, lk, lv) => {
                if *lh == h && lk == k {
                    let newv = f(lv).map_err(UpdateError::ValueCallbackError)?;
                    Ok(node.clear_or_replace_at(
                        level_hash,
                        newv.map(|x| Arc::new(Entry::Leaf(*lh, lk.clone(), x))),
                    ))
                } else {
                    Err(UpdateError::KeyNotFound)
                }
            }
            Entry::LeafMany(lh, col) => {
                assert_eq!(*lh, h);
                let replacement = col.update(h, k, f)?;
                Ok(Some(node.replace_at(idx, Arc::new(replacement))))
            }
            Entry::SubNode(sub) => match update_rec(sub, h, lvl + 1, k, f)? {
                None => Ok(node.clear_at(level_hash)),
                Some(newsub) => {
                    let e = Entry::SubNode(newsub);
                    Ok(Some(node.replace_at(idx, Arc::new(e))))
                }
            },
        }
    }
}

// recursively try to replace a key's value.
pub fn replace_rec<K: PartialEq + Clone, V: Clone>(
    node: &Node<K, V>,
    h: HashedKey,
    lvl: usize,
    k: &K,
    v: V,
) -> Result<(Node<K, V>, V), ReplaceError> {
    let level_hash = h.level_index(lvl);
    let idx = node.bitmap.get_index_sparse(level_hash);
    if idx.is_not_found() {
        Err(ReplaceError::KeyNotFound)
    } else {
        match &(node.get_child(idx)).as_ref() {
            Entry::Leaf(lh, lk, lv) => {
                if *lh == h && lk == k {
                    let new_ent = Arc::new(Entry::Leaf(*lh, lk.clone(), v));
                    Ok((node.replace_at(idx, new_ent), lv.clone()))
                } else {
                    Err(ReplaceError::KeyNotFound)
                }
            }
            Entry::LeafMany(lh, col) => {
                assert_eq!(*lh, h);
                let (replacement, old_value) = col.replace(k, v)?;
                Ok((
                    node.replace_at(idx, Arc::new(Entry::LeafMany(*lh, replacement))),
                    old_value,
                ))
            }
            Entry::SubNode(sub) => {
                let (newsub, oldv) = replace_rec(sub, h, lvl + 1, k, v)?;
                let e = Entry::SubNode(newsub);
                Ok((node.replace_at(idx, Arc::new(e)), oldv))
            }
        }
    }
}

// recursively try to replace a key's value.
pub fn replace_with_rec<K: PartialEq + Clone, V: Clone, F>(
    node: &Node<K, V>,
    h: HashedKey,
    lvl: usize,
    k: &K,
    f: F,
) -> Result<Node<K, V>, ReplaceError>
where
    F: FnOnce(&V) -> V,
{
    let level_hash = h.level_index(lvl);
    let idx = node.bitmap.get_index_sparse(level_hash);
    if idx.is_not_found() {
        Err(ReplaceError::KeyNotFound)
    } else {
        match &(node.get_child(idx)).as_ref() {
            Entry::Leaf(lh, lk, lv) => {
                if *lh == h && lk == k {
                    let new_ent = Arc::new(Entry::Leaf(*lh, lk.clone(), f(lv)));
                    Ok(node.replace_at(idx, new_ent))
                } else {
                    Err(ReplaceError::KeyNotFound)
                }
            }
            Entry::LeafMany(lh, col) => {
                assert_eq!(*lh, h);
                let replacement = col.replace_with(k, f)?;
                Ok(node.replace_at(idx, Arc::new(Entry::LeafMany(*lh, replacement))))
            }
            Entry::SubNode(sub) => {
                let newsub = replace_with_rec(sub, h, lvl + 1, k, f)?;
                let e = Entry::SubNode(newsub);
                Ok(node.replace_at(idx, Arc::new(e)))
            }
        }
    }
}

pub fn size_rec<K, V>(node: &Node<K, V>) -> usize {
    let mut sum = 0;
    for c in node.children.iter() {
        match &c.as_ref() {
            Entry::Leaf(_, _, _) => sum += 1,
            Entry::LeafMany(_, col) => sum += col.len(),
            Entry::SubNode(sub) => sum += size_rec(sub),
        }
    }
    sum
}

// debug module
pub mod debug {
    use super::*;
    use std::cmp;

    pub fn depth_rec<K, V>(node: &Node<K, V>) -> usize {
        let mut max_depth = 0;
        for c in node.children.iter() {
            match &c.as_ref() {
                Entry::Leaf(_, _, _) => {}
                Entry::LeafMany(_, _) => {}
                Entry::SubNode(sub) => {
                    let child_depth = depth_rec(sub);
                    max_depth = cmp::max(max_depth, child_depth)
                }
            }
        }
        max_depth
    }
}
