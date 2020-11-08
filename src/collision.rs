use super::hash::HashedKey;
use super::mutable::{InsertError, RemoveError, ReplaceError, UpdateError};
use super::node::Entry;

use std::slice;

pub struct Collision<K, V>(pub Box<Box<[(K, V)]>>);

impl<K, V> Collision<K, V> {
    pub fn from_vec(vec: Vec<(K, V)>) -> Self {
        assert!(vec.len() >= 2);
        Collision(Box::new(vec.into()))
    }
    pub fn from_box(b: Box<[(K, V)]>) -> Self {
        assert!(b.len() >= 2);
        Collision(Box::new(b))
    }
    pub fn len(&self) -> usize {
        self.0.len()
    }
    pub fn iter(&self) -> slice::Iter<'_, (K, V)> {
        self.0.iter()
    }
}

impl<K: Clone + PartialEq, V: Clone> Collision<K, V> {
    pub fn insert(&self, k: K, v: V) -> Result<Self, InsertError> {
        if self.0.iter().any(|(lk, _)| lk == &k) {
            Err(InsertError::EntryExists)
        } else {
            let mut new_array = Vec::with_capacity(self.0.len() + 1);
            new_array.extend_from_slice(&self.0[..]);
            new_array.push((k, v));
            Ok(Collision::from_box(new_array.into()))
        }
    }

    fn get_record_and_pos(&self, k: &K) -> Option<(usize, &(K, V))> {
        self.0.iter().enumerate().find(|(_, (fk, _))| fk == k)
    }

    pub fn remove(&self, h: HashedKey, k: &K) -> Result<Entry<K, V>, RemoveError> {
        let (pos, _) = self.get_record_and_pos(k).ok_or(RemoveError::KeyNotFound)?;
        if self.0.len() == 2 {
            let to_keep = if pos == 0 { &self.0[1] } else { &self.0[0] };
            Ok(Entry::Leaf(h, to_keep.0.clone(), to_keep.1.clone()))
        } else {
            let col = Collision::from_box(helper::clone_array_and_remove_at_pos(&self.0, pos));
            Ok(Entry::LeafMany(h, col))
        }
    }

    pub fn remove_match(&self, h: HashedKey, k: &K, v: &V) -> Result<Entry<K, V>, RemoveError>
    where
        V: PartialEq,
    {
        let (pos, _) = self.get_record_and_pos(k).ok_or(RemoveError::KeyNotFound)?;
        if &self.0[pos].1 != v {
            Err(RemoveError::ValueNotMatching)
        } else if self.0.len() == 2 {
            let to_keep = if pos == 0 { &self.0[1] } else { &self.0[0] };
            Ok(Entry::Leaf(h, to_keep.0.clone(), to_keep.1.clone()))
        } else {
            let col = Collision::from_box(helper::clone_array_and_remove_at_pos(&self.0, pos));
            Ok(Entry::LeafMany(h, col))
        }
    }

    pub fn update<F, U>(&self, h: HashedKey, k: &K, f: F) -> Result<Entry<K, V>, UpdateError<U>>
    where
        F: FnOnce(&V) -> Result<Option<V>, U>,
    {
        let (pos, (_, v)) = self.get_record_and_pos(k).ok_or(UpdateError::KeyNotFound)?;
        match f(v).map_err(UpdateError::ValueCallbackError)? {
            None => {
                if self.0.len() == 2 {
                    let to_keep = if pos == 0 { &self.0[1] } else { &self.0[0] };
                    Ok(Entry::Leaf(h, to_keep.0.clone(), to_keep.1.clone()))
                } else {
                    let col =
                        Collision::from_box(helper::clone_array_and_remove_at_pos(&self.0, pos));
                    Ok(Entry::LeafMany(h, col))
                }
            }
            Some(newv) => {
                let newcol = Collision::from_box(helper::clone_array_and_set_at_pos(
                    &self.0,
                    (k.clone(), newv),
                    pos,
                ));
                Ok(Entry::LeafMany(h, newcol))
            }
        }
    }

    pub fn replace(&self, k: &K, v: V) -> Result<(Self, V), ReplaceError> {
        let (pos, (_, oldv)) = self
            .get_record_and_pos(k)
            .ok_or(ReplaceError::KeyNotFound)?;
        let newcol = Collision::from_box(helper::clone_array_and_set_at_pos(
            &self.0,
            (k.clone(), v),
            pos,
        ));
        Ok((newcol, oldv.clone()))
    }

    pub fn replace_with<F>(&self, k: &K, f: F) -> Result<Self, ReplaceError>
    where
        F: FnOnce(&V) -> V,
    {
        let (pos, (_, oldv)) = self
            .get_record_and_pos(k)
            .ok_or(ReplaceError::KeyNotFound)?;
        let v = f(oldv);
        let newcol = Collision::from_box(helper::clone_array_and_set_at_pos(
            &self.0,
            (k.clone(), v),
            pos,
        ));
        Ok(newcol)
    }
}

mod helper {
    #[inline]
    pub fn clone_array_and_set_at_pos<A: Clone>(v: &[A], a: A, pos: usize) -> Box<[A]> {
        // copy all elements except at pos where a replaces it.
        let mut new_array: Vec<A> = Vec::with_capacity(v.len());
        if pos > 0 {
            new_array.extend_from_slice(&v[0..pos]);
        }
        new_array.push(a);
        if pos + 1 < v.len() {
            new_array.extend_from_slice(&v[(pos + 1)..]);
        }
        new_array.into()
    }

    #[inline]
    pub fn clone_array_and_remove_at_pos<A: Clone>(v: &[A], pos: usize) -> Box<[A]> {
        let mut v: Vec<_> = v.to_vec();
        v.remove(pos);
        v.into()
    }
}
