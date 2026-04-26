//! Public API re-exports and the [`hamt!`] macro.
//!
//! See the crate-level documentation for an overview of HAMT, thaw/freeze,
//! and structural sharing.

pub use super::immutable::*;
pub use super::mutable::*;

/// Construct a [`Hamt`] from a sequence of key/value pairs using hashtable-like syntax.
///
/// In the case of duplicated keys, the latest value is used.
///
/// ```ignore
/// hamt!{"key" => "value"}
/// ```
///
/// # Examples
///
/// ```
/// # use std::collections::HashSet;
/// # use hamst::{hamt, Hamt};
/// let m: Hamt<u32, u32> = hamt!{ 1 => 11, 2 => 22, 3 => 33 };
/// let got: HashSet<(u32, u32)> = m.iter().map(|(k, v)| (*k, *v)).collect();
/// assert_eq!(got, HashSet::from([(1, 11), (2, 22), (3, 33)]));
/// ```
#[macro_export]
macro_rules! hamt {
    () => { $crate::Hamt::new() };

    ( $( $key:expr => $value:expr ),* ) => {{
        let mut h = $crate::HamtMut::new();
        $({
            h.insert_or_update_simple($key, $value, |_| Some($value));
        })*;
        h.freeze()
    }};

    ( $( $key:expr => $value:expr ,)* ) => {{
        let mut h = $crate::HamtMut::new();
        $({
            h.insert_or_update_simple($key, $value, |_| Some($value));
        })*;
        h.freeze()
    }};
}
