pub use super::immutable::*;
pub use super::mutable::*;

/// Construct a HAMT from a sequence of key/value pairs using a hashtable like syntax.
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
/// # #[macro_use] extern crate hamst;
/// # use hamst::Hamt;
/// # use std::iter::FromIterator;
/// # use std::collections::hash_map::DefaultHasher;
/// # fn main() {
/// let reference : Hamt<u32, u32, DefaultHasher> = Hamt::from_iter(vec![(1, 11), (2, 22), (3, 33)].into_iter());
/// assert!(hamt!{ 1 => 11, 2 => 22, 3 => 33 } == reference);
/// # }
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
