HAMST - Hash Array Mapped Shareable Trie
========================================

An [HAMT](https://en.wikipedia.org/wiki/Hash_array_mapped_trie) data structure in rust,
that uses immutable nodes that are shareable between new copies.

Modification in this HAMT, only create new nodes and leave the previous nodes available for
any remaining old copies. Once the root of a copy disappear, then all the nodes that
are unique to this copy, are garbage collected also.
