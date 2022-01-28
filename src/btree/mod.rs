use std::alloc::{Allocator, Global};
use std::mem::size_of;

use log::{debug, trace};
use stackvec::StackVec;

use crate::btree::node::InsertError;
use crate::buffer_manager::swip::OptimisticError;
use crate::buffer_manager::{
    buffer_frame::{LatchStrategy, PageGuard},
    swip::Swip,
    BufferManager,
};

use self::node::Node;

mod node;

const HEURISTIC_RESIZE_THRESHOLD: usize = 6;

// The size of this stack limits the depth that a tree can become.
// We need to statically hold this max depth to avoid heap allocations.
// 32 should be more than plenty.
//
// Worse case, 2 keys per node. Nodes are 16kb. Depth of 32 means we
// have 2^32 leaf page -- roughly 68TB of leaf pages. In practice,
// we limit the key size to 2kb, ensuring a minimum fanout of around
// 7. This means worst case is 7^32 * 16kb, or 1.7E19 TBs of space (infinite).
//
// We could reduce the depth to 16, and with a min fanout of 7, have
// ~500PB of leaf data. This would save stack space.
type GuardPath = StackVec<[PageGuard<Node>; 32]>;

/// `BTree` is a core data structure. It presents a concurrent, thread safe, ordered
/// map backed by a provided `BufferManager`.
///
/// `BTree` goes out of its way to not perform heap allocations and to minimize copying
/// data (outside of actually copying bytes into the `BTree`'s pages) in order to
/// maximize performance.
///
/// `BTree`'s are clonable, as all the backing data is stored within the page data which
/// is thread safe. For this reason, `BTree`'s are also Send and Sync, safely being able
/// to be shared across threads without needed `Arc` or other sync primitives wrapping.
#[derive(Clone)]
pub struct BTree<'a> {
    buffer_manager: &'a BufferManager,
    root_page: Swip<Node>,
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum InsertInfo {
    SplitRoot(u64),
    SplitPage(u64),
    ResizedPage(u64)
}

impl<'a> BTree<'a> {
    /// Create a new `BTree` using the backing `BufferManager`. This will immediately
    /// ask the `BufferManager` for an initial page, which will be treated as the root
    /// page for this `BTree`.
    ///
    /// Many `BTree`s can share the same `BufferManager`. For instance, you may want different
    /// `BTree`s for different indexes or tables in a SQL database.
    pub fn new(buffer_manager: &'a BufferManager) -> Self {
        let root_swip: Swip<Node> = buffer_manager.new_page().unwrap();
        let mut node = root_swip
            .coupled_page_guard::<Node>(None, LatchStrategy::Exclusive)
            .expect("infallible");

        node.init_header(
            1,
            root_swip.page_id(),
            0,
            // None,
            true,
            b"",
            b"",
        );

        BTree {
            buffer_manager: buffer_manager,
            root_page: root_swip,
        }
    }

    /// Insert a key-value pair into the `BTree`. On conflict, this will replace
    /// the current value in.
    ///
    /// Potential costs:
    /// 1. It may compact nodes to fit the key-value pair
    /// 2. It may split nodes to fit the key-value pair
    /// 3. [once disk backed] It may require loading nodes from disk
    pub fn insert(&self, key: &[u8], value: &[u8]) -> StackVec<[InsertInfo; 32]> {
        if key.len() == 0 || value.len() == 0 {
            // Err?
            let info: StackVec<[InsertInfo; 32]> = StackVec::default();
            return info;
        }

        'restart: loop {
            let mut info: StackVec<[InsertInfo; 32]> = StackVec::default();
            let (mut node, mut path) = self.search_to_leaf(&key, LatchStrategy::Exclusive);

            #[cfg(debug_assertions)]
            {
                if !node.shares_prefix(key) {
                    let children = path.last().unwrap().all_children();

                    children.windows(2).for_each(|w| {
                        let left = &w[0];
                        let right = &w[1];

                        if left.upper_fence() != right.lower_fence() {
                            println!("Fence mismatch:\nLeft: {:?}\nRight: {:?}", left, right);
                        }
                    });

                    panic!("Key: {:?}\n\nNode: {:?}\n\nPath: {:?}", key, node, path);
                }

                path.windows(2).for_each(|w| {
                    let parent = &w[0];
                    let child = &w[1];
                    debug_assert!(
                        parent.contains(&child),
                        "Parent: {:?}\n  Child: {:?}",
                        parent,
                        child
                    );
                });

                if let Some(parent) = path.last() {
                    debug_assert!(
                        parent.contains(&node),
                        "Parent: {:?}\n  Child: {:?}",
                        parent,
                        node
                    );
                }
            }

            trace!("TRYING BASE INSERT");
            match node.insert(&key, &value) {
                Ok(()) => {
                    return info;
                },
                Err(node::InsertError::InsufficientSpace) => (),
            };

            // We don't have enough space in the node, and therefore need to perform a
            // split or resize.

            if path.len() == 0 {
                trace!("SPLIT ROOT #1");

                let right_swip: Swip<Node> = self.buffer_manager.new_page().unwrap();
                let left_swip: Swip<Node> = self.buffer_manager.new_page().unwrap();

                let mut right = right_swip
                    .coupled_page_guard::<Node>(None, LatchStrategy::Exclusive)
                    .expect("infallible");
                let mut left = left_swip
                    .coupled_page_guard::<Node>(None, LatchStrategy::Exclusive)
                    .expect("infallible");

                // let pivot_key = node.pivot_key_for_split(&key, value.len(), &mut temp);
                let (pivot, _, _) = node.pivot_for_split();
                node.root_split(&mut left, &mut right, pivot);

                if key < left.upper_fence() {
                    debug!("Inserting Left: k:{:?} v:({})", key, value.len());
                    left.insert(key, value).expect("Space to now exist");
                } else {
                    debug!("Inserting Right: k:{:?} v:({})", key, value.len());
                    right.insert(key, value).expect("Space to now exist");
                }

                info.try_push(InsertInfo::SplitRoot(node.pid())).expect("infallible");
                return info;
            }

            // We need to split the page, start by locking parents until we either
            // hit the root, or we've found a parent that has space for a new page
            // pointer.
            let path_len = path.len();
            let mut parents = 0usize;

            while path_len - parents > 0 {
                parents += 1;

                let parts = path.split_at_mut(path_len - parents + 1);
                let parent = parts.0.last_mut().expect("infallible");
                let child = parts.1.first().unwrap_or(&node);

                // let parent = &mut path[path_len - parents];
                let version = parent.version();
                parent.upgrade_exclusive();

                if version != parent.version() {
                    debug!("Version mismatch, restart.");
                    continue 'restart;
                }

                // let child = path.get(path_len - parents - 1).unwrap_or(&node);
                let (_pivot, klen, _vlen) = child.pivot_for_split();

                if parent.has_space_for(klen, size_of::<u64>()) {
                    break;
                } else if parents == path_len {
                    // we need to split the root!

                    trace!("SPLIT ROOT #2");

                    let (pivot, _klen, _vlen) = parent.pivot_for_split();

                    let right_swip: Swip<Node> = self.buffer_manager.new_page().unwrap();
                    let left_swip: Swip<Node> = self.buffer_manager.new_page().unwrap();

                    let mut right = right_swip
                        .coupled_page_guard::<Node>(None, LatchStrategy::Exclusive)
                        .expect("infallible");
                    let mut left = left_swip
                        .coupled_page_guard::<Node>(None, LatchStrategy::Exclusive)
                        .expect("infallible");

                    parent.root_split(&mut left, &mut right, pivot);
                    info.try_push(InsertInfo::SplitRoot(parent.pid())).expect("infallible");

                    // TODO(perf): restarting works, but is inefficient as we need to redo all
                    // of our locks
                    continue 'restart;

                    // let s = node
                    //     .get_next(&key)
                    //     .expect("All inner nodes should return _something_");

                    // let n = if s == left.swip_bytes() {
                    //     right.downgrade();
                    //     left
                    // } else {
                    //     left.downgrade();
                    //     right
                    // };

                    // parent.downgrade();
                    // let _ = path.pop();
                    // path.try_push(n).expect("infallible");

                    // break;
                }
            }

            // At this point we hold exclusive locks from the leaf up to a parent
            // that can hold a new swip, or the entire path, which means root node.

            for i in path_len - parents..path_len {
                let parts = path.split_at_mut(i + 1);
                let parent = parts.0.last().expect("infallible");
                let left = parts.1.first().unwrap_or(&node);

                
                debug_assert!(parent.contains(left),
                    "Mismatched fences between parent and child.\nKey: {:?}\nParent: {:?}\nLeft: {:?}\n\nPath: {:?}\nNode:{:?}\nInfo: {:?}",
                    key, parent, left, parts, node, info);

                let parent = parts.0.last_mut().expect("infallible");
                let left = parts.1.first_mut().unwrap_or(&mut node);

                if left.is_leaf() && left.entry_count() < HEURISTIC_RESIZE_THRESHOLD {
                    // Instead of splitting, in this case we will resize the node
                    let mut new_node = self.resize_node(left, key, value);
                    // TODO: release left

                    trace!("INSERT AFTER RESIZE. k({}), v({}), remainder: {}", 
                        key.len(), value.len(), new_node.available_space());
                    new_node.insert(key, value).expect("Space to now exist");

                    trace!("RESIZE NODE: {:?}", new_node);

                    parent.insert_inner(&new_node).unwrap();
                    info.try_push(InsertInfo::ResizedPage(new_node.pid())).expect("infallible");

                    debug_assert_eq!(
                        parent.get_next(new_node.lower_fence()).unwrap(),
                        new_node.swip_bytes(),
                    );

                    return info;
                }

                let (pivot, _klen, _vlen) = left.pivot_for_split();

                trace!("SPLIT NODE: {:?}", left);

                // TODO(safety): handle error
                let right_swip: Swip<Node> = self.buffer_manager.new_page_with_capacity(left.data_capacity()).unwrap();
                let mut right = right_swip
                    .coupled_page_guard::<Node>(None, LatchStrategy::Exclusive)
                    .expect("infallible");

                left.split(&mut right, pivot);
                debug_assert_eq!(right.lower_fence(), left.upper_fence());
                debug_assert!(right.upper_fence() == &[] || left.lower_fence() < right.upper_fence());
                info.try_push(InsertInfo::SplitPage(left.pid())).expect("infallible");

                parent.insert_inner(&right).expect("infallible");

                debug_assert_eq!(
                    parent.get_next(right.lower_fence()).unwrap(),
                    right.swip_bytes(),
                );

                parent.downgrade();

                if !left.is_leaf() {
                    if key >= left.upper_fence() {
                        path[i + 1] = right;
                    }

                    continue;
                } else {
                    if key < left.upper_fence() {
                        // TODO: It is not guaranteed that LEFT is large enough.
                        // Given:
                        //  LEFT holds > 6 keys, and thus was split
                        //  New key:value is large enough in which it doesn't fit
                        //  Notably, if LEFT is left weighted large keys, left might
                        //  still be huge.
                        //  Even if we select a pivot based on space (which we should do),
                        //  This will still be an issue if new key:value >= 50% of the
                        //  capacity of the node.
                        //
                        //  We also cannot just resize and insert into parent, as the parent
                        //  may not have space :/
                        //
                        // Cheat by restarting, non-optimized

                        match left.insert(key, value) {
                            Ok(()) => trace!("SPLIT NODE LEAF INSERT LEFT"),
                            Err(InsertError::InsufficientSpace) => {
                                trace!("SPLIT NODE LEAF INSERT LEFT -- FAILED, RESTARTING");
                                continue 'restart;
                            }
                        };
                    } else {
                        match right.insert(key, value) {
                            Ok(()) => trace!("SPLIT NODE LEAF INSERT RIGHT"),
                            Err(InsertError::InsufficientSpace) => {
                                trace!("SPLIT NODE LEAF INSERT RIGHT -- FAILED, RESTARTING");
                                continue 'restart;
                            }
                        }
                    }

                    return info;
                }
            }

            unreachable!();
        }
    }

    fn resize_node(&self, node: &mut PageGuard<Node>, key: &[u8], value: &[u8]) -> PageGuard<Node> {
        let required_space = node.capacity_with(key, value);
        let buffered_space = (required_space as f64 * 1.3) as usize;
        let new_swip: Swip<Node> = self.buffer_manager.new_page_with_capacity(buffered_space).unwrap();

        let mut new_node = new_swip
            .coupled_page_guard::<Node>(None, LatchStrategy::Exclusive)
            .expect("infallible");

        node.clone_to(&mut new_node);

        debug_assert_eq!(node.lower_fence(), new_node.lower_fence());
        debug_assert_eq!(node.upper_fence(), new_node.upper_fence());

        new_node
    }

    pub fn delete(&self, key: &[u8]) -> bool {
        let (mut node, _) = self.search_to_leaf(&key, LatchStrategy::Exclusive);
        node.delete(key)

        // TODO: merge low capacity nodes
    }

    /// Looks up a value that matches the provided key in the `BTree`,
    /// returning a copy of the value if found, else returns `None`.
    ///
    /// Cost: Performs a single heap allocation to copy the data from the
    /// backing page data to return a safe copy.
    ///
    /// TODO: implement Alloc variant `get_in`
    pub fn get(&self, key: &[u8]) -> Option<Box<[u8]>> {
        let (node, _) = self.search_to_leaf(&key, LatchStrategy::Shared);
        node.get(key).map(|v| v.to_vec().into_boxed_slice())
    }

    /// Returned PageGuard is returned locked via `strategy`
    /// Returned PageGuards in GuardPath are Optimistic
    fn search_to_leaf(&self, key: &[u8], strategy: LatchStrategy) -> (PageGuard<Node>, GuardPath) {
        trace!("search_to_leaf: {:?}", key);

        'restart: loop {
            let mut path: GuardPath = Default::default();
            let mut swip = self.root_page;

            loop {
                let parent = path.last();
                let node = match swip.coupled_page_guard(parent, LatchStrategy::OptimisticSpin) {
                    Ok(n) => n,
                    Err(OptimisticError::Conflict) => continue 'restart,
                };

                if let Some(parent) = parent {
                    debug_assert!(
                        parent.contains(&node),
                        "Parent: {:?}\n  Node: {:?}",
                        parent,
                        node
                    );

                    debug_assert_eq!(
                        parent.get(node.lower_fence()).unwrap(),
                        &swip.value_bytes()[..]
                    );
                }

                if node.is_leaf() {
                    // TODO: If we tracked height, we could perform this Shared latch without the spin above
                    let node = match swip.coupled_page_guard(parent, strategy) {
                        Ok(n) => n,
                        Err(OptimisticError::Conflict) => continue 'restart,
                    };

                    trace!("[search_to_leaf] {:?}\nNode: {:?}", path, node);

                    return (node, path);
                } else {
                    let s = node
                        .get_next(&key)
                        .expect("All inner nodes should return _something_");
                    let ptr = usize::from_ne_bytes(s.try_into().expect("8 byte response"));

                    let child: Swip<Node> = Swip::new(ptr);

                    // Need to make sure the data in this node didn't shift under us
                    // TODO: is this actually needed?
                    if !node.is_valid() {
                        continue;
                    }

                    if !child.is_swizzled() {
                        // I assume I will need more information to perform the load.
                        // Possibly the path is enough, but in the end, the owner of the
                        // child needs to be updated to unswizzle the pointer in the page.
                        // Alt: Maybe just swizzle sync?
                        // Also: Parallel swizzling for scans would be neat
                        // return Err(SearchError::PageFault(child));

                        todo!("implement unswizzling");
                    }

                    // Safety: Assuming path's default value is large enough to hold tree depth
                    path.try_push(node).unwrap();
                    swip = child
                }
            }
        }
    }

    /// Creates an iterator over some range of the `BTree` returning chunks of
    /// key-value pairs each iteration.
    ///
    /// To limit the amount of returned data (and cloning of said data), caller
    /// can provide a predicate function to filter out results prior to cloning
    /// the data.
    ///
    /// Cost: Heap allocates (into the Global allocator), key-value pairs a
    /// page at a time. Key-value pairs filtered out via the provided predicate
    /// are not cloned.
    /// Leaf pages are reader locked while performing predicate scan and
    /// key-value cloning.
    ///
    /// TODO: Should we provide an interface that allows iterating without any
    /// heap allocation by holding the lock during the duration of the caller?
    pub fn range<F>(
        &'a self,
        start: &[u8],
        end: Option<&'a [u8]>,
        pred: F,
    ) -> BTreeRange<'a, Global, F>
    where
        F: Fn(&[u8]) -> bool + Clone,
    {
        self.range_in(Global.clone(), start, end, pred)
    }

    /// See `range`.
    ///
    /// Variant of `range` in which allocation happens on the provided `Alloc`.
    /// This is useful for the caller to manage alloc/dealloc costs by doing
    /// things like utilizing a simple Arena allocator in whihc
    pub fn range_in<F, A>(
        &'a self,
        alloc: A,
        start: &[u8],
        end: Option<&'a [u8]>,
        pred: F,
    ) -> BTreeRange<'a, A, F>
    where
        F: Fn(&[u8]) -> bool,
        A: Allocator + Clone,
    {
        BTreeRange {
            btree: &self,
            lower_fence: Some(start.to_vec()),
            upper_bound: end,
            pred,
            alloc,
        }
    }
}

pub struct BTreeRange<'a, A: Allocator + Clone, F>
where
    F: Fn(&[u8]) -> bool,
{
    btree: &'a BTree<'a>,
    lower_fence: Option<Vec<u8>>,
    upper_bound: Option<&'a [u8]>,
    alloc: A,
    pred: F,
}

impl<'a, A: Allocator + Clone, F> Iterator for BTreeRange<'a, A, F>
where
    F: Fn(&[u8]) -> bool,
{
    type Item = Vec<(Box<[u8], A>, Box<[u8], A>)>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.lower_fence.is_none() {
            return None;
        }

        let fence = self.lower_fence.take().unwrap();
        let (node, _) = self.btree.search_to_leaf(&fence, LatchStrategy::Shared);

        let upper_fence = node.upper_fence();
        if upper_fence.len() == 0 {
            self.lower_fence = None;
        } else {
            self.lower_fence = Some(node.upper_fence().to_vec());
        }

        Some(node.clone_key_values_until(
            self.upper_bound.as_ref().map(|k| &k[..]),
            &self.pred,
            self.alloc.clone(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use crate::buffer_manager;

    use super::*;
    use std::{time::Instant, assert_matches::assert_matches};

    fn make_buffer_manager() -> BufferManager {
        buffer_manager::Builder::new()
            .max_memory(4096 * 4096 * 5000)
            .build()
    }

    #[test]
    fn insert_get() {
        let bm = make_buffer_manager();
        let btree = BTree::new(&bm);
        btree.insert(b"foo", b"bar");
        assert_eq!(Some(b"bar".to_vec().into_boxed_slice()), btree.get(b"foo"));
    }

    #[test]
    fn delete() {
        let bm = make_buffer_manager();
        let btree = BTree::new(&bm);
        btree.insert(b"foo", b"bar");
        assert_eq!(Some(b"bar".to_vec().into_boxed_slice()), btree.get(b"foo"));
        assert!(btree.delete(b"foo"));
        assert_eq!(None, btree.get(b"foo"));
    }

    #[test]
    fn basic_range_query() {
        let bm = make_buffer_manager();

        let btree = BTree::new(&bm);
        btree.insert(b"aaa", b"aaa");
        btree.insert(b"bbb", b"bbb");
        btree.insert(b"ccc", b"ccc");
        btree.insert(b"ddd", b"ddd");
        btree.insert(b"eee", b"eee");
        btree.insert(b"fff", b"fff");

        let range = btree.range(b"aaa", Some(b"d"), |_| true);
        for chunk in range {
            // println!("Chunk: {:?}", chunk);
            for kv in chunk.iter() {
                println!("KV: {:?}", kv);
                // let k = &kv.0;
                // let v = &kv.1;

                // println!("k: {:?}", k[0]);
            }
        }
    }

    // #[test]
    // fn node_split_big_values() {
    //     let _ = env_logger::builder().is_test(true).try_init();
    //     let bm = make_buffer_manager();

    //     let btree = BTree::new(&bm);
    //     btree.insert(b"foo", &[1u8; 1028 * 10]);
    //     btree.insert(b"bar", &[2u8; 1028 * 10]);
    //     btree.insert(b"mmm", &[3u8; 1028 * 10]);

    //     assert_eq!(Some([1u8; 1028 * 10].to_vec()), btree.get(b"foo"));
    //     assert_eq!(Some([2u8; 1028 * 10].to_vec()), btree.get(b"bar"));
    //     assert_eq!(Some([3u8; 1028 * 10].to_vec()), btree.get(b"mmm"));
    // }

    #[test]
    fn node_split_many_single_level() {
        use rand::Rng;
        use std::collections::HashMap;

        let _ = env_logger::builder().is_test(true).try_init();

        let bm = make_buffer_manager();
        let btree = BTree::new(&bm);

        let mut hashmap = HashMap::new();

        for _ in 0..10000 {
            let key = rand::thread_rng().gen::<[u8; 32]>();
            let value = rand::thread_rng().gen::<[u8; 32]>();

            btree.insert(&key, &value);
            hashmap.insert(key, value);
        }

        for (i, (k, v)) in hashmap.iter().enumerate() {
            assert_eq!(
                Some(v.to_vec().into_boxed_slice()),
                btree.get(&k[..]),
                "Failed on check {}",
                i
            );
        }
    }

    fn force_root_split<const VLEN: usize>(btree: &BTree) {
        assert_eq!(6, HEURISTIC_RESIZE_THRESHOLD);

        let kvs = [
            (b"aaa", [1u8; VLEN]),
            (b"bbb", [2u8; VLEN]),
            (b"ccc", [3u8; VLEN]),
            (b"ddd", [4u8; VLEN]),
            (b"eee", [5u8; VLEN]),
        ];

        for (k, v) in kvs.iter() {
            assert_eq!(0, btree.insert(&k[..], &v[..]).len());
        }

        // Force the initial root split, we never expect the root to be resized
        let res = btree.insert(b"fff", &[6u8; VLEN]);
        assert_eq!(1, res.len());
        assert_matches!(res[0], InsertInfo::SplitRoot(_));
    }

    #[test]
    fn grow_page() {
        let bm = buffer_manager::Builder::new()
            .max_memory(4096 * 4096 * 5000)
            .base_page_size(2048)
            .build();

        let btree = BTree::new(&bm);

        const VLEN: usize = 2048 / HEURISTIC_RESIZE_THRESHOLD;
        force_root_split::<VLEN>(&btree);

        let kvs = [
            (b"baa", [11u8; VLEN]),
            (b"bab", [12u8; VLEN]),
            (b"bac", [13u8; VLEN]),
        ];

        for (k, v) in kvs.iter() {
            let res = btree.insert(&k[..], &v[..]);
            assert_eq!(0, res.len(), "Got {:?} inserting: {:?}", res, &k[..]);
        }

        // This node took 2 from the prior split, therefore the 4th insert
        // should cause a resize, since there are only 5 keys in the node
        let res = btree.insert(b"bad", &[14u8; VLEN]);
        assert_eq!(1, res.len());
        assert_matches!(res[0], InsertInfo::ResizedPage(_));

        assert_eq!(
            Some(vec![14u8; VLEN].into_boxed_slice()),
            btree.get(b"bad")
        );

        // Let's now force this resized node to split and insert to the right
        // to ensure it is of appropriate size.
        let kvs = [
            (b"bada", [21u8; VLEN]),
            (b"badb", [22u8; VLEN]),
            (b"babc", [23u8; VLEN]),
            (b"babd", [24u8; VLEN]),
            (b"babe", [25u8; VLEN]),
        ];

        for (k, v) in kvs.iter() {
            let res = btree.insert(&k[..], &v[..]);
            assert_eq!(0, res.len(), "Got {:?} inserting: {:?}", res, &k[..]);
        }

        // This node took 2 from the prior split, therefore the 4th insert
        // should cause a resize, since there are only 5 keys in the node
        let res = btree.insert(b"babf", &[26u8; VLEN]);
        assert_eq!(1, res.len());
        assert_matches!(res[0], InsertInfo::SplitPage(_));
    }

    #[test]
    #[ignore]
    fn large_value_on_split() {
        let bm = buffer_manager::Builder::new()
            .max_memory(4096 * 4096 * 5000)
            .base_page_size(2048)
            .build();

        let btree = BTree::new(&bm);

        const VLEN: usize = 2048 / HEURISTIC_RESIZE_THRESHOLD;
        force_root_split::<VLEN>(&btree);

        let kvs = [
            (b"baa", [11u8; VLEN / 2]),
            (b"bab", [12u8; VLEN / 2]),
            (b"bac", [13u8; VLEN / 2]),
            (b"bad", [14u8; VLEN / 2]),
            (b"bae", [15u8; VLEN / 2]),
            (b"baf", [16u8; VLEN / 2]),
        ];

        for (k, v) in kvs.iter() {
            let res = btree.insert(&k[..], &v[..]);
            assert_eq!(0, res.len(), "Got {:?} inserting: {:?}", res, &k[..]);
        }

        // This will cause a split, but the new key is too large for the
        // new split node
        let res = btree.insert(b"bag", &[17u8; VLEN*4]);
        assert_eq!(1, res.len());
        assert_matches!(res[0], InsertInfo::SplitPage(_));

        assert_eq!(
            Some(vec![17u8; VLEN*4].into_boxed_slice()),
            btree.get(b"bad")
        );
    }

    #[test]
    #[ignore]
    fn node_split_many_multi_level() {
        use rand::Rng;
        use std::collections::HashMap;

        let _ = env_logger::builder().is_test(true).try_init();

        let bm = make_buffer_manager();
        let btree = BTree::new(&bm);

        let count = 10000000;

        let mut hashmap = HashMap::with_capacity(count);
        let mut vec = Vec::with_capacity(count);

        for _ in 0..count {
            let key = rand::thread_rng().gen::<[u8; 16]>();
            let value = rand::thread_rng().gen::<[u8; 16]>();

            vec.push((key, value));
            hashmap.insert(key, value);
        }

        let start = Instant::now();

        for (k, v) in &vec {
            btree.insert(k, v);
        }

        let elapsed = start.elapsed().as_nanos();

        println!("Insert completed in {}ms", elapsed / 1000000);
        println!("Ops/s: {}", 1000000000 / (elapsed / count as u128));
        println!("Checking results...");

        for (i, (k, v)) in hashmap.iter().enumerate() {
            if Some(v.to_vec().into_boxed_slice()) != btree.get(&k[..]) {
                let (node, path) = btree.search_to_leaf(k, LatchStrategy::OptimisticSpin);

                fn debug_fence_mismatch(node: &PageGuard<Node>) {
                    let children = node.all_children();

                    children.windows(2).for_each(|w| {
                        let left = &w[0];
                        let right = &w[1];

                        if left.upper_fence() != right.lower_fence() {
                            println!(
                                "Fence mismatch:\nParent: {:?}\n    Left:   {:?}\n    Right:  {:?}",
                                node, left, right
                            );
                        }
                    });

                    for child in children.iter() {
                        if !child.is_leaf() {
                            debug_fence_mismatch(child);
                        }
                    }
                }

                debug_fence_mismatch(path.first().unwrap());

                panic!(
                    "Failed on check {}\n\nKey: {:?}\n\nNode: {:?}\n\nPath: {:?}",
                    i, k, node, path
                );
            }
        }
    }

    #[test]
    #[ignore]
    fn threaded_split_many_multi_level() {
        use rand::Rng;
        use std::sync::Arc;

        let _ = env_logger::builder().is_test(true).try_init();

        // let mut threads = vec![];
        let bm = make_buffer_manager();
        let btree = BTree::new(&bm);

        let count = 10000000;
        let thread_count = 4;

        let mut kv_buckets = vec![];

        for _ in 0..thread_count {
            // let mut hashmap = HashMap::with_capacity(count);
            let mut vec = Vec::with_capacity(count);

            for _ in 0..count {
                let key = rand::thread_rng().gen::<[u8; 16]>();
                let value = rand::thread_rng().gen::<[u8; 16]>();
                vec.push((key, value));
                // hashmap.insert(key, value);
            }

            kv_buckets.push(vec);
        }

        let all_key_values: Vec<_> = kv_buckets.clone().into_iter().flatten().collect();

        let start = Instant::now();

        crossbeam::thread::scope(|s| {
            for i in 0..thread_count {
                let vec = kv_buckets[i].clone();
                let inner_btree = btree.clone();
                s.spawn(move |_| {
                    let btree = inner_btree;

                    for (k, v) in vec.into_iter() {
                        btree.insert(&k, &v);
                    }

                    // for (_i, (k, v)) in hashmap.iter().enumerate() {
                    //     assert_eq!(Some(v.to_vec()), btree.get(&k[..]).unwrap());
                    // }
                });
            }
        })
        .unwrap();

        let elapsed = start.elapsed().as_nanos();

        println!("Insert completed in {}ms", elapsed / 1000000);
        println!(
            "Ops/s: {}",
            1000000000 / (elapsed / (count * thread_count) as u128)
        );

        // all_key_values
        use rand::distributions::{Distribution, Uniform};

        let kvs = Arc::new(all_key_values);
        // let mut threads = vec![];
        let thread_count = thread_count * 2;

        let start = Instant::now();

        crossbeam::thread::scope(|s| {
            for _ in 0..thread_count {
                let inner_kvs = kvs.clone();
                let inner_btree = btree.clone();

                s.spawn(move |_| {
                    let kvs = inner_kvs;
                    let btree = inner_btree;

                    let mut rng = rand::thread_rng();
                    let die = Uniform::from(0..kvs.len());

                    for _ in 0..count {
                        let idx = die.sample(&mut rng);
                        btree.get(&kvs[idx].0).unwrap();
                    }
                });
            }
        })
        .unwrap();

        let elapsed = start.elapsed().as_nanos();

        println!("Read completed in {}ms", elapsed / 1000000);
        println!(
            "Ops/s: {}",
            1000000000 / (elapsed / (count * thread_count) as u128)
        );
    }
}
