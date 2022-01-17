use std::alloc::{Allocator, Global};
use std::mem::size_of;

use log::debug;
use stackvec::StackVec;

use crate::buffer_manager::swip::OptimisticError;
use crate::buffer_manager::{BufferManager, swip::Swip, buffer_frame::{PageGuard, LatchStrategy}};

use self::node::Node;

mod node;

// The size of this stack limits the depth that a tree can become.
// We need to statically hold this max depth, so that we can 
type GuardPath = StackVec<[PageGuard<Node>; 32]>;


#[derive(Clone)]
pub struct BTree<'a> {
    buffer_manager: &'a BufferManager,
    root_page: Swip<Node>,
}

impl<'a> BTree<'a> {
    pub fn new(buffer_manager: &'a BufferManager) -> Self {
        let root_swip: Swip<Node> = buffer_manager.new_page().unwrap();
        let mut node = root_swip.coupled_page_guard::<Node>(None, LatchStrategy::Exclusive).expect("infallible");
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

    pub fn insert(&self, key: &[u8], value: &[u8]) {
        'outer: loop {
            let (mut node, mut path) = self.search_to_leaf(&key, LatchStrategy::Exclusive);

            // all_children

            #[cfg(debug_assertions)] {
                if !node.shares_prefix(key) {
                    let children = path.last().unwrap().all_children();

                    children.windows(2).for_each(|w| {
                        let left = &w[0];
                        let right = &w[1];

                        if left.upper_fence() != right.lower_fence() {
                            println!("Fence mismatch:\nLeft: {:?}\nRight: {:?}", left, right);
                        }
                    });

                    panic!("Key: {:?}\n\nNode: {:?}\n\nPath: {:?}", 
                        key, node, path);
                }

                path.windows(2).for_each(|w| {
                    let parent = &w[0];
                    let child = &w[1];
                    debug_assert!(parent.contains(&child), "Parent: {:?}\n  Child: {:?}", parent, child);
                });

                if let Some(parent) = path.last() {
                    debug_assert!(parent.contains(&node), "Parent: {:?}\n  Child: {:?}", parent, node);
                }
            }

            match node.insert(&key, &value) {
                Ok(()) => return,
                Err(node::InsertError::InsufficientSpace) => (),
            };

            if path.len() == 0 {
                debug!("SPLIT ROOT #1");

                let right_swip: Swip<Node> = self.buffer_manager.new_page().unwrap();
                let left_swip: Swip<Node> = self.buffer_manager.new_page().unwrap();

                let mut right = right_swip.coupled_page_guard::<Node>(None, LatchStrategy::Exclusive).expect("infallible");
                let mut left = left_swip.coupled_page_guard::<Node>(None, LatchStrategy::Exclusive).expect("infallible");

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

                return;
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
                    continue 'outer;
                }

                // let child = path.get(path_len - parents - 1).unwrap_or(&node);
                let (pivot, klen, _vlen) = child.pivot_for_split();

                if parent.has_space_for(klen, size_of::<u64>()) {
                    break;
                } else if parents == path_len {
                    // we need to split the root!

                    debug!("SPLIT ROOT #2");

                    let right_swip: Swip<Node> = self.buffer_manager.new_page().unwrap();
                    let left_swip: Swip<Node> = self.buffer_manager.new_page().unwrap();

                    let mut right = right_swip.coupled_page_guard::<Node>(None, LatchStrategy::Exclusive).expect("infallible");
                    let mut left = left_swip.coupled_page_guard::<Node>(None, LatchStrategy::Exclusive).expect("infallible");

                    parent.root_split(&mut left, &mut right, pivot);

                    let s = node.get_next(&key).expect("All inner nodes should return _something_");

                    let n = if s == left.swip_bytes() {
                        right.downgrade();
                        left
                    } else {
                        left.downgrade();
                        right
                    };

                    parent.downgrade();
                    let _ = path.pop();
                    path.try_push(n).expect("infallible");

                    break;
                }
            }

            // if parents == path_len && path[0].has_space_for(key_len, val_len)

            // At this point we hold exclusive locks from the leaf up to a parent
            // that can hold a new swip, or the entire path, which means root node.

            let mut right_guard = None;

            for i in path_len-parents..path_len {
                // let temp = &mut [0u8; MAX_KEY_LEN];

                let parts = path.split_at_mut(i + 1);
                let parent = parts.0.last().expect("infallible");
                let left = parts.1.first().unwrap_or(&node);

                debug_assert!(parent.contains(left),
                    "Mismatched fences between parent and child.\nKey: {:?}\nParent: {:?}\nLeft: {:?}\n\nPath: {:?}\nNode:{:?}",
                    key, parent, left, parts, node);

                let parent = parts.0.last_mut().expect("infallible");
                let left = parts.1.first_mut().unwrap_or(&mut node);
                let (pivot, klen, vlen) = left.pivot_for_split();

                // TODO(safety): handle error
                let right_swip: Swip<Node> = self.buffer_manager.new_page().unwrap();
                let mut right = right_swip.coupled_page_guard::<Node>(None, LatchStrategy::Exclusive).expect("infallible");

                // debug!("[btree] parent.insert: {:?}=>{}", pivot_key, &right_swip.value());

                left.split(&mut right, pivot);

                debug_assert!(parent.shares_prefix(right.lower_fence()));
                debug_assert!(parent.contains(&right), "Parent: {:?}\n  Left: {:?}\n  Right: {:?}",
                    parent, left, right);

                match parent.insert_inner(&right) {
                    Ok(()) => (),
                    Err(e) => panic!("Error: {:?} trying to insert {}\n{:?}", e, klen+vlen, parent),
                };
                parent.downgrade();

                if !left.is_leaf() && key >= left.upper_fence() {
                    path[i+1] = right;
                } else {
                    right_guard = Some(right);
                }
            }

            if key < node.upper_fence() {
                debug_assert!(node.shares_prefix(key));
                node.insert(key, value).expect("Space to now exist");
            } else {
                let mut right = right_guard.expect("infallible");
                debug_assert!(right.shares_prefix(key));
                debug_assert!(key >= right.lower_fence());
                right.insert(key, value).expect("Space to now exist");
            }

            return;
        }
    }

    pub fn get(&self, key: &[u8]) -> Option<Vec<u8>> {
        let (node, _) = self.search_to_leaf(&key, LatchStrategy::Shared);
        node.get(key).map(|v| v.to_vec())
    }

    /// Returned PageGuard are returned locked via `strategy`
    /// Returned PageGuards in LockPath are Optimistic
    fn search_to_leaf(&self, key: &[u8], strategy: LatchStrategy) -> (PageGuard<Node>, GuardPath) {
        debug!("search_to_leaf: {:?}", key);

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
                    debug_assert!(parent.contains(&node), "Parent: {:?}\n  Node: {:?}", parent, node);
                }
                
                if node.is_leaf() {
                    // TODO: If we tracked height, we could perform this Shared latch without the spin above
                    let node = match swip.coupled_page_guard(parent, strategy) {
                        Ok(n) => n,
                        Err(OptimisticError::Conflict) => continue 'restart,
                    };

                    return (node, path);
                } else {
                    let s = node.get_next(&key).expect("All inner nodes should return _something_");
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

    pub fn range<F>(
        &'a self, start: &[u8], end: Option<&'a [u8]>, pred: F
    ) -> BTreeRange<'a, Global, F>
    where F: Fn(&[u8]) -> bool + Clone {
        self.range_in(Global.clone(), start, end, pred)
    }

    pub fn range_in<F, A>(
        &'a self, alloc: A, start: &[u8], end: Option<&'a [u8]>, pred: F
    ) -> BTreeRange<'a, A, F>
    where F: Fn(&[u8]) -> bool, A: Allocator + Clone {
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
    where F: Fn(&[u8]) -> bool
{
    btree: &'a BTree<'a>,
    lower_fence: Option<Vec<u8>>,
    upper_bound: Option<&'a [u8]>,
    alloc: A,
    pred: F,
}

impl<'a, A: Allocator + Clone, F> Iterator for BTreeRange<'a, A, F>
    where F: Fn(&[u8]) -> bool
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
            self.alloc.clone())
        )
    }
}




#[cfg(test)]
mod tests {
    use std::time::Instant;
    use super::*;

    fn make_buffer_manager() -> BufferManager {
        BufferManager::new(4096 * 4096 * 5000)
    }

    #[test]
    fn insert_get() {
        let bm = make_buffer_manager();
        let btree = BTree::new(&bm);
        btree.insert(b"foo", b"bar");
        assert_eq!(Some(b"bar".to_vec()), btree.get(b"foo"));
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
            assert_eq!(Some(v.to_vec()), btree.get(&k[..]), "Failed on check {}", i);
        }
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
            if Some(v.to_vec()) != btree.get(&k[..]) {
                let (node, path) = btree.search_to_leaf(k, LatchStrategy::OptimisticSpin);

                fn debug_fence_mismatch(node: &PageGuard<Node>) {
                    let children = node.all_children();

                    children.windows(2).for_each(|w| {
                        let left = &w[0];
                        let right = &w[1];

                        if left.upper_fence() != right.lower_fence() {
                            println!("Fence mismatch:\nParent: {:?}\n    Left:   {:?}\n    Right:  {:?}", node, left, right);
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
                    i, 
                    k,
                    node,
                    path
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

        }).unwrap();

        let elapsed = start.elapsed().as_nanos();

        println!("Insert completed in {}ms", elapsed / 1000000);
        println!("Ops/s: {}", 1000000000 / (elapsed / (count * thread_count) as u128));

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
        }).unwrap();

        let elapsed = start.elapsed().as_nanos();

        println!("Read completed in {}ms", elapsed / 1000000);
        println!("Ops/s: {}", 1000000000 / (elapsed / (count * thread_count) as u128));
    }
}
