use std::{mem::size_of, fmt::Debug};
use std::sync::Arc;

use log::debug;
use stackvec::StackVec;

use crate::buffer_manager::swip::OptimisticError;
use crate::buffer_manager::{BufferManager, swip::Swip, buffer_frame::{PageGuard, LatchStrategy}};

use self::node::{Node, MAX_KEY_LEN};

mod node;
// mod key_chunks;

#[derive(Debug, PartialEq)]
enum InsertError {
    PageFault(Swip<Node>),
}

impl From<SearchError> for InsertError {
    fn from(e: SearchError) -> Self {
        match e {
            SearchError::PageFault(s) => InsertError::PageFault(s),
        }
    }
}

#[derive(Debug, PartialEq)]
pub(crate) enum SearchError {
    PageFault(Swip<Node>)
}

// The size of this stack limits the depth that a tree can become.
// We need to statically hold this max depth, so that we can 
type NodePath = StackVec<[(Swip<Node>, u64); 32]>;
type LockPath = StackVec<[PageGuard<Node>; 32]>;


#[derive(Clone)]
struct BTree {
    buffer_manager: Arc<BufferManager>,
    root_page: Swip<Node>,
}

impl BTree {
    pub fn new(buffer_manager: Arc<BufferManager>) -> Self {
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
            buffer_manager,
            root_page: root_swip,
        }
    }

    pub fn insert(&self, key: &[u8], value: &[u8]) -> Result<(), InsertError> {
        'outer: loop {
            let (mut node, mut path) = self.search_to_leaf(&key, LatchStrategy::Exclusive)?;

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
                Ok(()) => return Ok(()),
                Err(node::InsertError::InsufficientSpace) => (),
            };

            if path.len() == 0 {
                println!("SPLIT ROOT #1");

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

                return Ok(());
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

                    println!("SPLIT ROOT #2");

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

            return Ok(());
        }
    }

    pub fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>, SearchError> {
        let (node, _) = self.search_to_leaf(&key, LatchStrategy::Shared)?;
        Ok(node.get(key).map(|v| v.to_vec()))
    }

    /// Returned PageGuard are returned locked via `strategy`
    /// Returned PageGuards in LockPath are Optimistic
    fn search_to_leaf(&self, key: &[u8], strategy: LatchStrategy) -> Result<(PageGuard<Node>, LockPath), SearchError> {
        debug!("search_to_leaf: {:?}", key);

        'restart: loop {
            let mut path: LockPath = Default::default();
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

                debug!("pid: {}, is_leaf: {}", node.pid(), node.is_leaf());
                
                if node.is_leaf() {
                    // TODO: If we tracked height, we could perform this Shared latch without the spin above
                    let node = match swip.coupled_page_guard(parent, strategy) {
                        Ok(n) => n,
                        Err(OptimisticError::Conflict) => continue 'restart,
                    };

                    return Ok((node, path));
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
                        return Err(SearchError::PageFault(child));
                    }

                    // Safety: Assuming path's default value is large enough to hold tree depth
                    path.try_push(node).unwrap();
                    swip = child
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{time::Instant, thread};

    use crate::tests::FakeDiskManager;

    use super::*;

    #[test]
    fn insert_get() {
        let dm = Arc::new(FakeDiskManager::default());
        let bm = Arc::new(BufferManager::new(dm, 4096 * 128));

        let btree = BTree::new(bm);
        btree.insert(b"foo", b"bar").unwrap();
        assert_eq!(Some(b"bar".to_vec()), btree.get(b"foo").unwrap());
    }

    #[test]
    fn node_split_big_values() {
        let _ = env_logger::builder().is_test(true).try_init();

        let dm = Arc::new(FakeDiskManager::default());
        let bm = Arc::new(BufferManager::new(dm, 4096 * 128));

        let btree = BTree::new(bm);
        btree.insert(b"foo", &[1u8; 1028 * 10]).unwrap();
        btree.insert(b"bar", &[2u8; 1028 * 10]).unwrap();
        btree.insert(b"mmm", &[3u8; 1028 * 10]).unwrap();

        assert_eq!(Some([1u8; 1028 * 10].to_vec()), btree.get(b"foo").unwrap());
        assert_eq!(Some([2u8; 1028 * 10].to_vec()), btree.get(b"bar").unwrap());
        assert_eq!(Some([3u8; 1028 * 10].to_vec()), btree.get(b"mmm").unwrap());
    }

    #[test]
    fn node_split_many_single_level() {
        use rand::Rng;
        use std::collections::HashMap;

        let _ = env_logger::builder().is_test(true).try_init();

        let dm = Arc::new(FakeDiskManager::default());
        let bm = Arc::new(BufferManager::new(dm, 4096 * 4096 * 4));
        let btree = BTree::new(bm);

        let mut hashmap = HashMap::new();

        for _ in 0..10000 {
            let key = rand::thread_rng().gen::<[u8; 32]>();
            let value = rand::thread_rng().gen::<[u8; 32]>();

            btree.insert(&key, &value).unwrap();
            hashmap.insert(key, value);
        }
        
        for (i, (k, v)) in hashmap.iter().enumerate() {
            assert_eq!(Some(v.to_vec()), btree.get(&k[..]).unwrap(), "Failed on check {}", i);
        }
    }

    #[test]
    fn node_split_many_multi_level() {
        use rand::Rng;
        use std::collections::HashMap;

        let _ = env_logger::builder().is_test(true).try_init();

        let dm = Arc::new(FakeDiskManager::default());
        let bm = Arc::new(BufferManager::new(dm, 4096 * 4096 * 100));
        let btree = BTree::new(bm);

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
            btree.insert(k, v).unwrap();
        }

        let elapsed = start.elapsed().as_nanos();

        println!("Insert completed in {}ms", elapsed / 1000000);
        println!("Ops/s: {}", 1000000000 / (elapsed / count as u128));
        println!("Checking results...");
        
        for (i, (k, v)) in hashmap.iter().enumerate() {
            if Some(v.to_vec()) != btree.get(&k[..]).unwrap() {
                let (node, path) = btree.search_to_leaf(k, LatchStrategy::OptimisticSpin).unwrap();

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
    fn threaded_split_many_multi_level() {
        use rand::Rng;
        use std::collections::HashMap;

        let _ = env_logger::builder().is_test(true).try_init();

        let mut threads = vec![];
        let dm = Arc::new(FakeDiskManager::default());
        let bm = Arc::new(BufferManager::new(dm, 4096 * 4096 * 5000));
        let btree = BTree::new(bm);

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

        for i in 0..thread_count {
            let vec = kv_buckets[i].clone();
            let inner_btree = btree.clone();
            let handle = thread::spawn(move || {
                let btree = inner_btree;

                for (k, v) in vec.into_iter() {
                    btree.insert(&k, &v).unwrap();
                }
                
                // for (_i, (k, v)) in hashmap.iter().enumerate() {
                //     assert_eq!(Some(v.to_vec()), btree.get(&k[..]).unwrap());
                // }
            });

            threads.push(handle);
        }

        for t in threads.into_iter() {
            t.join().unwrap();
        }

        let elapsed = start.elapsed().as_nanos();

        println!("Insert completed in {}ms", elapsed / 1000000);
        println!("Ops/s: {}", 1000000000 / (elapsed / (count * thread_count) as u128));

        // all_key_values
        use rand::distributions::{Distribution, Uniform};
        
        let kvs = Arc::new(all_key_values);
        let mut threads = vec![];
        let thread_count = thread_count * 2;

        let start = Instant::now();

        for _ in 0..thread_count {
            let inner_kvs = kvs.clone();
            let inner_btree = btree.clone();

            let handle = thread::spawn(move || {
                let kvs = inner_kvs;
                let btree = inner_btree;

                let mut rng = rand::thread_rng();
                let die = Uniform::from(0..kvs.len());

                for _ in 0..count {
                    let idx = die.sample(&mut rng);
                    btree.get(&kvs[idx].0).unwrap();
                }
            });

            threads.push(handle);
        }

        for t in threads.into_iter() {
            t.join().unwrap();
        }

        let elapsed = start.elapsed().as_nanos();

        println!("Read completed in {}ms", elapsed / 1000000);
        println!("Ops/s: {}", 1000000000 / (elapsed / (count * thread_count) as u128));
    }
}
