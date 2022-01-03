use std::{mem::size_of, fmt::Debug};

use stackvec::StackVec;

use crate::buffer_manager::{BufferManager, swip::Swip, buffer_frame::{ExclusiveGuard, OptimisticError, PageGuard, SharedGuard2, LatchStrategy}};

use self::node::{Node, MAX_KEY_LEN};

mod node;
mod node_path;
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

struct BTree<'a> {
    buffer_manager: &'a BufferManager,
    root_page: Swip<Node>,
}

impl<'a> BTree<'a> {
    pub fn new(buffer_manager: &'a BufferManager) -> Self {
        let root_swip: Swip<Node> = buffer_manager.new_page().unwrap();
        let mut node = root_swip.exclusive_lock();
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

            match node.insert(&key, &value) {
                Ok(()) => return Ok(()),
                Err(node::InsertError::InsufficientSpace) => (),
            };

            // TODO: Convert to KeyChunks? This is another limiting factor of key lengths
            let mut pivot_key = [0u8; MAX_KEY_LEN];
            let pivot_key = node.pivot_key_for_split(&key, value.len(), &mut pivot_key);

            if path.len() == 0 {
                let right_swip: Swip<Node> = self.buffer_manager.new_page().unwrap();
                let left_swip: Swip<Node> = self.buffer_manager.new_page().unwrap();

                let mut right = right_swip.coupled_page_guard::<Node>(None, LatchStrategy::Exclusive).expect("infallible");
                let mut left = left_swip.coupled_page_guard::<Node>(None, LatchStrategy::Exclusive).expect("infallible");

                node.root_split(&mut left, &mut right, &pivot_key);

                if key < &pivot_key {
                    println!("Inserting Left: k:{:?} v:({})", key, value.len());
                    left.insert(key, value).expect("Space to now exist");
                } else {
                    println!("Inserting Right: k:{:?} v:({})", key, value.len());
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

                let parent = &mut path[path_len - parents];
                let version = parent.version();
                parent.upgrade_exclusive();

                if version != parent.version() {
                    println!("Version mismatch, restart.");
                    continue 'outer;
                }

                if parents == path_len {
                    if !parent.has_space_for(pivot_key.len(), size_of::<u64>()) {
                        // we need to split the root!

                        let right_swip: Swip<Node> = self.buffer_manager.new_page().unwrap();
                        let left_swip: Swip<Node> = self.buffer_manager.new_page().unwrap();

                        let mut right = right_swip.coupled_page_guard::<Node>(None, LatchStrategy::Exclusive).expect("infallible");
                        let mut left = left_swip.coupled_page_guard::<Node>(None, LatchStrategy::Exclusive).expect("infallible");

                        parent.root_split(&mut left, &mut right, &pivot_key);

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

                if parent.has_space_for(pivot_key.len(), size_of::<u64>()) {
                    break;
                }
            }

            // if parents == path_len && path[0].has_space_for(key_len, val_len)

            // At this point we hold exclusive locks from the leaf up to a parent
            // that can hold a new swip, or the entire path, which means root node.

            let mut right_guard = None;

            for i in path_len-parents..path_len {
                let temp = &mut [0u8; MAX_KEY_LEN];

                let pk = {
                    let left = path.get_mut(i + 1).unwrap_or(&mut node);
                    left.pivot_key_for_split(key, value.len(), temp)
                };

                let parent = &mut path[i];

                // if i == 0 {
                //     let right_swip: Swip<Node> = self.buffer_manager.new_page().unwrap();
                //     let left_swip: Swip<Node> = self.buffer_manager.new_page().unwrap();

                //     let mut right = right_swip.coupled_page_guard::<Node>(None, LatchStrategy::Exclusive).expect("infallible");
                //     let mut left = left_swip.coupled_page_guard::<Node>(None, LatchStrategy::Exclusive).expect("infallible");

                //     parent.root_split(&mut left, &mut right, &pivot_key);
                //     parent.downgrade();

                //     continue;
                // }

                // TODO(safety): handle error
                let right_swip: Swip<Node> = self.buffer_manager.new_page().unwrap();
                let mut right = right_swip.coupled_page_guard::<Node>(None, LatchStrategy::Exclusive).expect("infallible");

                println!("[btree] parent.insert: {:?}=>{}", pivot_key, &right_swip.value());

                parent.insert(&pk, &right_swip.value().to_ne_bytes()).expect("space should exist");
                parent.downgrade();

                let left = path.get_mut(i + 1).unwrap_or(&mut node);

                left.split(&mut right, &pk);
                right_guard = Some(right);
            }

            if key < &pivot_key {
                node.insert(key, value).expect("Space to now exist");
            } else {
                right_guard.expect("right").insert(key, value).expect("Space to now exist");
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
        println!("search_to_leaf: {:?}", key);

        'restart: loop {
            let mut path: LockPath = Default::default();
            let mut swip = self.root_page;

            loop {
                let parent = path.last();
                let node = match swip.coupled_page_guard(parent, LatchStrategy::OptimisticSpin) {
                    Ok(n) => n,
                    Err(OptimisticError::Conflict) => continue 'restart,
                };

                println!("pid: {}, is_leaf: {}", node.pid(), node.is_leaf());

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

// impl<'a> Debug for BTree<'a> {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         f.debug_struct("BTree")
//             .field("root", self.root_page)
//             .finish()
//     }
// }

#[cfg(test)]
mod tests {
    use crate::tests::FakeDiskManager;

    use super::*;

    #[test]
    fn insert_get() {
        let dm = Box::new(FakeDiskManager::default());
        let bm = BufferManager::new(dm, 4096 * 128);

        let btree = BTree::new(&bm);
        btree.insert(b"foo", b"bar").unwrap();
        assert_eq!(Some(b"bar".to_vec()), btree.get(b"foo").unwrap());
    }

    #[test]
    fn node_split_big_values() {
        let _ = env_logger::builder().is_test(true).try_init();

        let dm = Box::new(FakeDiskManager::default());
        let bm = BufferManager::new(dm, 4096 * 128);

        let btree = BTree::new(&bm);
        btree.insert(b"foo", &[1u8; 1028 * 10]).unwrap();
        btree.insert(b"bar", &[2u8; 1028 * 10]).unwrap();

        btree.insert(b"mmm", &[3u8; 1028 * 10]).unwrap();

        assert_eq!(Some([1u8; 1028 * 10].to_vec()), btree.get(b"foo").unwrap());
        assert_eq!(Some([2u8; 1028 * 10].to_vec()), btree.get(b"bar").unwrap());
        assert_eq!(Some([3u8; 1028 * 10].to_vec()), btree.get(b"mmm").unwrap());
    }

    #[test]
    fn node_many_split() {
        use rand::Rng;
        use std::collections::HashMap;

        let _ = env_logger::builder().is_test(true).try_init();

        let dm = Box::new(FakeDiskManager::default());
        let bm = BufferManager::new(dm, 4096 * 4096 * 4);
        let btree = BTree::new(&bm);

        let mut hashmap = HashMap::new();

        for _ in 0..10000 {
            let key = rand::thread_rng().gen::<[u8; 32]>();
            let value = rand::thread_rng().gen::<[u8; 32]>();

            btree.insert(&key, &value).unwrap();
            hashmap.insert(key, value);
        }
        // for i in 0..10000usize {
        //     // let key = rand::thread_rng().gen::<[u8; 32]>();
        //     // let value = rand::thread_rng().gen::<[u8; 32]>();

        //     let key = i.to_ne_bytes();
        //     let value = i.to_ne_bytes();

        //     btree.insert(&key, &value).unwrap();
        //     hashmap.insert(key, value);
        // }
        
        for (k, v) in &hashmap {
            assert_eq!(Some(v.to_vec()), btree.get(&k[..]).unwrap());
        }
    }
}
