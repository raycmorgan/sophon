use std::io::IntoInnerError;
use std::ops::Range;
use std::path::PathBuf;
use byte_unit::{n_kib_bytes, n_mib_bytes};

use crate::{DiskManager, DiskManagerAllocError, Unswizzle, Swip, buffer_manager};

// use crate::buffer_manager::Page;

mod page;
mod node;

// const NODE_TYPE_INNER: u64 = 1;
// const NODE_TYPE_LEAF: u64 = 1 << 1;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[repr(u64)]
pub(crate) enum NodeType {
    // 0bxx represents node type
    Inner = 0b0001,
    Leaf  = 0b0010,
}

impl TryFrom<&[u8]> for NodeType {
    type Error = std::array::TryFromSliceError;

    fn try_from(value: &[u8]) -> Result<Self, Self::Error> {
        let res = u64::from_le_bytes(value[0..8].try_into()?);
        if res & (NodeType::Inner as u64) > 0 {
            return Ok(NodeType::Inner);
        } else if res & (NodeType::Leaf as u64) > 0 {
            return Ok(NodeType::Leaf);
        } else {
            panic!("Invalid header. {:?}", &value[0..8]);
        }
    }
}

use crate::buffer_manager::BufferManager;
use page::Page;

use self::page::PageIter;

struct BTree<'a, D: DiskManager> {
    buffer_manager: &'a BufferManager<D>,
    root_page: Page<'a>,
}

impl<'a, D: DiskManager> BTree<'a, D> {
    pub fn from(buffer_manager: &'a BufferManager<D>, root_page: &'a mut [u8]) -> Self {
        let mut root_page = Page::from(root_page);

        let initial_version = buffer_manager.version_boundary();
        unsafe { root_page.init(initial_version) };

        BTree {
            buffer_manager,
            root_page,
        }
    }

    pub fn bootstrap(buffer_manager: &'a BufferManager<D>, data: (Unswizzle, &'a mut [u8])) -> Result<Self, DiskManagerAllocError> {
        let mut root_page = Page::from(data.1);
        let initial_version = buffer_manager.version_boundary();

        unsafe {
            let leaf_data = buffer_manager.new_page()?;
            let mut leaf_page = Page::from(leaf_data.1);
            leaf_page.bootstrap_leaf(leaf_data.0.into(), &[], initial_version);
            root_page.bootstrap_inner(data.0, &[], (b"", leaf_data.0), initial_version);
        }

        Ok(BTree {
            buffer_manager,
            root_page,
        })
    }

    pub fn insert(&self, key: &[u8], value: &[u8]) {
        let res = self.search_to_leaf(key, None)
            .and_then(|mut p| {
                p.write_lock().map_err(|e| e.into())
            });

        let mut lock = match res {
            Ok(l) => l,
            Err(e) => todo!("Handle err case: {:?}", e),
        };

        match lock.insert(key, value) {
            Ok(()) => (),
            Err(e) => {
                todo!("We need to handle error cases! {:?}", e);
            }
        }   
    }

    pub fn get(&self, key: &[u8]) -> Option<Vec<u8>> {
        loop {
            match self.get_(key) {
                Ok(r) => return r,
                Err(SearchError::OptimisticConflict) => continue,
            }
        }
    }

    #[inline]
    fn get_(&self, key: &[u8]) -> Result<Option<Vec<u8>>, SearchError> {
        let page = self.search_to_leaf(key, None)?;

        let pre_version = page.version()?;
        let res = match page.search(key) {
            // We need to clone the data and then validate the version. This
            // ensures that the caller isn't looking at invalide data.
            Ok(v) => Some(v.into()),
            Err(_) => None,
        };

        if pre_version != page.version()? {
            Err(SearchError::OptimisticConflict)
        } else {
            Ok(res)
        }
    }

    // pub fn scan(&self, range: Range<&[u8]>)

    fn search_to_leaf(&self, key: &[u8], inner_page: Option<Page>) -> Result<Page<'a>, SearchError> {
        let page = inner_page.as_ref().unwrap_or(&self.root_page);
        let pre_version = page.version()?;

        // Conversion into the Swip will copy the underlying bytes. Therefore
        // after this we can validate with confidence that we have a valid
        // pointer.
        // What happens if after we do this the referenced page is unswizzled?
        let swip: Swip = page.get_nearby(key).into();

        let post_version = page.version()?;
        if pre_version != post_version {
            return Err(SearchError::OptimisticConflict);
        }

        match swip {
            Swip::Swizzle(s) => {
                // Safety: we know that the root page is an inner page. And
                // we know data from inner pages are Swips
                let data = unsafe { self.buffer_manager.load_swizzled(s) };
                let child = Page::from(data);

                match child.page_type() {
                    NodeType::Inner => {
                        return self.search_to_leaf(key, Some(child));
                    }

                    NodeType::Leaf => {
                        return Ok(child);
                    }
                }
            }

            Swip::Unswizzle(u) => {
                todo!("Need to load Unswizzle from disk");
            }
        }
    }
}


use page::PageIterKey;

struct ScanIterator<'a, T: DiskManager> {
    btree: &'a BTree<'a, T>,
    range: Range<&'a [u8]>,
    leaf: Option<Page<'a>>,
}

impl<'a, T: DiskManager> ScanIterator<'a, T> {
    fn new(btree: &'a BTree<'a, T>, range: Range<&'a [u8]>) -> Self {
        ScanIterator {
            btree,
            range,
            leaf: None,
        }
    }
}

impl<'a, T: DiskManager> Iterator for ScanIterator<'a, T> {
    type Item = (PageIterKey<'a>, &'a [u8]);

    fn next(&mut self) -> Option<Self::Item> {
        unimplemented!()

        // if self.sub_iter.is_some() {
        //     let sub_iter = self.sub_iter.as_mut().unwrap();

        //     // TODO: Need to pass through key as (prefix, slot, suffix). This
        //     // avoids copies.

        //     let next = sub_iter.next();

        //     if next.is_some() {
        //         return next;
        //     } else {
        //         let iter = self.sub_iter.take().unwrap();
        //         let next_key = iter.last_key.unwrap();

        //         // HEAP ALLOC :(
        //         let mut next_page_key = vec![0u8; next_key.len() + 1];
        //         let mut c = 0;
                
        //         next_page_key[c..next_key.0.len()].copy_from_slice(next_key.0);
        //         c += next_key.0.len();

        //         next_page_key[c..c+next_key.1.len()].copy_from_slice(next_key.1);
        //         c += next_key.1.len();

        //         next_page_key[c..c+next_key.2.len()].copy_from_slice(next_key.2);
        //         c += next_key.2.len();

        //         self.lower_fence = next_page_key;
        //     }
        // }

        // loop {
        //     let page = match self.btree.search_to_leaf(&self.lower_fence, None) {
        //         Ok(p) => p,
        //         Err(SearchError::OptimisticConflict) => continue,
        //     };

        //     // let pre_version = match page.version() {
        //     //     Ok(v) => v,
        //     //     Err(_) => continue,
        //     // };

        //     // let range = &self.lower_fence[..]..self.range.end;

        //     // let sub_iter = page.scan(range);
        //     // page.read_lock();
        //     // self.sub_iter = Some(sub_iter);

        //     // let lock = page.read_lock();
        //     // self.sub_iter = Some(page.scan(self.cursor));
        //     // self.sub_iter_lock = lock;

        //     todo!("need to implement page.scan")
        // }

        

        // unimplemented!()
    }
}

/*


for page_iter in btree.scan(b"foo"..b"qux") {
    let mut results = ...;

    for (k, v) in page_iter {
        results.push(...)
    }

    if page_iter.validate() {
        // good to go! we can store the results

        yeild results / push to outer results, etc
    } else {
        // next iteration will contain the same data
        // We need to forget this iteration's results
        std::mem::drop(results);
    }

    // if page_iter panics if validate() wasn't called before drop
}



btree
    .scan(b"foo"..b"qux")
    .



*/



#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SearchError {
    OptimisticConflict
}

impl From<page::VersionLoadError> for SearchError {
    fn from(e: page::VersionLoadError) -> Self {
        match e {
            page::VersionLoadError::Locked => SearchError::OptimisticConflict,
            page::VersionLoadError::Unloaded => SearchError::OptimisticConflict
        }
    }
}



struct BTreeOptions {
    file: PathBuf,
    max_disk_size: usize,
    max_buffer_pool_size: usize,
    base_page_size: usize,

    schema: Vec<ColumnDefinition>,
}

struct ColumnDefinition {
    required: bool,
    has_var_part: bool,
    fixed_bits: usize,
}

fn foo() {
    BTreeOptions {
        file: "./foo.db".into(),
        max_disk_size: n_mib_bytes!(64) as usize,
        max_buffer_pool_size: n_mib_bytes!(16) as usize,
        base_page_size: n_kib_bytes!(64) as usize,

        schema: vec![
            ColumnDefinition {
                required: true,
                has_var_part: false,
                fixed_bits: 64,
            },

            ColumnDefinition {
                required: true,
                has_var_part: false,
                fixed_bits: 32,
            }
        ]
    };
}


struct Column<'a, T> {
    data: &'a [u8],
    marker: std::marker::PhantomData<T>
}

impl<'a, T> Column<'a, T, > {
    fn collect_eq(&self, val: &[u8]) -> Vec<[u8; std::mem::size_of::<T>()]> {
        let width = std::mem::size_of::<T>();
        let num_chunks = self.data.len() / width;
        let mut out = Vec::with_capacity(num_chunks);

        for i in 0..num_chunks {
            let source = &self.data[i*width..i*width+width];
            if source == val {
                let mut arr = [0u8; std::mem::size_of::<T>()];
                arr.copy_from_slice(source);
                out.push(arr);
            }
        }

        out
    }

    fn eq_bitmap(&self, val: &[u8]) -> [u64; 8] {
        let width = std::mem::size_of::<T>();
        let num_chunks = self.data.len() / width;

        let mut out = [0u64; 8];


        for i in 0..num_chunks {
            let source = &self.data[i*width..i*width+width];
            if source == val {
                let out_idx = i / 64;
                let bit_idx = i % 64;

                out[out_idx] |= 1 << bit_idx;
            }
        }

        out
    }
}

#[inline]
fn bitand_segment(left: &[u64; 8], right: &[u64; 8]) -> [u64; 8] {
    let mut out = [0u64; 8];

    for i in 0..8 {
        out[i] = left[i] & right[i];
    }

    out
}

#[test]
fn bar() {
    let col1: Column<u8> = Column { data: b"hello world", marker: Default::default() };
    let col2: Column<u8> = Column { data: b"spicy world", marker: Default::default() };
    let res = col1.collect_eq(b"l");

    eprintln!("Test out: {:?}", res);

    let res = col1.eq_bitmap(b"l");
    eprintln!("Test out: {:b}", res[0]);

    let res2 = col2.eq_bitmap(b"l");
    eprintln!("Test out: {:b}", res2[0]);

    let bit = bitand_segment(&res, &res2);
    eprintln!("Test out: {:b}", bit[0]);
}

// struct ColumnIterator {
// }



/*

Root:
[
    swip to Root Inner
    [Free list]
]

Inner:
[
    [page prefix]
    [sorted(next 4 bytes of ids)]

    [var length tail, swip]
]

Leaf:
[
    [sorted(id range), segment offset]

    segment [
        [mvcc versions]
        [
            [[null bitmap], col1],
            [[null bitmap], col...],
            [[null bitmap], coln]
        ]
    ]

    [var length data]
]

*/

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::FakeDiskManager;
    use crate::buffer_manager::BufferManager;

    #[test]
    fn basic_operations() {
        let disk_manager = FakeDiskManager::default();
        let max_memory = disk_manager.capacity();
        let buffer_manager = BufferManager::new(disk_manager, max_memory);

        let page = unsafe { buffer_manager.new_page().unwrap() };
        let btree = BTree::bootstrap(&buffer_manager, (page.0.into(), page.1)).unwrap();

        btree.insert(b"foo", b"bar");
        btree.insert(b"qux", b"zap");
        btree.insert(b"foobar", b"zzz");
        btree.insert(b"aaa", b"bbb");

        let p0: Page = unsafe { buffer_manager.load_swizzled((0, 0).into()).into() };
        let p1: Page = unsafe { buffer_manager.load_swizzled((1, 0).into()).into() };

        eprintln!("p0: {:#?}", p0);
        eprintln!("p1: {:#?}", p1);
    }
}
