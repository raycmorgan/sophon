use std::path::PathBuf;
use byte_unit::{n_kib_bytes, n_mib_bytes};

// use crate::buffer_manager::Page;

mod inner_page;

// const NODE_TYPE_INNER: u64 = 1;
// const NODE_TYPE_LEAF: u64 = 1 << 1;

#[repr(u64)]
enum NodeType {
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

trait BTree {
    fn new();
    fn from_file();
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