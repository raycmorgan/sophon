#![no_main]
use libfuzzer_sys::fuzz_target;
use libfuzzer_sys::arbitrary::{self, Arbitrary};
use sophon::buffer_manager::Builder;
use sophon::btree::BTree;
use std::collections::HashSet;

#[derive(Arbitrary, Debug)]
enum BTreeOp<'a> {
    Insert(&'a [u8], &'a [u8]),
    Get(&'a [u8])
}

fuzz_target!(|ops: Vec<BTreeOp>| {
    // fuzzed code goes here
    // println!("ops: {:?}", ops);

    let mut inserted: HashSet<&[u8]> = HashSet::with_capacity(ops.len());

    let bm = Builder::new()
        .max_memory(4096 * 4096 * 5000)
        .build();
    let btree = BTree::new(&bm);

    for op in ops.iter() {
        match op {
            BTreeOp::Insert(k, v) => {
                btree.insert(k, v);

                if k.len() > 0 && v.len() > 0 {
                    inserted.insert(k);
                }
            }

            BTreeOp::Get(k) => {
                if k.len() == 0 {
                    continue;
                }

                let res = btree.get(k);
                
                if inserted.contains(k) {
                    assert!(res.is_some(), "key: {:?}", k);
                } else {
                    assert!(res.is_none(), "key: {:?}", k);
                }
            }
        }
    }
});
