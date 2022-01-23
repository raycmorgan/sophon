use sophon::btree::BTree;
use sophon::buffer_manager::Builder;

#[derive(Debug, PartialEq)]
enum BTreeOp {
    Insert(Vec<u8>, Vec<u8>),
    Get(Vec<u8>),
}

#[test]
fn case_1() {
    let bm = Builder::new()
        .max_memory(1024 * 16 * 100)
        .build();
    let btree = BTree::new(&bm);
    let ops = include!("fuzz_cases/case1.rs.data");

    for op in ops.iter() {
        match op {
            BTreeOp::Insert(k, v) => {
                btree.insert(k, v);
                // inserted.insert(k);
            }

            BTreeOp::Get(k) => {
                if k.len() == 0 {
                    continue;
                }

                let _ = btree.get(k);
                
                // if inserted.contains(k) {
                //     assert!(res.is_some());
                // } else {
                //     assert!(res.is_none());
                // }
            }
        }
    }

    // assert_eq!(t, case);
}