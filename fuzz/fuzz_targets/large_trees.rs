#![no_main]
use libfuzzer_sys::fuzz_target;
use sophon::buffer_manager::Builder;
use sophon::btree::BTree;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

fuzz_target!(|data: u64| {
    // env_logger::init();
    do_it(data);
});

fn do_it(seed: u64) {
    const MAX_KEY_LEN: usize = 1024;
    const MAX_VAL_LEN: usize = 1024 * 2;

    // let mut seed_data = [0u8; 32];
    // seed_data[0..16].copy_from_slice(&seed.0.to_ne_bytes());
    // seed_data[16..32].copy_from_slice(&seed.1.to_ne_bytes());

    let mut rng = StdRng::seed_from_u64(seed);

    let op_count = rng.gen_range(0..100000usize);
    // println!("Seed: {:?}, Ops: {}", seed, op_count);

    let bm = Builder::new()
        .max_memory(4096 * 4096 * 5000)
        .build();
    let btree = BTree::new(&bm);

    let mut key = [0u8; MAX_KEY_LEN];
    let mut val = [0u8; MAX_VAL_LEN];

    for _ in 0..op_count {
        let key_len = rng.gen_range(1..MAX_KEY_LEN);
        let val_len = rng.gen_range(1..MAX_VAL_LEN);

        rng.fill(&mut key[0..key_len]);
        rng.fill(&mut val[0..val_len]);

        btree.insert(&key[0..key_len], &val[0..val_len]);
    }
}
