use std::{mem::{size_of, ManuallyDrop}, sync::atomic::AtomicU64, ops::{Range, Index}, fmt::Debug};
use log::debug;
use parking_lot::RawRwLock;
use stackvec::{StackVec, TryCollect};

use crate::buffer_manager::{
    buffer_frame::{USABLE_PAGE_SIZE, ExclusiveGuard, PageGuard},
    swip::Swip
};

pub(crate) const MAX_KEY_LEN: usize = 4096;

#[repr(C)]
#[derive(Default, Clone, Copy, Debug)]
struct Fence {
    pos: u16,
    len: u16,
}

impl Fence {
    fn as_range(&self) -> Range<usize> {
        self.pos as usize .. (self.pos + self.len) as usize
    }
}

#[repr(C)]
#[derive(Default)]
struct NodeHeader {
    db_version: u32,
    pid: u64,
    height: u32,

    // upper_swip: Option<[u8; size_of::<Swip<Node>>()]>,

    lower_fence: Fence,
    upper_fence: Fence,

    // move to BufferFrame
    // version: AtomicU64,
    // latch: RawRwLock,

    is_leaf: bool,
    slot_count: u16,
    space_used: u16,
    space_active: u16,
    prefix_len: u16,
}

#[repr(C, packed)]
#[derive(Clone, Copy, Debug, PartialEq)]
struct Slot {
    key: [u8; 4], // add 6 here?
    data_ptr: u16,
    data_len: u16,
    key_len: u16, // 10 :(
}

const RAW_ALIGN_OFFSET: usize = (size_of::<NodeHeader>() % 8).abs_diff(8) % 8;
const RAW_SIZE: usize = USABLE_PAGE_SIZE - size_of::<NodeHeader>() - RAW_ALIGN_OFFSET;
const SLOTS_CAPACITY: usize = RAW_SIZE / std::mem::size_of::<Slot>();
const DATA_CAPACITY: usize = SLOTS_CAPACITY * size_of::<Slot>();


#[repr(C, packed)]
union NodeContents {
    slots: [Slot; SLOTS_CAPACITY],
    data: [u8; DATA_CAPACITY],
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum InsertError {
    InsufficientSpace,
}

#[repr(C, align(8))]
pub(crate) struct Node {
    header: NodeHeader,
    contents: NodeContents
}

impl Node {
    pub(crate) fn init_header(
        &mut self,
        db_version: u32,
        pid: u64,
        height: u32,

        // upper_swip: Option<[u8; size_of::<Swip<Node>>()]>,
        is_leaf: bool,

        lower_fence: &[u8],
        upper_fence: &[u8],
    ) {
        let prefix_len = lower_fence
            .iter()
            .zip(upper_fence)
            .take_while(|(a, b)| a==b)
            .count();
        let prefix = &lower_fence[0..prefix_len];

        self.header.db_version = db_version;
        self.header.height = height;
        self.header.pid = pid;
        // self.header.upper_swip = upper_swip;
        self.header.is_leaf = is_leaf;
        self.header.slot_count = 0;
        self.header.prefix_len = prefix.len() as u16;

        self.header.space_used = (prefix.len() + lower_fence.len() + upper_fence.len()) as u16;
        self.header.space_active = (prefix.len() + lower_fence.len() + upper_fence.len()) as u16;

        let data_len = self.data().len();
        let mut_data = self.data_mut();

        mut_data[data_len-prefix.len()..].copy_from_slice(prefix);

        let lowerf = Fence {
            len: lower_fence.len() as u16,
            pos: (data_len-prefix.len()-lower_fence.len()) as u16
        };

        mut_data[lowerf.as_range()]
            .copy_from_slice(lower_fence);

        let upperf = Fence {
            len: upper_fence.len() as u16,
            pos: lowerf.pos - upper_fence.len() as u16,
        };

        mut_data[upperf.as_range()]
            .copy_from_slice(upper_fence);

        self.header.lower_fence = lowerf;
        self.header.upper_fence = upperf;

        debug!("[init_header] lower fence: {:?}, upper fence: {:?}, prefix: {:?}",
        lower_fence, upper_fence, prefix);
    }

    #[cfg(test)]
    fn testable() -> Node {
        Node {
            header: NodeHeader::default(),
            contents: NodeContents { data: [0u8; SLOTS_CAPACITY * size_of::<Slot>()] },
        }
    }

    pub(crate) fn insert(&mut self, key: &[u8], value: &[u8]) -> Result<(), InsertError> {
        assert!(key.len() < MAX_KEY_LEN); // return error
        debug_assert!(value.len() < 1024 * 15);
        debug_assert!(&key[..self.header.prefix_len as usize] == self.prefix());

        let key = &key[self.header.prefix_len as usize..];
        let key_len = key.len();
        let key_parts = key_parts(&key);
        let data_len = key_parts.1.len() + value.len();

        if data_len + key_parts.1.len() + size_of::<Slot>() > self.available_space() {
            // TODO: Check to see if we can compact to prevent a split
            return Err(InsertError::InsufficientSpace);
        }

        let data_ptr = self.data().len()
            - self.header.space_used as usize
            - data_len;
        
        let value_ptr = data_ptr+key_parts.1.len();

        self.data_mut()[data_ptr..data_ptr+key_parts.1.len()].copy_from_slice(&key_parts.1);
        self.data_mut()[value_ptr..value_ptr+value.len()].copy_from_slice(&value);

        self.header.space_used += data_len as u16;
        self.header.space_active += data_len as u16;

        let pos = match self.search(&key_parts) {
            Ok(pos) => pos,
            Err(pos) => {
                let slot_count = self.header.slot_count as usize;
                
                if pos < slot_count {
                    unsafe {
                        self.contents.slots.copy_within(pos..slot_count, pos + 1);
                    }
                }

                self.header.slot_count += 1;
                pos
            }
        };

        // if !self.is_leaf() && pos == (self.header.slot_count - 1) {
        //     // Special case: need to move the upper swip into the
        //     // slot, and bump the 
        // }

        unsafe {
            let mut k = [0u8; 4];
            k[0..key_parts.0.len()].copy_from_slice(&key_parts.0[0..]);

            self.contents.slots[pos] = Slot {
                key: k,
                data_ptr: data_ptr as u16,
                data_len: value.len() as u16,
                key_len: key_len as u16,
            }
        }

        Ok(())
    }

    pub(crate) fn get(&self, key: &[u8]) -> Option<&[u8]> {
        debug_assert!(key.len() < MAX_KEY_LEN);
        debug_assert!(&key[..self.header.prefix_len as usize] == self.prefix());

        let key = &key[self.header.prefix_len as usize..];
        let key_parts = key_parts(&key);

        match self.search(&key_parts) {
            Ok(pos) => {
                let slot = self.slots()[pos];
                Some(self.get_data_value(slot))
            }

            Err(_) => None
        }
    }

    pub(crate) fn get_next(&self, key: &[u8]) -> Option<&[u8]> {
        debug_assert!(key.len() < 1024 * 8);
        debug_assert!(&key[..self.header.prefix_len as usize] == self.prefix());

        let key = &key[self.header.prefix_len as usize..];
        let key_parts = key_parts(&key);

        // let idx = self.search(&key_parts).unwrap_or_else(|i| i);

        let pos = match self.search(&key_parts) {
            Ok(pos) => {
                pos

                // let slot = self.slots()[pos];
                // Some(self.get_data_value(slot))
            }

            Err(pos) => {
                pos
            }
        } - 1;

        let slot = self.slots()[pos];
        Some(self.get_data_value(slot))

        // if pos < self.header.slot_count as usize {
        //     let slot = self.slots()[pos];
        //     Some(self.get_data_value(slot))
        // } else {
        //     self.header.upper_swip
        //         .as_ref()
        //         .map(|s| &s[..])
        // }

        // match self.search(&key_parts) {
        //     Ok(pos) => {
        //         let slot = self.slots()[pos];
        //         Some(self.get_data_value(slot))
        //     }

        //     Err(pos) => {
        //         if pos < self.header.slot_count as usize {
        //             let slot = self.slots()[pos];
        //             Some(self.get_data_value(slot))
        //         } else {
        //             self.header.upper_swip
        //                 .as_ref()
        //                 .map(|s| &s[..])
        //         }
        //     }
        // }
    }

    #[inline]
    pub(crate) fn pivot_key_for_split<'b>(&self, key: &[u8], _value_len: usize, dst: &'b mut [u8]) -> &'b [u8] {
        // TODO: Optimize split based on contents and position of data.
        // For example, if the key is greater than all items in the node,
        //   the right page should be mostly empty (optimize for appends).

        let pivot = (self.header.slot_count as usize) / 2;
        let pivot_slot = self.slots()[pivot];
        let temp = &mut [0u8; MAX_KEY_LEN];

        let pivot_key = self.copy_key(pivot_slot, temp);

        // Special case: there is only 1 element and the new key is less than the
        // existing key. We want to make sure the new key is inserted into the
        // left node and the existing item moves to the right node.
        if pivot == 0 && key < pivot_key {
            dst[0..key.len()].copy_from_slice(key);
            return &dst[0..key.len()];
        }

        return self.copy_key(pivot_slot, dst);


        // if key <= pivot_key

        

        // let trimmed_key = &key[self.header.prefix_len as usize..];
        // let key_parts = key_parts(&trimmed_key);
        // let mut split_idx = self.search(&key_parts).unwrap_or_else(|p| p);
        // let count = self.header.slot_count as usize;
        // let mid = count / 2;
        // let slots = self.slots();

        // // Let's just split at the mid point for now
        // split_idx = mid;

        // // if split_idx == count {
        // //     todo!("implement right most entry optimization");
        // // }

        // // let slot_usages = slots.iter().map(|s| {
        // //     (s.data_len + s.key_len) as usize + size_of::<Slot>()
        // // });

        // // slot_usages.windows(2).enumerate().map(|(i, usage)| {
            
        // // });

        // // slots.windows(2).fold()

        // let (left, right) = slots.iter()
        //     .enumerate()
        //     .fold((0,0), |(l,r), (i, slot)| {
        //         if i < split_idx {
        //             (l + (slot.data_len + slot.key_len) as usize, r)
        //         } else {
        //             (l, r + (slot.data_len + slot.key_len) as usize)
        //         }
        //     });

        // loop {
        //     let left_overhead = split_idx * size_of::<Slot>();
        //     let right_overhead = (slots.len() - split_idx) * size_of::<Slot>();
        //     let prefix_len = self.header.prefix_len as usize;
        //     let pivot = slots[split_idx];
        //     let split_key = (
        //         &pivot.key[0..pivot.key_len.min(4) as usize],
        //         self.get_data_key(pivot)
        //     );

        //     println!("key_parts: {:?} < split_key: {:?} => {}", key_parts, split_key, key_parts < split_key);

        //     let out = if mid == 0 && key_parts <= split_key {
        //         dst[0..key.len()].copy_from_slice(key);
        //         &dst[0..key.len()]
        //     } else {
        //         self.copy_key(pivot, dst)
        //     };

        //     return out;


        //     let capacity = if key_parts < split_key {
        //         DATA_CAPACITY - left_overhead + left
        //     } else {
        //         DATA_CAPACITY - right_overhead + right
        //     };

        //     if capacity > size_of::<Slot>() + key_parts.1.len() + value.len() {
        //         // return KeyChunks::new(&[self.prefix(), &split_key.0, split_key.1]);
                
        //         // let mut v = Vec::with_capacity(prefix_len + 4 + split_key.1.len());
        //         // v[0..prefix_len].copy_from_slice(self.prefix());
        //         // v[prefix_len..prefix_len+4].copy_from_slice(&split_key.0);
        //         // v[prefix_len+4..].copy_from_slice(split_key.1);

        //         // // TODO: Just return the parts as references

        //         // return v;

        //         // use std::io::{Write, Cursor};

        //         // let mut c = Cursor::new(dst);

        //         // c.write(self.prefix()).expect("infallible");
        //         // c.write(&split_key.0).expect("infallible");
        //         // c.write(&split_key.1).expect("infallible");

        //         // let dst = c.into_inner();

        //         // // dst[0..prefix_len].copy_from_slice(self.prefix());
        //         // // dst[prefix_len..prefix_len+4].copy_from_slice(&split_key.0);
        //         // // dst[prefix_len+4..].copy_from_slice(split_key.1);

        //         // let len = prefix_len + 4 + split_key.1.len();

        //         // if 
        //         let out = self.copy_key(pivot, dst);

        //         println!("dst: {:?}", out);
        //         return &out;
        //     } else {
        //         todo!(r#"
        //             We don't have enough space to perform an insert. 
        //             Shift pivot key, multi-split, use large page?
        //         "#);

        //         // We don't have enough space to perform the insert, shift the pivot key

        //         // [1-4, 3-4, 7-4, 8-3]
        //         // insert: 6-10

        //         // [1, 3, 6] [7, 8]
        //         // [1, 3] [6, 7, 8]
        //         // [1], [3, 6, 7, 8]
        //         // [1, 3, 6, 7] [8]
        //     }
        // }
    }

    #[inline]
    pub(crate) fn split(&mut self, right: &mut PageGuard<Node>, pivot_key: &[u8]) {
        // let key = &pivot_key[self.header.prefix_len as usize..];
        // let key_parts = key_parts(&key);
        // let pivot = self.search(&key_parts).unwrap_or_else(|i| i);

        // reinsert all slots >= pivot into right
        // update self.slot_count to remove tail slots >= pivot
        // compact self

        #[cfg(debug_assertions)]
        let slot_count = self.header.slot_count;

        let pivot = self.search(&key_parts(&pivot_key))
            .map(|i| i)
            .unwrap_or_else(|e| e);
        let pivot_slot = self.slots()[pivot];

        let mut temp = [0u8; MAX_KEY_LEN];
        let split_key = self.copy_key(pivot_slot, &mut temp);

        debug!("[split] Splitting non-root: {} at {:?}:{} | pivot_key: {:?}", self.header.pid, split_key, pivot, pivot_key);
        debug!("[split] Lower fence: {:?}, Upper fence: {:?}", self.lower_fence(), self.upper_fence());

        let pid = right.pid();
        right.init_header(
            1,
            pid,
            self.header.height,
            // self.header.upper_swip, // todo: steal from left
            self.is_leaf(),
            &split_key,
            self.upper_fence(),
        );

        debug!("[split] Created Right Peer {}, is_leaf: {}", pid, right.is_leaf());

        let mut i = pivot + 1;

        // if !self.is_leaf() {
        //     // let value = self.get_data_value(pivot_slot);
        //     // self.header.upper_swip = Some(value.try_into().expect("len match"));
        // }

        let slots = self.slots();
        while let Some(slot) = slots.get(i) {
            let key = self.copy_key(*slot, &mut temp);
            let value = self.get_data_value(*slot);

            debug!("[split] Insert right: {:?}=>({})", key, value.len());
            right.insert(key, value).expect("infallible");

            i += 1;
        }
        
        self.header.slot_count = (pivot + 1) as u16;
        self.compact(&mut temp, Some(&right.lower_fence()));

        debug!("Post compact: prefix=>{:?}", self.prefix());

        #[cfg(debug_assertions)] {
            debug_assert_eq!(slot_count, self.header.slot_count + right.header.slot_count);
            debug_assert_eq!(self.upper_fence(), right.lower_fence());

            let slots = self.slots();
            let min_key = self.copy_key(slots[0], &mut temp);
            debug_assert!(self.lower_fence() <= min_key, "{:?} <= {:?}", self.lower_fence(), min_key);
            let max_key = self.copy_key(*slots.last().expect("infallible"), &mut temp);
            if self.upper_fence() != b"" { // special case upper_fence
                debug_assert!(self.upper_fence() >= max_key, "{:?} < {:?}", self.upper_fence(), max_key);
            }

            let slots = right.slots();
            let min_key = right.copy_key(slots[0], &mut temp);
            debug_assert!(right.lower_fence() <= min_key, "{:?} <= {:?}", right.lower_fence(), min_key);
            let max_key = right.copy_key(*slots.last().expect("infallible"), &mut temp);

            if right.upper_fence() != b"" { // special case upper_fence
                debug_assert!(right.upper_fence() >= max_key, "{:?} >= {:?}", right.upper_fence(), max_key);
            }
        }
    }

    #[inline]
    fn compact(&mut self, tmp_buffer: &mut [u8], upper_fence: Option<&[u8]>) {
        // let tmp = [0u8; DATA_CAPACITY];

        let mut tmp = Node {
            header: Default::default(),
            contents: NodeContents { data: [0u8; DATA_CAPACITY] }
        };

        tmp.init_header(
            1, 
            0,
            0,
            // None,
            self.is_leaf(),
            self.lower_fence(),
            upper_fence.unwrap_or(self.upper_fence())
        );

        for slot in self.slots() {
            let key = self.copy_key(*slot, tmp_buffer);
            let value = self.get_data_value(*slot);
            tmp.insert(key, value).expect("infallible");
        }

        self.data_mut().copy_from_slice(tmp.data());
        self.header.lower_fence = tmp.header.lower_fence;
        self.header.upper_fence = tmp.header.upper_fence;
        self.header.prefix_len = tmp.header.prefix_len;
        self.header.space_active = tmp.header.space_active;
        self.header.space_used = tmp.header.space_used;
    }

    #[inline]
    pub(crate) fn root_split(&mut self, left: &mut PageGuard<Node>, right: &mut PageGuard<Node>, pivot_key: &[u8]) {
        // let key = &pivot_key[self.header.prefix_len as usize..];
        // let key_parts = key_parts(&key);
        // let pivot = self.search(&key_parts).unwrap_or_else(|i| i);

        // reinsert all slots >= pivot into right
        // update self.slot_count to remove tail slots >= pivot
        // compact self

        debug!("Splitting root. Slot Count: {}, Pivot: {:?}", self.slots().len(), pivot_key);

        let mut temp = [0u8; MAX_KEY_LEN];

        let l_pid = left.pid();
        let r_pid = right.pid();

        let pivot = self
            .search(&key_parts(&pivot_key))
            .unwrap_or_else(|e| e);
        let pivot_slot = self.slots()[pivot];
        let split_key = self.copy_key(pivot_slot, &mut temp);


        // [a, b, d, e, t].z;
        // [a, b, d] [e, t, z]
        // [a, b].d  [e, t].z

        left.init_header(
            1,
            l_pid,
            self.header.height,
            // None, // point to right
            self.is_leaf(),
            self.lower_fence(),
            &split_key,
        );

        right.init_header(
            1,
            r_pid,
            self.header.height,
            // self.header.upper_swip,
            self.is_leaf(),
            &split_key,
            self.upper_fence(),
        );

        

        for (i, slot) in self.slots().iter().enumerate() {
            // Special case: if left/right are inner nodes, we will need to
            // set the value to the upper_swip
            // if !left.is_leaf() && i == pivot {
            //     left.header.upper_swip = Some(right.swip_bytes());
            //     continue;
            // }

            let key = self.copy_key(*slot, &mut temp);
            let value = self.get_data_value(*slot);

            if i <= pivot {
                debug!("Insert left: {:?}, v({})", key, value.len());
                left.insert(key, value).expect("infallible");
            } else {
                debug!("Insert right: {:?}, v({})", key, value.len());
                right.insert(key, value).expect("infallible");
            }
        }

        debug!("Left pid {}, is_leaf: {}", l_pid, left.is_leaf());
        debug!("Right pid {}, is_leaf: {}", r_pid, right.is_leaf());

        // TODO: clear out self!
        self.header.slot_count = 0;
        self.header.space_active = 0;
        self.header.space_used = 0;
        self.header.is_leaf = false;
        self.header.height += 1;

        let data = self.data_mut();
        data.fill(0);

        // insert left and right
        // self.header.upper_swip = Some(right.swip_bytes());
        self.insert(left.lower_fence(), &left.swip_bytes()).expect("infallible");
        self.insert(right.lower_fence(), &right.swip_bytes()).expect("infallible");

        // println!("Parent, header.upper_swip: {:?}", self.header.upper_swip);
        debug!("Parent, insert: {:?}", pivot_key);

        #[cfg(debug_assertions)] {
            for (name, n) in &[
                ("parent", &*self),
                ("left", left.data_structure()),
                ("right", right.data_structure())
            ] {
                let slots = n.slots();

                if slots.len() == 0 {
                    continue;
                }

                
                let min_key = n.copy_key(slots[0], &mut temp);
                debug_assert!(n.lower_fence() <= min_key, "{}: {:?} <= {:?}", name, n.lower_fence(), min_key);
                
                let max_key = n.copy_key(*slots.last().expect("infallible"), &mut temp);

                if n.upper_fence() != b"" { // special case upper_fence
                    debug_assert!(n.upper_fence() >= max_key, "{}: {:?} >= {:?}", name, n.upper_fence(), max_key);
                }
            }


            

            // let slots = right.slots();
            // let min_key = right.copy_key(slots[0], &mut temp);
            // debug_assert!(right.lower_fence() < min_key, "{:?} < {:?}", right.lower_fence(), min_key);
            // let max_key = right.copy_key(*slots.last().expect("infallible"), &mut temp);
            // debug_assert!(right.upper_fence() >= max_key, "{:?} < {:?}", right.upper_fence(), max_key);
        }
    }

    #[inline]
    pub(crate) fn has_space_for(&self, key_len: usize, val_len: usize) -> bool {
        val_len + key_len.saturating_sub(4) + size_of::<Slot>() <= self.available_space()
    }

    #[inline]
    fn search(&self, key_parts: &KeyParts) -> Result<usize, usize> {
        let slot_key = key_parts.0;

        // TODO(safety): There is a case in which we're optimistically reading
        // data within this node and a unlikely (but possible) structural
        // causes slot length to change which would cause the  interal pointers
        // to be incorrect (outside of the bounds)
    
        #[cfg(debug_assertions)] {
            let mut str = String::with_capacity(DATA_CAPACITY);

            str.push_str("Slots: <");

            for slot in self.slots() {
                if self.is_leaf() {
                    // str.push_str(slot.key);
                    // str.push_str("=>{}, ");
                    // str.push_str(&format!("{:?}=>{}, ", slot.key, slot.data_len));
                } else {
                    // str.push_str(&format!("{:?}=>{:?}, ", 
                    //     slot.key, 
                    //     u64::from_ne_bytes(self.get_data_value(*slot).try_into().unwrap())
                    // ));
                }
            }
            str.push_str(" >");
            // print!("{:?}", self.header.upper_swip.map(|s| u64::from_ne_bytes(s)));
            debug!("{}", str);
        }

        self.slots().binary_search_by(|s| {
            use std::cmp::Ordering;

            let sk = &s.key[0..s.key_len.min(4) as usize];

            match sk.cmp(slot_key) {
                Ordering::Greater => Ordering::Greater,
                Ordering::Less => Ordering::Less,
                Ordering::Equal => {
                    // if s.key_len as usize == slot_key.len() && key_parts.1.len() == 0 {
                    //     return Ordering::Equal;
                    // }

                    let suffix = self.get_data_key(*s);
                    suffix.cmp(&key_parts.1)
                }
            }
        })
    }

    #[inline]
    fn data(&self) -> &[u8] {
        // let offset = self.header.slot_count as usize * size_of::<Slot>();

        // Safety: We know the contents can always be considered an &[u8].
        // ~We're additionally ensuring not to overwrite any slots.~
        unsafe {
            &self.contents.data[..] //offset..]
        }
    }

    #[inline]
    fn data_mut(&mut self) -> &mut [u8] {
        // let offset = self.header.slot_count as usize * size_of::<Slot>();

        // Safety: We know the contents can always be considered an &[u8].
        // ~We're additionally ensuring not to overwrite any slots.~
        unsafe {
            &mut self.contents.data[..]
        }
    }

    #[inline]
    fn slots(&self) -> &[Slot] {
        unsafe {
            &self.contents.slots[0..self.header.slot_count as usize]
        }
    }

    #[inline]
    fn available_space(&self) -> usize {
        size_of::<NodeContents>()
         - self.header.slot_count as usize * size_of::<Slot>()
         - self.header.space_used as usize
    }

    #[inline]
    fn prefix(&self) -> &[u8] {
        let data = self.data();
        &data[data.len()-self.header.prefix_len as usize..data.len()]
    }

    #[inline]
    fn get_data_key(&self, slot: Slot) -> &[u8] {
        let dp = slot.data_ptr as usize;
        let key_len = slot.key_len.saturating_sub(4) as usize;
        &self.data()[dp..dp+key_len]
    }

    #[inline]
    fn get_data_value(&self, slot: Slot) -> &[u8] {
        let dp = slot.data_ptr as usize;
        let key_len = slot.key_len.saturating_sub(4) as usize;
        let pos = dp + key_len;
        &self.data()[pos..pos+slot.data_len as usize]
    }

    #[inline]
    pub(crate) fn lower_fence(&self) -> &[u8] {
        &self.data()[self.header.lower_fence.as_range()]
    }

    #[inline]
    pub(crate) fn upper_fence(&self) -> &[u8] {
        &self.data()[self.header.upper_fence.as_range()]
    }

    #[inline]
    pub (crate) fn is_leaf(&self) -> bool {
        self.header.is_leaf
    }

    #[inline]
    fn copy_key<'b>(&self, slot: Slot, dst: &'b mut [u8]) -> &'b [u8] {
        use std::io::{Write, Cursor};

        let dst_len = dst.len();
        let len = self.prefix().len() + slot.key_len as usize;
        debug_assert!(dst_len >= len);

        let mut c = Cursor::new(dst);

        c.write(self.prefix()).expect("infallible");
        c.write(&slot.key[0..slot.key_len.min(4) as usize]).expect("infallible");
        c.write(self.get_data_key(slot)).expect("infallible");

        let dst = c.into_inner();
        &dst[0..len]
    }
}

type KeyParts<'a> = (&'a [u8], &'a [u8]);

#[inline]
fn slot_key(key: &[u8]) -> [u8; 4] {
    [
        *key.get(0).unwrap_or(&0),
        *key.get(1).unwrap_or(&0),
        *key.get(2).unwrap_or(&0),
        *key.get(3).unwrap_or(&0),
    ]
}

#[inline]
fn key_parts(key: &[u8]) -> KeyParts {
    (
        &key[0..key.len().min(4)], // slot_key(&key),
        if key.len() <= 4 { b"" } else { &key[4..] }
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_search() {
        let node = Node::testable();
        assert_eq!(Err(0), node.search(&key_parts(b"foo")));
    }

    #[test]
    fn init_header() {
        let mut node = Node::testable();
        node.init_header(
            1, 
            1,
            0,
            // None,
            false,
            b"abc",
            b"xyz",
        );

        assert_eq!(b"abc", node.lower_fence());
        assert_eq!(b"xyz", node.upper_fence());
    }

    #[test]
    fn insert() {
        let mut node = Node::testable();
        node.insert(b"foo", b"bar").unwrap();
        node.insert(b"aaa", b"dux").unwrap();
        node.insert(b"zzzaaa", b"yyyy").unwrap();

        let slot1 = node.slots()[0];
        let slot2 = node.slots()[1];
        assert_eq!(b"", node.get_data_key(slot1));
        assert!(slot1.key < slot2.key, "{:?} < {:?}", slot1.key, slot2.key);

        let slot = node.slots()[2];
        assert_eq!(b"aa", node.get_data_key(slot));
        assert_eq!(b"yyyy", node.get_data_value(slot));

        assert_eq!(b"bar", node.get(b"foo").unwrap());
        assert_eq!(None, node.get(b"foo\0"));
        assert_eq!(b"dux", node.get(b"aaa").unwrap());
        assert_eq!(b"yyyy", node.get(b"zzzaaa").unwrap());
        assert_eq!(None, node.get(b"aa"));

        assert_eq!(b"yyyy", node.get_next(b"zzzaa").unwrap());
        assert_eq!(None, node.get_next(b"zzzz"));
    }

    #[test]
    fn replace() {
        let mut node = Node::testable();

        node.insert(b"foo", b"bar").unwrap();
        assert_eq!(b"bar", node.get(b"foo").unwrap());

        node.insert(b"foo", b"baz").unwrap();
        assert_eq!(b"baz", node.get(b"foo").unwrap());
    }
}








// ----

#[repr(C)]
struct LargeNode {
    raw_len: usize,
    // raw: ?? I literally have a chunk of memory here..
    raw: [u8; 1], // This actually has space in it, just using it as a marker
}

impl LargeNode {
    fn raw(&mut self) -> &mut [u8] {
        let ptr = self.raw.as_mut_ptr();
        unsafe { 
            std::slice::from_raw_parts_mut(ptr, self.raw_len)
        }
    }

    fn bump(&mut self) {
        self.raw()[0..4].copy_from_slice(&[0u8; 4]);
    }
}

// fn foo() { 
//     let mut swip: Swip<Node> = Swip::new(1);
//     let mut guard = ExclusiveGuard::new(&mut swip);
//     let dts = guard.data_structure();
//     let _s: Swip<LargeNode> = Swip::new(2);
// }
