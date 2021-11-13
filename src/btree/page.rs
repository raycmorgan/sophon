use crate::{Swip, Unswizzle, Swizzle};
use std::sync::atomic::{AtomicU64, Ordering};
use std::mem::transmute;
use std::ops::Range;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FindSlot {
    Insert(usize),
    Replace(usize),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SearchPos {
    Exact(usize),
    Nonexact(usize),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum VersionLoadError {
    Unloaded,
    Locked,
}


pub struct Page<'a> {
    data: &'a mut [u8],
}

// Header [64]: 
//      [page head: 0..8]
const HPOS_PAGE_INFO: Range<usize> = 0..8;
//      [page unswizzle: 8..16]
const HPOS_PAGE_SWIP: Range<usize> = 8..16;
//      [version: 16..24]
const HPOS_VERSION: Range<usize> = 16..24;
//      [rwlock: 24..32]
const HPOS_RWLOCK: Range<usize> = 24..32;
//      [slots used: 32..36]
const HPOS_SLOT_LEN: Range<usize> = 32..36;
//      [next suffix offset: 36..40]
const HPOS_SUFFIX_POINTER: Range<usize> = 36..40;
//      [freed suffix len: 40..44]
const HPOS_FREED_SUFFIX_LEN: Range<usize> = 36..40;
//      [[common prefix len: 1][prefix: N, max 19] 44..64]
const HPOS_PREFIX: Range<usize> = 44..64;
const MAX_PREFIX_LEN: usize = HPOS_PREFIX.end - HPOS_PREFIX.start - 1;

const SLOT_WIDTH: usize = 8;
//
// Slots: [prefix: 4][suffix offset: 4][page pointer: 8]
// Tails: [len: 2][suffix: len] or [len: 2][next free offset: 4]


// Slots: [prefix: 4][overflow ptr: 4]
// Overflow: [suffix_len: 2][suffix: len][data_len: 2][data: data_len]

pub(crate) type ReadGuard<'a> = parking_lot::lock_api::RwLockReadGuard<'a, parking_lot::RawRwLock, ()>;
pub(crate) type WriteGuard<'a> = parking_lot::lock_api::RwLockWriteGuard<'a, parking_lot::RawRwLock, ()>;

use parking_lot::RwLock;

use super::NodeType;

impl<'a> From<&'a mut [u8]> for Page<'a> {
    fn from(data: &'a mut [u8]) -> Self {
        Page::from(data)
    }
}

impl<'a> Clone for Page<'a> {
    fn clone(&self) -> Self {
        let data_ptr = self.data.as_ptr();

        Page {
            data: unsafe {
                std::slice::from_raw_parts_mut(data_ptr as *mut u8, self.data.len())
            }
        }
    }
}

impl<'a> Page<'a> {
    pub fn from(data: &'a mut [u8]) -> Self {
        Page { 
            data: data
        }
    }

    /// Safety: This method assumes only one reference to the underlying data
    /// when called. Given this cannot be guaranteed at this layer, caller is
    /// responsible for upholding that guarantee.
    /// 
    /// This function is to be called when the page is first allocated.
    pub(crate) unsafe fn bootstrap_inner(&mut self, unswizzle: Unswizzle, prefix: &[u8], initial_page: (&[u8], Swizzle), initial_version: u64) {
        debug_assert!(prefix.len() < MAX_PREFIX_LEN);

        let header = super::NodeType::Inner as u64;

        self.data[HPOS_PAGE_INFO].copy_from_slice(&header.to_le_bytes());
        self.data[HPOS_PAGE_SWIP].copy_from_slice(&unswizzle.as_u64().to_le_bytes());
        self.data[16..64].copy_from_slice(&[0u8; 48]);

        self.data[HPOS_PREFIX][0] = prefix.len() as u8;
        self.data[HPOS_PREFIX][1..1+prefix.len()].copy_from_slice(prefix);

        self.init(initial_version);
        let mut lock = self.write_lock().expect("ownership");

        // TODO: Handle error case
        lock.insert(initial_page.0, &(initial_page.1).as_u64().to_le_bytes()).unwrap();
    }

    /// Safety: This method assumes only one reference to the underlying data
    /// when called. Given this cannot be guaranteed at this layer, caller is
    /// responsible for upholding that guarantee.
    /// 
    /// This function is to be called when the page is first allocated.
    pub(crate) unsafe fn bootstrap_leaf(&mut self, unswizzle: Unswizzle, prefix: &[u8], initial_version: u64) {
        debug_assert!(prefix.len() < MAX_PREFIX_LEN);

        let header = super::NodeType::Leaf as u64;

        self.data[HPOS_PAGE_INFO].copy_from_slice(&header.to_le_bytes());
        self.data[HPOS_PAGE_SWIP].copy_from_slice(&unswizzle.as_u64().to_le_bytes());
        self.data[16..64].copy_from_slice(&[0u8; 48]);

        self.data[HPOS_PREFIX][0] = prefix.len() as u8;
        self.data[HPOS_PREFIX][1..1+prefix.len()].copy_from_slice(prefix);

        self.init(initial_version);
    }

    /// Safety: This method assumes only one reference to the underlying data
    /// when called. Given this cannot be guaranteed at this layer, caller is
    /// responsible for upholding that guarantee.
    /// 
    /// This function is to be called whenever the page is loaded from disk.
    pub(crate) unsafe fn init(&mut self, initial_version: u64) {
        // Instead of bootstrapping the version to 1, we have the
        // BufferManager provide a version that is larger than any version
        // it has observed being unswizzled. This ensures that
        // if the memory is recycled, it will always show a larger version than
        // previous. This also shouldn't have too much overhead, as this only
        // happens when moving things between hot and cold, which is a non-free
        // operation.
        //
        // This creates a nice separation of concerns between the
        // BufferManager and the btree. The btree bootstrapping'ing to 1 implies
        // requires the version to be stored on disk and that the memory segment
        // is only used for that page id (which is true in the mmap'ed Umbra
        // manager but is not true for standard LeanStore)

        // assert version is 0, though that doesn't do all that much
        self.data[HPOS_VERSION].copy_from_slice(&(initial_version + 1).to_le_bytes());
        self.data[HPOS_RWLOCK].copy_from_slice(&[0u8; 8]);
    }

    #[inline]
    pub(crate) fn page_type(&self) -> NodeType {
        (&self.data[HPOS_PAGE_INFO]).try_into().unwrap()
    }

    #[inline]
    pub fn read_lock(&self) -> ReadGuard {
        self.get_lock().read()
    }

    #[inline]
    pub(crate) fn write_lock(&mut self) -> Result<WriteLock<'a>, VersionLoadError> {
        // Warning: This is super not safe. The lifetime here allows you to drop
        // the page before you drop the lock. This results in explosions.
        // Fix option 1: Figure out how to model the lifetimes appropriately
        // Fix option 2: Mint a new Page that the lock owns. This is safe as once
        //   we've acquired a lock on a Page, it cannot be unloaded until the lock
        //   is released.

        let guard = self.get_lock().write();
        self.lock_version(&guard)?;

        Ok(WriteLock {
            guard,
            page: self.clone(),
        })
    }

    #[inline]
    pub(crate) fn is_active(&self) -> bool {
        &self.data[HPOS_PAGE_INFO] != [0u8; 8]
    }

    #[inline]
    fn load_version(&self) -> u64 {
        self.get_version().load(Ordering::Acquire)
    }

    #[inline]
    pub(crate) fn version(&self) -> Result<u64, VersionLoadError> {
        let version = self.load_version();

        if version == 0 {
            Err(VersionLoadError::Unloaded)
        } else if version & 1 << 63 > 0 {
            Err(VersionLoadError::Locked)
        } else {
            Ok(version)
        }
    }

    pub(crate) fn insert(&mut self, guard: &WriteGuard<'_>, key: &[u8], value: &[u8]) -> Result<(), InsertPageError> {
        // 0. Do we have space?
        // 1. Get and update tail pos
        // 2. Insert 4 byte key in header, offset of tail pos
        // 2. Incr key count
        // 4. Insert at tail pos: suffix
        eprintln!("data len: {}", self.data.len());
        let prefix = read_u8_len_bytes(&self.data[HPOS_PREFIX]);
        if &key[..prefix.len()] != prefix {
            return Err(InsertPageError::IncorrectPrefix);
        }

        let trimmed_key = &key[prefix.len()..];
        let slots_len = read_u32(&self.data[HPOS_SLOT_LEN]) as usize;
        let suffix_pointer = read_u32(&self.data[HPOS_SUFFIX_POINTER]) as usize;

        // If the data consumed by adding an additional slot + the page header
        // overlaps the suffix_pointer, we don't have any more readily available
        // space on this page.
        let current_slot_len = slots_len * SLOT_WIDTH;
        let header_len = 64usize;
        // let additional_slot_requirement
        let current_suffix_storage = self.data.len() - suffix_pointer;

        let key_overflow_len = trimmed_key.len().checked_sub(4).unwrap_or(0);
        let overflow_width = 4 + key_overflow_len + 4 + value.len();

        if current_slot_len + header_len + SLOT_WIDTH + overflow_width > current_suffix_storage {
            return Err(InsertPageError::PageFull);
        }

        let overflow_pos = self.fetch_update_suffix_pos(guard, overflow_width) as usize;
        let overflow = self.data.len() - overflow_pos;

        self.data[overflow..overflow+4].copy_from_slice(&(key_overflow_len as u32).to_le_bytes());

        if key_overflow_len > 0 {
            self.data[overflow+4..overflow+4+key_overflow_len].copy_from_slice(&trimmed_key[4..]);
        }

        let value_ptr = overflow+4+key_overflow_len;
        self.data[value_ptr..value_ptr+4].copy_from_slice(&(value.len() as u32).to_le_bytes());
        self.data[value_ptr+4..value_ptr+4+value.len()].copy_from_slice(&value);

        self.slot_insert(trimmed_key, overflow as u32);

        return Ok(());
    }

    #[inline]
    pub(crate) fn get_nearby(&self, key: &[u8]) -> &[u8] {
        self.search(key).unwrap_or_else(|d| d)
    }

    pub(crate) fn search(&self, key: &[u8]) -> Result<&[u8], &[u8]> {
        #[cfg(debug_assertions)] {
            let prefix = read_u8_len_bytes(&self.data[HPOS_PREFIX]);
            debug_assert!(&key[..prefix.len()] == prefix);
        }

        let res = self.search_slots(key);
        let exact = res.is_ok();

        let pos = res.unwrap_or_else(|i| i - 1);
        let slot = &self.data[self.slot_range_for_idx(pos)];

        let overflow = read_u32(&slot[4..8]) as usize;
        let suffix_len = read_u32(&self.data[overflow..overflow+4]) as usize;

        let data = read_u32_len_bytes(&self.data[overflow+4+suffix_len..]);

        if exact { Ok(data) } else { Err(data) }
    }

    pub(crate) fn scan(self, range: &Range<&'a [u8]>) -> PageIter<'a> {
        PageIter::new(self)
    }

    #[inline]
    fn ptr_len(&self) -> usize {
        if self.data.len() <= 65536 { 2 } else { 4 }
    }

    #[inline]
    fn slot_range_for_idx(&self, idx: usize) -> Range<usize> {
        idx*SLOT_WIDTH+64..idx*SLOT_WIDTH+64+SLOT_WIDTH
    }

    #[inline]
    fn search_position(&self, key: &[u8]) -> usize {
        // Non-exact means we overshot the position, so need to
        // move back one.
        self.search_slots(key).unwrap_or_else(|i| i - 1)
    }

    #[inline]
    fn find_slot_pos(&self, key: &[u8]) -> FindSlot {
        match self.search_slots(key) {
            Ok(i) => FindSlot::Replace(i),
            Err(i) => FindSlot::Insert(i),
        }
    }

    fn search_slots(&self, key: &[u8]) -> Result<usize, usize> {
        let prefix_len = self.data[HPOS_PREFIX.start] as usize;
        let key = &key[prefix_len..];
        let slot_key = &pad_right_slice::<4>(&key);

        let slots_len = read_u32(&self.data[HPOS_SLOT_LEN]) as usize;
        let slots_data = &self.data[64..64+(SLOT_WIDTH * slots_len)];

        // We are confident that our data is a multiple of SLOT_WIDTH, as that
        // is both the definition of our data structure and we just fetched
        // a multiple of it above to create slots_data.
        let slots = unsafe { slots_data.as_chunks_unchecked::<SLOT_WIDTH>() };

        slots.binary_search_by(|slot| {
            use std::cmp::Ordering;

            if &slot_key[..] > &slot[0..4] {
                return Ordering::Less;
            }

            if &slot_key[..] == &slot[0..4] {
                let suffix_offset = read_u32(&slot[4..8]) as usize;
                let suffix_slice = read_u32_len_bytes(&self.data[suffix_offset..]);

                if key.len() <= 4 && suffix_slice.len() == 0 {
                    return Ordering::Equal;
                } else {
                    return suffix_slice.cmp(&key[4..]);
                }
            }

            return Ordering::Greater;
        })
    }

    // ---- //
    // ---- //

    #[inline]
    fn get_lock(&self) -> &'static RwLock<()> {
        // Safety: The idea is that we cast a 8 byte RwLock into the 24th byte
        // offset of the page. This transmute goes against all that is holy
        // in Rust, and honestly might break things due to UB. Said another
        // way -- the idea is sane, but it's not actually safe.
        // Add to that the 'static lifetime :horror: to allow accessing the
        // page mutably after while holding the lock. :more_horror:
        unsafe {
            transmute(self.data.as_ptr().offset(24))
        }
    }

    #[inline]
    fn get_version(&self) -> &AtomicU64 {
        unsafe {
            transmute(self.data.as_ptr().offset(16))
        }
    }
    
    #[inline]
    fn incr_version(&self, _guard: &WriteGuard<'_>) {
        self.get_version().fetch_add(1, Ordering::AcqRel);
    }

    #[inline]
    fn lock_version(&self, _guard: &WriteGuard<'_>) -> Result<(), VersionLoadError> {
        // debug_assert!(self.load_version());

        let version = self.get_version();
        let mut cur = version.load(Ordering::Relaxed);
        // self.get_version().fetch_or(1 << 63, Ordering::AcqRel);

        loop {
            if cur == 0 {
                return Err(VersionLoadError::Unloaded);
            }

            match version.compare_exchange_weak(cur, cur | 1 << 63, Ordering::AcqRel, Ordering::Acquire) {
                Ok(_) => return Ok(()),
                Err(c) => cur = c,
            }
        }
    }

    #[inline]
    fn unlock_version(&self, guard: &WriteGuard<'_>) {
        // We can get rid of the double atomic operations here either via
        // a cas or by always just incrementing the version by 1 treating
        // odds as locked. The trick with that is we need to be certain
        // not to increment the version so it needs to be tightly tied to
        // the locking code so there is no footgun.

        self.incr_version(guard);
        self.get_version().fetch_and(u64::MAX >> 1, Ordering::AcqRel);
    }

    #[inline]
    fn mut_body(&mut self, _guard: &WriteGuard<'_>) -> &mut [u8] {
        &mut self.data[64..]
    }

    #[inline]
    fn fetch_update_suffix_pos(&mut self, _guard: &WriteGuard<'_>, item_width: usize) -> u32 {
        let width = item_width; // 2 for len of item
        let tail_pos = read_u32(&self.data[HPOS_SUFFIX_POINTER]);
        let updated_tail_pos = tail_pos + width as u32;

        self.data[HPOS_SUFFIX_POINTER].copy_from_slice(&updated_tail_pos.to_le_bytes());

        updated_tail_pos
    }

    #[inline]
    fn set_slot_len(&mut self, len: u32) {
        self.data[HPOS_SLOT_LEN].copy_from_slice(&len.to_le_bytes());
    }

    #[inline]
    fn slot_insert(&mut self, key: &[u8], suffix_offset: u32) {
        // let idx = self.find_slot_pos(&key);
        let slot_len = read_u32(&self.data[HPOS_SLOT_LEN]) as usize;
        let cur_slot_end = SLOT_WIDTH * slot_len;
        let slot_key = pad_right_slice::<4>(&key[..]);

        match self.find_slot_pos(&key) {
            FindSlot::Insert(idx) => {
                let extended_slots_data = &mut self.data[64..64+cur_slot_end+SLOT_WIDTH];
                let start = idx * SLOT_WIDTH;

                // eprintln!("start: {}, cur_slot_end: {}, extended_slots_data.len: {}", start, cur_slot_end, extended_slots_data.len());
                extended_slots_data.copy_within(start..cur_slot_end, start+SLOT_WIDTH);

                extended_slots_data[start..start+4].copy_from_slice(&slot_key);
                extended_slots_data[start+4..start+8].copy_from_slice(&suffix_offset.to_le_bytes());

                // eprintln!("unswizzle: {}", unswizzle.as_u64());

                // todo!("Move to data [Insert]");
                // extended_slots_data[start+8..start+16].copy_from_slice(&unswizzle.as_u64().to_le_bytes());

                self.set_slot_len(slot_len as u32 + 1);
            }

            FindSlot::Replace(idx) => {
                let start = idx * SLOT_WIDTH + 64;

                if self.data[start+4..start+8] != [0u8; 4] {
                    let overflow_ptr = read_u32(&self.data[start+4..start+8]) as usize;
                    let overflow_len = read_u16(&self.data[overflow_ptr..overflow_ptr+2]) as u32;
                    
                    let freed = read_u32(&self.data[HPOS_FREED_SUFFIX_LEN]) + overflow_len;
                    self.data[HPOS_FREED_SUFFIX_LEN].copy_from_slice(&freed.to_le_bytes());
                }

                self.data[start..start+4].copy_from_slice(&slot_key);
                self.data[start+4..start+8].copy_from_slice(&suffix_offset.to_le_bytes());

                // todo!("Move to data [Replace]");
                // self.data[start+8..start+16].copy_from_slice(&unswizzle.as_u64().to_le_bytes());
            }
        }
    }
}

impl<'a> std::fmt::Debug for Page<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        // unimplemented!()

        let slot_len = read_u32(&self.data[HPOS_SLOT_LEN]) as usize;
        let slots_data = &self.data[64..64+(SLOT_WIDTH * slot_len)];

        let mut o = f.debug_struct("InnerPage");

        // #[derive(Debug)]
        struct SlotData<'a> {
            slot_key: &'a [u8],
            overflow_data: &'a [u8],
            data: &'a [u8],
            slot_offset: u32,
        }

        impl<'a> std::fmt::Debug for SlotData<'a> {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
                f.debug_struct("Slot")
                    .field("key", &format!("{:?}", self.slot_key))
                    .field("data", &format!("{:?}", self.data))
                    .field("overflow_offset", &self.slot_offset)
                    .field("overflow", &format!("{:?}", self.overflow_data))
                    .finish()
            }
        }

        let slots: Vec<SlotData> = slots_data.chunks_exact(SLOT_WIDTH).map(|slot| {
            let slot_key = &slot[0..4];
            let slot_offset = read_u32(&slot[4..8]);
            // let swip: Swip = read_u64(&slot[8..16]).into();

            let so = slot_offset as usize;
            let overflow_data = read_u32_len_bytes(&self.data[so..]);
            let data = read_u32_len_bytes(&self.data[so+4+overflow_data.len()..]);

            // if InnerPage, maybe show data as a Swip

            SlotData {
                slot_key,
                overflow_data,
                data,
                slot_offset,
            }
        }).collect();

        o.field("data_len", &self.data.len());
        o.field("slots", &slots);
        o.finish()
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub(crate) enum InsertPageError {
    PageFull,
    IncorrectPrefix,
}

fn read_u16(data: &[u8]) -> u16 {
    u16::from_le_bytes(data[0..2].try_into().expect("correct size of buffer"))
}

fn read_u32(data: &[u8]) -> u32 {
    u32::from_le_bytes(data[0..4].try_into().expect("correct size of buffer"))
}

fn read_u64(data: &[u8]) -> u64 {
    u64::from_le_bytes(data[0..8].try_into().expect("correct size of buffer"))
}

fn read_u8_len_bytes(data: &[u8]) -> &[u8] {
    let len = data[0] as usize;
    &data[1..1+len]
}

fn read_u16_len_bytes(data: &[u8]) -> &[u8] {
    let len = read_u16(&data[0..2]) as usize;
    &data[2..2+len]
}

fn read_u32_len_bytes(data: &[u8]) -> &[u8] {
    let len = read_u32(&data[0..4]) as usize;
    &data[4..4+len]
}

fn pad_right_slice<const N: usize>(input: &[u8]) -> [u8; N] {
    let mut key_slice = [0u8; N];
    let key_slice_len = std::cmp::min(N, input.len());
    key_slice[0..key_slice_len].copy_from_slice(&input[..key_slice_len]);
    key_slice
}

pub(crate) struct PageIter<'a> {
    page: Page<'a>,
    lock: ReadGuard<'static>,
}

impl<'a> PageIter<'a> {
    fn new(page: Page<'a>) -> Self {
        // We know that this item will drop with the iter, but this
        // is super sketch. What we're trying to do is to simply
        // tie the lock to the lifetime of the iterator.
        let lock = unsafe { 
            std::mem::transmute(page.read_lock())
        };

        PageIter {
            page,
            lock,
        }
    }

    pub(crate) fn page_prefix(&self) -> &[u8] {
        read_u8_len_bytes(&self.page.data[HPOS_PREFIX])
    }
}

impl<'a> Iterator for PageIter<'a> {
    type Item = (PageIterKey<'a>, &'a [u8]);

    fn next(&mut self) -> Option<Self::Item> {
        unimplemented!()
    }
}

// struct PageIterKey<'a> {
//     prefix: &'a [u8],
//     slot: &'a [u8],
//     suffix: &'a [u8],
// }

pub(crate) struct PageIterKey<'a>(&'a [u8], &'a [u8], &'a [u8]);

pub(crate) struct WriteLock<'a> {
    guard: WriteGuard<'a>,
    page: Page<'a>,
}

impl<'a> WriteLock<'a> {
    // fn page(&mut self) -> &mut Page<'a> {
    //     unsafe { self.page.as_mut().unwrap() }
    // }

    pub(crate) fn insert(&mut self, key: &[u8], data: &[u8]) -> Result<(), InsertPageError> {
        // let page = self.page.as_mut().unwrap() };
        // eprintln!("WriteLock Page: {:#?}", page);
        self.page.insert(&self.guard, key, data)
    }

    pub(crate) fn get_nearby(&mut self, key: &[u8]) -> &[u8] {
        // let page = unsafe { self.page.as_mut().unwrap() };
        self.page.get_nearby(key)
    }

    pub(crate) fn search(&mut self, key: &[u8]) -> Result<&[u8], &[u8]> {
        // let page = unsafe { self.page.as_mut().unwrap() };
        self.page.search(key)
    }
}

impl<'a> Drop for WriteLock<'a> {
    fn drop(&mut self) {
        // let page = unsafe { self.page.as_mut().unwrap() };
        self.page.unlock_version(&self.guard);
        // self.page.drop_write_lock(self.guard);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Swip;

    fn helper_insert(lock: &mut WriteLock, key: &[u8], data: Swizzle) -> Result<(), InsertPageError> {
        lock.insert(key, &data.as_u64().to_le_bytes())
    }

    #[test]
    fn try_locking() {
        let mut buffer = [0u8; 4096];
        let mut page = Page::from(&mut buffer);
        unsafe {
            page.bootstrap_inner(
                Unswizzle((213 << 7) + 1), 
                b"",
                (&[0u8; 4], Swizzle::new(1, 0)),
                1
            );
        }
        let g1 = page.read_lock();
        let g2 = page.read_lock();
        let g3 = page.read_lock();

        assert!(page.get_lock().try_write().is_none());

        // eprintln!("Header: {:?}", &page.data[0..32]);

        std::mem::drop(g1);
        std::mem::drop(g2);
        std::mem::drop(g3);

        // eprintln!("Header: {:?}", &page.data[0..32]);

        // let guard = page.get_lock().write();
        // page.lock_version(&guard);

        let mut lock = page.write_lock().unwrap();
        eprintln!("Header: {:?}", &page.data[0..32]);

        let swiz = Swizzle::new(10, 0);

        helper_insert(&mut lock, b"foo", swiz).unwrap();
        helper_insert(&mut lock, b"qux", (11, 0).into()).unwrap();
        helper_insert(&mut lock, b"bar", (12, 0).into()).unwrap();

        assert!(page.get_lock().try_write().is_none());
        assert!(page.get_lock().try_read().is_none());

        assert_eq!(Swip::Swizzle(swiz), page.get_nearby(b"fuu").into());
        assert_eq!(Swip::Swizzle(swiz), page.get_nearby(b"foo").into());

        eprintln!("Header: {:?}", &page.data[0..32]);
    }

    #[test]
    fn write_lock_v2_test() {
        let mut buffer = [0u8; 512];
        let mut page = Page::from(&mut buffer);
        unsafe {
            page.bootstrap_inner(
                Unswizzle((213 << 7) + 1), 
                b"",
                (&[0u8; 4], Swizzle::new(1, 0)),
                0
            );
        }

        {
            let mut lock = page.write_lock().unwrap();
            helper_insert(&mut lock, b"foo", (1, 0).into()).unwrap();

            assert!(page.get_lock().try_read().is_none());

            let version = page.load_version();
            assert!(version & 2 == 2);
            assert!(version & (1 << 63) > 0);
        }

        assert_eq!(3, page.load_version());
        assert!(page.get_lock().try_read().is_some());
    }

    #[test]
    fn page_full_with_overflow() {
        let mut buffer = [0u8; 4096];
        let mut page = Page::from(&mut buffer);
        unsafe {
            page.bootstrap_inner(
                Unswizzle((213 << 7) + 1), 
                b"",
                (&[0u8; 4], Swizzle::new(1, 0)),
                1
            );
        }

        let mut lock = page.write_lock().unwrap();
        let res = helper_insert(&mut lock, &[1u8; 5000], (11, 0).into());
        assert_eq!(Err(InsertPageError::PageFull), res);
    }

    #[test]
    fn page_full_without_overflow() {
        let mut buffer = [0u8; 512];
        let mut page = Page::from(&mut buffer);
        unsafe {
            page.bootstrap_inner(
                Unswizzle((213 << 7) + 1), 
                b"",
                (&[0u8; 4], Swizzle::new(1, 0)),
                1
            );
        }

        let mut lock = page.write_lock().unwrap();

        let max_entries = (512 - 64) / (SLOT_WIDTH + 16) - 1;

        for i in 0..max_entries {
            helper_insert(
                &mut lock,
                &[(i+1) as u8; 4], 
                (i as usize+1, 0).into()
            ).expect("to write just fine");
        }

        let res = helper_insert(
            &mut lock,
            &[max_entries as u8; 4], 
            (max_entries, 0).into()
        );
        assert_eq!(Err(InsertPageError::PageFull), res);
    }

    #[test]
    fn page_full_overflow_collision() {
        // TODO
    }

    #[test]
    fn inserting_pages_with_same_slot_prefix() {
        let mut buffer = [0u8; 512];
        let mut page = Page::from(&mut buffer);
        unsafe {
            page.bootstrap_inner(
                Unswizzle((213 << 7) + 1), 
                b"",
                (&[0u8; 4], (1, 0).into()),
                1
            );
        }

        let mut lock = page.write_lock().unwrap();
        helper_insert(&mut lock, b"aaaafoo", (3, 0).into()).unwrap();
        helper_insert(&mut lock, b"aaaabar", (2, 0).into()).unwrap();
        helper_insert(&mut lock, b"aaaaqux", (4, 0).into()).unwrap();
        std::mem::drop(lock);

        eprintln!("InnerPage: {:#?}", page);

        assert_eq!(1, helper_get_nearby_page_id(&page, b"aaaa"));
        assert_eq!(2, helper_get_nearby_page_id(&page, b"aaaabb"));
        assert_eq!(2, helper_get_nearby_page_id(&page, b"aaaafo"));
        assert_eq!(3, helper_get_nearby_page_id(&page, b"aaaafp"));
        assert_eq!(3, helper_get_nearby_page_id(&page, b"aaaafoo"));
        assert_eq!(3, helper_get_nearby_page_id(&page, b"aaaafood"));
        assert_eq!(3, helper_get_nearby_page_id(&page, b"aaaaqu"));
        assert_eq!(4, helper_get_nearby_page_id(&page, b"aaaaqux"));
        assert_eq!(4, helper_get_nearby_page_id(&page, b"aaaaquxxxx"));
        assert_eq!(4, helper_get_nearby_page_id(&page, b"aaaaqz"));
    }

    fn helper_get_nearby_page_id(page: &Page, key: &[u8]) -> usize {
        let swip: Swip = page.get_nearby(key).into();
        swip.page_id()
    }

    #[test]
    fn insert_and_get() {
        let mut buffer = [0u8; 512];
        let mut page = Page::from(&mut buffer);
        unsafe {
            page.bootstrap_inner(
                Unswizzle((213 << 7) + 1), 
                b"",
                (&[0u8; 4], (1, 0).into()),
                1
            );
        }

        let mut lock = page.write_lock().unwrap();
        helper_insert(&mut lock, b"bar", (2, 0).into()).unwrap();
        helper_insert(&mut lock, b"qux", (3, 0).into()).unwrap();
        std::mem::drop(lock);

        eprintln!("Page: {:#?}", page);

        assert_eq!(1, helper_get_nearby_page_id(&page, b""));
        assert_eq!(1, helper_get_nearby_page_id(&page, b"a"));
        assert_eq!(1, helper_get_nearby_page_id(&page, b"b"));
        assert_eq!(1, helper_get_nearby_page_id(&page, b"baa"));
        assert_eq!(2, helper_get_nearby_page_id(&page, b"bar"));
        assert_eq!(2, helper_get_nearby_page_id(&page, b"barbar"));
        assert_eq!(2, helper_get_nearby_page_id(&page, b"bars"));
        assert_eq!(3, helper_get_nearby_page_id(&page, b"qux"));
        assert_eq!(3, helper_get_nearby_page_id(&page, b"quz"));
        assert_eq!(3, helper_get_nearby_page_id(&page, &[255, 255, 255, 255]));
    }

    #[test]
    fn replacing_pages_with_same_key() {
        let mut buffer = [0u8; 512];
        let mut page = Page::from(&mut buffer);
        unsafe {
            page.bootstrap_inner(
                Unswizzle((213 << 7) + 1), 
                b"",
                (&[0u8; 4], (1, 0).into()),
                1
            );
        }

        let mut lock = page.write_lock().unwrap();
        helper_insert(&mut lock, b"aaaafoo", (2, 0).into()).unwrap();
        helper_insert(&mut lock, b"aaaafoo", (3, 0).into()).unwrap();

        assert_eq!(Swip::Swizzle(Swizzle::new(3, 0)), page.get_nearby(b"aaaafoo").into());
    }

    fn bootstrap_page(buffer: &mut [u8]) -> Page {
        let mut page = Page::from(buffer);
        unsafe {
            page.bootstrap_inner(
                Unswizzle((213 << 7) + 1), 
                b"",
                (&[0u8; 4], (1, 0).into()),
                1
            );
        }
        page
    }

    #[test]
    fn version_when_locked() {
        let mut buffer = [0u8; 512];
        let mut page = bootstrap_page(&mut buffer);

        let _lock = page.write_lock().unwrap();
        assert_eq!(VersionLoadError::Locked, page.version().unwrap_err());
    }

    #[test]
    fn version_when_unloaded() {
        let mut buffer = [0u8; 512];
        let page = Page::from(&mut buffer);

        assert_eq!(VersionLoadError::Unloaded, page.version().unwrap_err());
    }
}
