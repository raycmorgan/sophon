use parking_lot::{
    lock_api::{RawRwLock, RawRwLockUpgrade},
    RawRwLock as RwLock,
};
use std::{
    mem::size_of,
    sync::atomic::{AtomicU64, Ordering},
};

mod page_guard;

pub(crate) use page_guard::{LatchStrategy, PageGuard};
pub(crate) const PAGE_DATA_RESERVED: usize = size_of::<PageHeader>();

struct Header {
    version: AtomicU64,
    latch: RwLock,
}

pub(crate) struct BufferFrame<'a> {
    header: Header,
    page: &'a mut Page, // points to the underlying mmap'ed buffer
}

impl<'a> BufferFrame<'a> {
    #[cfg(test)]
    fn testable_fake_page() -> BufferFrame<'static> {
        // Safety: Caller must not utilize the page field, it is invalid
        unsafe {
            #[allow(mutable_transmutes)]
            BufferFrame {
                header: Header {
                    version: AtomicU64::new(0),
                    latch: RwLock::INIT,
                },
                page: std::mem::transmute(b""),
            }
        }
    }

    pub(crate) unsafe fn init<'b>(bf: *mut BufferFrame<'b>, page: &'b mut Page) {
        let mut frame = bf.as_mut().expect("reference");
        frame.header.version = AtomicU64::new(0);
        frame.header.latch = RwLock::INIT;
        frame.page = page;
    }

    fn latch_exclusive(&self) {
        self.header.latch.lock_exclusive();
        self.header.version.fetch_or(1u64 << 63, Ordering::AcqRel);
    }

    // Safety: Caller must be sure they own the exclusive lock on this BufferFrame
    unsafe fn unlatch_exclusive(&self) {
        self.header.version.fetch_add(1, Ordering::AcqRel);
        self.header.version.fetch_xor(1u64 << 63, Ordering::AcqRel);
        self.header.latch.unlock_exclusive();
    }

    fn latch_shared(&self) {
        self.header.latch.lock_shared();
    }

    // Safety: Caller must be sure they own a shared lock on this BufferFrame
    unsafe fn unlatch_shared(&self) {
        self.header.latch.unlock_shared();
    }

    unsafe fn upgrade_latch(&self) {
        self.header.latch.upgrade();
        self.header.version.fetch_or(1u64 << 63, Ordering::AcqRel);
    }

    fn version(&self) -> u64 {
        self.header.version.load(Ordering::Acquire)
    }
}

pub(crate) struct PageHeader {
    _gsn: u64,
}

#[repr(C)]
pub(crate) struct Page {
    header: PageHeader,
    // Placeholder -- real len in noted self contained inside data,
    // it is at least large enough to know its own length
    data: [u8; 0], // [u8; USABLE_PAGE_SIZE],
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constants_make_sense() {
        // assert_eq!(PAGE_SIZE, std::mem::size_of::<Page>());
    }

    #[test]
    fn exclusive_lock() {
        let bf = BufferFrame::testable_fake_page();
        let preversion = bf.version();
        bf.latch_exclusive();
        assert_ne!(preversion, bf.version());
        assert!(bf.header.latch.is_locked());
        unsafe {
            bf.unlatch_exclusive();
        }
        assert_eq!(false, bf.header.latch.is_locked());
        assert!(preversion < bf.version());
    }
}
