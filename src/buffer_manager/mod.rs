use crate::{DiskManager, DiskManagerAllocError, buffer_manager::swip::Swip};
use std::{fmt, mem::size_of};
use std::sync::Arc;
use std::cell::UnsafeCell;
use log::debug;
use madvise::AdviseMemory;
use memmap::MmapMut;
use std::sync::atomic::{AtomicU64, Ordering};

use self::buffer_frame::{BufferFrame, PageGuard, PAGE_DATA_RESERVED};

pub(crate) mod swip;
pub(crate) mod buffer_frame;

pub(crate) struct BufferManager {
    disk_manager: Box<dyn DiskManager>,
    base_page_size: usize,
    max_memory: usize,
    page_classes: Vec<PageClass>,
    version_boundary: AtomicU64,
    frames: UnsafeCell<MmapMut>,
}

unsafe impl Send for BufferManager {}
unsafe impl Sync for BufferManager {}

pub(crate) trait Swipable {
    fn set_backing_len(&mut self, len: usize);
}

impl fmt::Debug for BufferManager {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt.debug_struct("PageClass")
           .field("base_page_size", &self.base_page_size)
           .field("max_memory", &self.max_memory)
           .field("page_classes", &self.page_classes)
           .finish()
    }
}

impl BufferManager {
    pub fn new(disk_manager: Box<dyn DiskManager>, max_memory: usize) -> Self {
        let base_page_size = disk_manager.base_page_size();
        let mut page_classes = Vec::new();

        let mut class_size = base_page_size;
        while class_size < max_memory {
            page_classes.push(PageClass::new(class_size, max_memory));
            class_size *= 2;
        }

        // Preallocate all possible frames, which is limited based on how much
        // memory we reserve for the BufferManager and the smallest page size.
        //
        // We never delloc (for the duration of the BufferManager) these slots
        // as there can always be live pointers. Additionally, we never decrease
        // the version of a Frame (even if the underlying page is unswizzled)
        // once it is init'ed, which ensures optimistic latches never see an
        // invalid page.
        let max_frames =  max_memory / base_page_size;
        let frames = UnsafeCell::new(MmapMut::map_anon(max_frames).unwrap());

        Self {
            disk_manager,
            base_page_size,
            max_memory,
            page_classes,
            version_boundary: AtomicU64::new(0),
            frames,
        }
    }

    pub fn new_page<T: Swipable>(&self) -> Result<Swip<T>, DiskManagerAllocError> {
        let page_id = self.disk_manager.allocate_page(0)?;
        let page: &mut buffer_frame::Page = unsafe {
            std::mem::transmute(self.page_classes[0].get_page(page_id).as_ptr())
        };

        // TODO: assert within range
        let bf = unsafe {
            self.frames
                .get()
                .offset((page_id * size_of::<BufferFrame>()) as isize)
                as *mut BufferFrame
        };

        // let bf = Box::new(BufferFrame::new(page));
        unsafe { BufferFrame::init(bf, page); }
        let swip = Swip::new(bf as usize);

        let mut guard: PageGuard<T> = PageGuard::new_exclusive(swip);
        guard.set_backing_len(self.page_classes[0].page_size - PAGE_DATA_RESERVED);
        std::mem::drop(guard);

        Ok(swip)

        // Ok((unswizzle.into(), self.page_classes[0].get_page(unswizzle.page_id())))
    }

    pub fn free_page() {}
    pub fn flush_pages() {}

    /// Safety: This function will take anything that looks like a swizzled
    /// pointer and convert that into a Page. Caller needs to be certain that
    /// the referenced pointer is in fact pointing to one of the mmap'ed
    /// segements created by PageClass.
    // pub unsafe fn load_swizzled(&self, swizzle: Swizzle) -> &mut [u8] {
    //     unimplemented!();
    //     let page_class = &self.page_classes[swizzle.size_class()];
    //     page_class.get_page(swizzle.page_id())
    // }

    pub async fn load_unswizzled(&self, unswizzle: usize) -> &mut [u8] {
        // TODO: This function will look at the cooling stage and either reheat
        // or perform a fetch from the cold store (disk). The latter is why this
        // function is async -- if it needs to go to disk, that will require an
        // async operation.
        unimplemented!()
    }

    pub(crate) fn version_boundary(&self) -> u64 {
        self.version_boundary.load(Ordering::Acquire)
    }

    // TODO: Ensure when unswizzling a page, we bump version_boundary
}

struct PageClass {
    page_size: usize,
    capacity: usize,
    mmap: UnsafeCell<MmapMut>,

    // marker: std::marker::PhantomData<&'a ()>,
}

unsafe impl Sync for PageClass {}

impl fmt::Debug for PageClass {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt.debug_struct("PageClass")
           .field("page_size", &self.page_size)
           .field("capacity", &self.capacity)
           .field("mmap", &format!("[mmap@{}]", self.mmap.get() as usize))
           .finish()
    }
}

impl PageClass {
    fn new(page_size: usize, capacity: usize) -> Self {
        assert!(capacity % 4096 == 0, "capacity must be divisible by 4096");
        let mmap = UnsafeCell::new(MmapMut::map_anon(capacity).unwrap());

        Self {
            page_size,
            capacity,
            mmap,
            // marker: Default::default()
        }
    }

    fn mmap_start_as_usize(&self) -> usize {
        self.mmap.get() as usize
    }

    /// Safety: This function will happily hand out multiple instances of the
    /// same underlying range. It is the responsibility of the caller to manage
    /// any potential data races.
    unsafe fn get_page(&self, page_id: usize) -> &mut [u8] {
        let start = page_id * self.page_size;
        // println!("get_page=> page_size: {}, pid: {}", self.page_size, page_id);

        let mmap = self.mmap.get().as_mut().unwrap();
        &mut mmap[start..start+self.page_size]
    }

    /// Safety: Calling this will clear out the data underlying the page. It
    /// is up to the users of this and get_page to manage race conditions that
    /// may occur.
    unsafe fn release_page(&self, page_id: usize) {
        // TODO: what to do when the advise fails?

        let start = page_id * self.page_size;
        let mmap = self.mmap.get().as_ref().unwrap();
        let _ = mmap[start..start+self.page_size]
            .advise_memory_access(madvise::AccessPattern::DontNeed)
            .unwrap();
    }
}

// pub struct Page<'a> {
//     /// Safety: This data is backed by an aliased mutable buffer. It must be
//     /// all accesses to this data must be done in a thread safe manner. At
//     /// any point, another thread may also reset the buffer to zero bytes.
//     /// See: `PageClass.get_page`.
//     data: &'a mut [u8],
// }

// impl<'a> Page<'a> {
//     /// Safety: This method assumes only one reference to the underlying data
//     /// when called. Given this cannot be guaranteed at this layer, caller is
//     /// responsible for upholding that guarantee.
//     unsafe fn bootstrap(&mut self, unswizzle: Unswizzle, page_size: usize, base_page_size: usize) {
//         use std::io::Write;
//         (&mut self.data[0..8]).write(&(unswizzle.as_u64()).to_le_bytes()).expect("Memory write should always succeed");
//     }

//     fn get_swizzle(&self) -> Swizzle {
//         Swizzle((u64::from_le_bytes(self.data[0..8].try_into().unwrap()) | 1) as usize)
//     }
// }




#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::FakeDiskManager;

    #[test]
    fn it_works() {
        let dm = FakeDiskManager::boxed();
        let _bm = BufferManager::new(dm, 4096 * 4096);
        // eprintln!("{:?}", bm);
    }
}

