use crate::buffer_manager::swip::Swip;
use std::{fmt, mem::size_of};
use std::cell::UnsafeCell;
use madvise::AdviseMemory;
use memmap::MmapMut;
use std::sync::atomic::{AtomicUsize, Ordering};

use self::buffer_frame::{BufferFrame, PageGuard, PAGE_DATA_RESERVED, PAGE_SIZE};

pub(crate) mod swip;
pub(crate) mod buffer_frame;

pub(crate) struct BufferManager {
    base_page_size: usize,
    max_memory: usize,
    page_classes: Vec<PageClass>,
    page_counter: AtomicUsize,
    frames: UnsafeCell<MmapMut>,
}

unsafe impl Send for BufferManager {}
unsafe impl Sync for BufferManager {}

pub(crate) trait Swipable {
    fn set_backing_len(&mut self, len: usize);
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum AllocError {
    OutOfSpace
}

impl BufferManager {
    pub fn new(max_memory: usize) -> Self {
        let base_page_size = PAGE_SIZE;
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
        let frames = MmapMut::map_anon(max_frames * size_of::<BufferFrame>()).unwrap();

        // println!("page_class -- mmap: {:p}", mmap.as_ref());
        // unsafe { println!("frames: {:p}", frames.as_ref()); }

        Self {
            base_page_size,
            max_memory,
            page_classes,
            page_counter: AtomicUsize::new(0),
            frames: UnsafeCell::new(frames),
            // frames
        }
    }

    pub fn new_page<T: Swipable>(&self) -> Result<Swip<T>, AllocError> {
        // Simple bump allocation of pages
        // TODO: implement reuse after free
        let page_id = self.page_counter.fetch_add(1, Ordering::SeqCst);

        let page: &mut buffer_frame::Page = unsafe {
            std::mem::transmute(self.page_classes[0].get_page(page_id).as_ptr())
        };

        // TODO: assert within range
        let bf = unsafe {
            let bf = self.frames
                .get().as_mut().unwrap()
                .as_mut_ptr()
                .offset((page_id * size_of::<BufferFrame>()) as isize)
                as *mut BufferFrame;

            BufferFrame::init(bf, page);
            bf
        };
        
        let swip = Swip::new(bf as usize);

        let mut guard: PageGuard<T> = PageGuard::new_exclusive(swip);
        guard.set_backing_len(self.page_classes[0].page_size - PAGE_DATA_RESERVED);
        std::mem::drop(guard);

        Ok(swip)
    }

    pub fn free_page() {}
    pub fn flush_pages() {}

    pub async fn load_unswizzled(&self, unswizzle: usize) -> &mut [u8] {
        // TODO: This function will look at the cooling stage and either reheat
        // or perform a fetch from the cold store (disk). The latter is why this
        // function is async -- if it needs to go to disk, that will require an
        // async operation.
        unimplemented!()
    }
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

struct PageClass {
    page_size: usize,
    capacity: usize,
    mmap: UnsafeCell<MmapMut>,
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
        assert!(capacity > 0, "capacity must be divisible by 4096");
        assert!(capacity % 4096 == 0, "capacity must be divisible by 4096");
        let mmap = MmapMut::map_anon(capacity).unwrap();

        // println!("page_class -- mmap: {:p}", mmap.as_ref());

        Self {
            page_size,
            capacity,
            mmap: UnsafeCell::new(mmap),
        }
    }

    /// Safety: This function will happily hand out multiple instances of the
    /// same underlying range. It is the responsibility of the caller to manage
    /// any potential data races.
    unsafe fn get_page(&self, page_id: usize) -> &mut [u8] {
        let start = page_id * self.page_size;
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
