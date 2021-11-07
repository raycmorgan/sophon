use crate::{DiskManager, Swizzle, Unswizzle};
use std::fmt;
use std::cell::UnsafeCell;
use madvise::AdviseMemory;
use memmap::MmapMut;

struct BufferManager<'a, D: DiskManager> {
    disk_manager: D,
    base_page_size: usize,
    max_memory: usize,
    page_classes: Vec<PageClass<'a>>,
}

impl<'a, D: DiskManager> fmt::Debug for BufferManager<'a, D> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt.debug_struct("PageClass")
           .field("base_page_size", &self.base_page_size)
           .field("max_memory", &self.max_memory)
           .field("page_classes", &self.page_classes)
           .finish()
    }
}

impl<'a, D: DiskManager> BufferManager<'a, D> {
    pub fn new(disk_manager: D,max_memory: usize) -> Self {
        let base_page_size = disk_manager.base_page_size();
        let mut page_classes = Vec::new();

        let mut class_size = base_page_size;
        while class_size < max_memory {
            page_classes.push(PageClass::new(class_size, max_memory));
            class_size *= 2;
        }

        Self {
            disk_manager,
            base_page_size,
            max_memory,
            page_classes,
        }
    }

    /// Safety: Caller is responsible for managing concurrency on the returned
    /// buffer. The BufferManager does not prevent aliasing. Even though this
    /// is logically a new page, it might be a recycled freed page.
    pub unsafe fn new_page(&self) -> Result<&'a mut [u8], u8> {
        let unswizzle = self.disk_manager.allocate_page(0)?;
        Ok(self.page_classes[0].get_page(unswizzle.page_id()))
    }

    pub fn free_page() {}
    pub fn flush_pages() {}

    /// Safety: This function will take anything that looks like a swizzled
    /// pointer and convert that into a Page. Caller needs to be certain that
    /// the referenced pointer is in fact pointing to one of the mmap'ed
    /// segements created by PageClass.
    pub unsafe fn load_swizzled(&self, swizzle: Swizzle) -> &'a mut [u8] {
        let page_class = &self.page_classes[swizzle.size_class()];
        page_class.get_page(swizzle.page_id())
    }

    pub async fn load_unswizzled(&self, unswizzle: usize) -> &'a mut [u8] {
        // TODO: This function will look at the cooling stage and either reheat
        // or perform a fetch from the cold store (disk). The latter is why this
        // function is async -- if it needs to go to disk, that will require an
        // async operation.
        unimplemented!()
    }
}

struct PageClass<'a> {
    page_size: usize,
    capacity: usize,
    mmap: UnsafeCell<MmapMut>,

    marker: std::marker::PhantomData<&'a ()>,
}

impl<'a> fmt::Debug for PageClass<'a> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt.debug_struct("PageClass")
           .field("page_size", &self.page_size)
           .field("capacity", &self.capacity)
           .field("mmap", &format!("[mmap@{}]", self.mmap.get() as usize))
           .finish()
    }
}

impl<'a> PageClass<'a> {
    fn new(page_size: usize, capacity: usize) -> Self {
        assert!(capacity % 4096 == 0, "capacity must be divisible by 4096");
        let mmap = UnsafeCell::new(MmapMut::map_anon(capacity).unwrap());

        Self {
            page_size,
            capacity,
            mmap,
            marker: Default::default()
        }
    }

    fn mmap_start_as_usize(&self) -> usize {
        self.mmap.get() as usize
    }

    /// Safety: This function will happily hand out multiple instances of the
    /// same underlying range. It is the responsibility of the caller to manage
    /// any potential data races.
    unsafe fn get_page(&self, page_id: usize) -> &'a mut [u8] {
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
    use crate::{DiskManager, Unswizzle};

    use std::sync::atomic::{AtomicUsize, Ordering};

    struct FakeDiskManager {
        next_page: AtomicUsize,
    }

    impl DiskManager for FakeDiskManager {
        fn capacity(&self) -> usize { 4096 * 1000 }
        fn base_page_size(&self) -> usize { 4096 }

        fn allocate_page(&self, size_class: usize) -> Result<Unswizzle, u8> {
            let next = self.next_page.fetch_add(1, Ordering::SeqCst);
            Ok(Unswizzle::from_parts(next, size_class))
        }

        fn deallocate_page() {}

        fn read_page() {}
        fn write_page() {}
        fn fsync() {}
    }

    #[test]
    fn it_works() {
        let dm = FakeDiskManager{ next_page: AtomicUsize::new(0) };
        let bm = BufferManager::new(dm, 4096 * 4096);
        // eprintln!("{:?}", bm);
    }
}

