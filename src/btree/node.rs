use std::{mem::size_of, sync::atomic::AtomicU64};
use parking_lot::RawRwLock;

#[repr(C)]
struct Fence {
    pos: u16,
    len: u16,
}

#[repr(C, packed)]
struct Slot {

}

#[repr(C)]
struct NodeHeader {
    db_version: u8,
    pid: u64,

    upper_swip: Option<Swip<Node>>,

    lower_fence: Fence,
    upper_fence: Fence,

    version: AtomicU64,
    latch: RawRwLock,

    is_leaf: bool,
    slot_count: u16,
    space_used: u16,
    prefix_len: u16,
}

// const MAX_SLOT_COUNT: usize = 

#[repr(C)]
struct Node {
    // header: NodeHeader,

    raw: [u8; PAGE_SIZE - size_of::<u64>()]
}



struct Swip<T> {
    ptr: usize, // either pid or pointer to BufferFrame
    _marker: std::marker::PhantomData<T>,
}

impl<T> Swip<T> {
    // fn cast(&self) -> &T {
    //     // let frame = unsafe { std::mem::transmute::<_, &mut BufferFrame>(&mut self) };
    //     unimplemented!()
    // }

    fn as_buffer_frame(&self) -> &BufferFrame {
        // debug_assert swizzle
        unsafe { (self.ptr as *const BufferFrame).as_ref().unwrap() }
    }

    fn as_buffer_frame_mut(&mut self) -> &mut BufferFrame {
        unsafe { (self.ptr as *mut BufferFrame).as_mut().unwrap() }
    }
}



struct BufferFrame<'a> {
    // header,
    page: &'a mut Page, // points to the underlying mmap'ed buffer
}

impl<'a> BufferFrame<'a> {
    // fn latch_exclusive(&self) -> ExclusiveGuard<'a> {
    //     // self.latch.lock_exclusive
    //     // latch frame
    // }
}


const PAGE_SIZE: usize = 1024 * 16;

#[repr(C)]
struct Page {
    gsn: u64,
    data: [u8; PAGE_SIZE - size_of::<u64>()],
}


struct ExclusiveGuard<'a, T> {
    frame: &'a mut BufferFrame<'a>,
    _marker: std::marker::PhantomData<T>,
}

impl<'a, T> ExclusiveGuard<'a, T> {
    fn new(swip: &'a mut Swip<T>) -> Self {
        let frame = swip.as_buffer_frame_mut();

        // frame.latch.exclusive_lock();

        ExclusiveGuard {
            frame,
            _marker: Default::default()
        }
    }

    fn data_structure(&mut self) -> &mut T {
        let ptr = self.frame.page.data.as_mut_ptr();
        unsafe { std::mem::transmute(ptr.as_mut().unwrap()) }
    }
}

impl<'a, T> Drop for ExclusiveGuard<'a, T> {
    fn drop(&mut self) {
        // self.frame.latch.unlock_exclusive()
    }
}

fn foo() {
    let mut swip: Swip<Node> = Swip { ptr: 1, _marker: Default::default() };
    let guard = ExclusiveGuard::new(&mut swip);
    let dts = guard.data_structure();
}
