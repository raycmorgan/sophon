use std::ops::{DerefMut, Deref};
use crate::buffer_manager::swip::Swip;

pub(crate) struct ExclusiveGuard<T> {
    swip: Swip<T>,
}

impl<T> ExclusiveGuard<T> {
    pub(crate) fn new(swip: Swip<T>) -> Self {
        let frame = unsafe { swip.as_buffer_frame_mut() };
        frame.latch_exclusive();

        ExclusiveGuard {
            swip,
        }
    }

    pub(crate) fn data_structure(&self) -> &T {
        let frame = unsafe { self.swip.as_buffer_frame() };
        let ptr = frame.page.data.as_ptr();
        unsafe { std::mem::transmute(ptr.as_ref().unwrap()) }
    }

    pub(crate) fn data_structure_mut(&mut self) -> &mut T {
        let frame = unsafe { self.swip.as_buffer_frame_mut() };
        let ptr = frame.page.data.as_mut_ptr();
        unsafe { std::mem::transmute(ptr.as_mut().unwrap()) }
    }
    
    #[inline]
    pub(crate) fn version(&self) -> u64 {
        let frame = unsafe { self.swip.as_buffer_frame() };
        frame.version()
    }
}

impl<T> Drop for ExclusiveGuard<T> {
    fn drop(&mut self) {
        // Safety: ExclusiveGuard takes an exclusive latch when created, thus
        // we know we own the latch.
        unsafe {
            let frame = self.swip.as_buffer_frame();
            frame.unlatch_exclusive();
        }
    }
}

impl<T> Deref for ExclusiveGuard<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.data_structure()
    }
}

impl<T> DerefMut for ExclusiveGuard<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.data_structure_mut()
    }
}
