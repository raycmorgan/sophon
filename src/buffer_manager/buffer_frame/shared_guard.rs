use std::fmt::Debug;
use std::ops::Deref;

use crate::buffer_manager::buffer_frame::BufferFrame;
use crate::buffer_manager::swip::Swip;

use super::ExclusiveGuard;

pub(crate) struct SharedGuard<'a, T> {
    frame: &'a BufferFrame<'a>,
    _marker: std::marker::PhantomData<T>,
}

impl<'a, T> SharedGuard<'a, T> {
    pub(crate) fn new(swip: &'a Swip<T>) -> Self {
        let frame = unsafe { swip.as_buffer_frame() };
        frame.latch_shared();

        SharedGuard {
            frame,
            _marker: Default::default()
        }
    }

    pub(crate) fn data_structure(&self) -> &T {
        let ptr = self.frame.page.data.as_ptr();
        unsafe { std::mem::transmute(ptr.as_ref().unwrap()) }
    }

    #[inline]
    pub(crate) fn version(&self) -> u64 {
        self.frame.version()
    }
}

impl<'a, T> Drop for SharedGuard<'a, T> {
    fn drop(&mut self) {
        // Safety: SharedGuard takes a shared latch when created, thus
        // we know we own a latch.
        unsafe { self.frame.unlatch_shared(); }
    }
}

impl<'a, T> Deref for SharedGuard<'a, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.data_structure()
    }
}



pub(crate) struct SharedGuard2<T> {
    swip: Swip<T>,
    initial_version: u64,
    upgraded: bool,
}

impl<T> SharedGuard2<T> {
    pub(crate) fn new(swip: Swip<T>) -> Self {
        let frame = unsafe { swip.as_buffer_frame() };
        let mut initial_version = frame.version();
        let mut upgraded = false;

        if initial_version & (1 << 63) > 0 {
            frame.latch_shared();
            initial_version = frame.version();
            upgraded = true;
        }

        SharedGuard2 {
            swip,
            initial_version,
            upgraded,
        }
    }

    #[inline]
    pub(crate) fn data_structure(&self) -> &T {
        let frame = unsafe { self.swip.as_buffer_frame() };
        let ptr = frame.page.data.as_ptr();
        unsafe { std::mem::transmute(ptr.as_ref().unwrap()) }
    }

    #[inline]
    pub(crate) fn version(&self) -> u64 {
        self.initial_version
    }

    #[inline]
    pub(crate) fn is_valid(&self) -> bool {
        self.upgraded || self.initial_version == self.version()
    }

    #[inline]
    pub(crate) fn upgrade(&mut self) {
        if !self.upgraded {
            let frame = unsafe { self.swip.as_buffer_frame() };
            frame.latch_shared();
            self.initial_version = frame.version();
            self.upgraded = true;
        }
    }
}

impl<T> Drop for SharedGuard2<T> {
    fn drop(&mut self) {
        // Safety: SharedGuard takes a shared latch when created, thus
        // we know we own a latch.
        unsafe {
            if self.upgraded {
                let frame = self.swip.as_buffer_frame();
                frame.unlatch_shared();
            }
        }
    }
}

impl<T> Deref for SharedGuard2<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.data_structure()
    }
}

impl<T> Debug for SharedGuard2<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LockGuard")
            .finish()
    }
}
