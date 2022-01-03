use std::fmt::Debug;
use std::ops::{Deref, DerefMut};

use crate::buffer_manager::buffer_frame::BufferFrame;
use crate::buffer_manager::swip::Swip;

#[derive(Copy, Clone, Debug, PartialEq)]
enum LockState {
    Optimistic,
    Shared,
    Exclusive
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub(crate) enum LatchStrategy {
    OptimisticSpin,
    OptimisticOrConflict,
    OptimisticOrShared,
    OptimisticOrExclusive,
    Shared,
    Exclusive,
}

pub(crate) struct PageGuard<T> {
    swip: Swip<T>,
    initial_version: u64,
    state: LockState,
}

impl<T> Debug for PageGuard<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LockGuard")
            .finish()
    }
}

impl<T> PageGuard<T> {
    pub(crate) fn new_optimistic_spin(swip: Swip<T>) -> Self {
        let frame = unsafe { swip.as_buffer_frame() };
        let state = LockState::Optimistic;
        let mut initial_version;

        loop {
            initial_version = frame.version();

            if initial_version & (1 << 63) == 0 {
                break;
            }
        }
        
        PageGuard {
            swip,
            initial_version,
            state,
        }
    }

    pub(crate) fn new_optimistic_or_shared(swip: Swip<T>) -> Self {
        let frame = unsafe { swip.as_buffer_frame() };
        let mut initial_version = frame.version();
        let mut state = LockState::Optimistic;

        if initial_version & (1 << 63) > 0 {
            frame.latch_shared();
            initial_version = frame.version();
            state = LockState::Shared;
        }

        PageGuard {
            swip,
            initial_version,
            state,
        }
    }

    pub(crate) fn new_optimistic_or_exclusive(swip: Swip<T>) -> Self {
        let frame = unsafe { swip.as_buffer_frame() };
        let mut initial_version = frame.version();
        let mut state = LockState::Optimistic;

        if initial_version & (1 << 63) > 0 {
            frame.latch_exclusive();
            initial_version = frame.version();
            state = LockState::Exclusive;
        }

        PageGuard {
            swip,
            initial_version,
            state,
        }
    }

    pub(crate) fn new_shared(swip: Swip<T>) -> Self {
        let frame = unsafe { swip.as_buffer_frame() };

        frame.latch_shared();
        let initial_version = frame.version();
        let state = LockState::Shared;

        PageGuard {
            swip,
            initial_version,
            state,
        }
    }

    pub(crate) fn new_exclusive(swip: Swip<T>) -> Self {
        let frame = unsafe { swip.as_buffer_frame() };

        frame.latch_exclusive();
        let initial_version = frame.version();
        let state = LockState::Exclusive;

        PageGuard {
            swip,
            initial_version,
            state,
        }
    }

    #[inline]
    pub(crate) fn data_structure(&self) -> &T {
        let frame = unsafe { self.swip.as_buffer_frame() };
        let ptr = frame.page.data.as_ptr();
        unsafe { std::mem::transmute(ptr.as_ref().unwrap()) }
    }

    #[inline]
    pub(crate) fn data_structure_mut(&self) -> &mut T {
        assert_eq!(LockState::Exclusive, self.state, "data_structure_mut requires exclusive lock");

        let frame = unsafe { self.swip.as_buffer_frame_mut() };
        let ptr = frame.page.data.as_mut_ptr();
        unsafe { std::mem::transmute(ptr.as_mut().unwrap()) }
    }

    #[inline]
    pub(crate) fn version(&self) -> u64 {
        let frame = unsafe { self.swip.as_buffer_frame() };

        if self.state == LockState::Exclusive {
            frame.version() ^ 1u64 << 63
        } else {
            frame.version()
        }
    }

    #[inline]
    pub(crate) fn is_valid(&self) -> bool {
        self.state != LockState::Optimistic || self.initial_version == self.version()
    }

    #[inline]
    pub(crate) fn pid(&self) -> u64 {
        self.swip.page_id()
    }

    #[inline]
    pub(crate) fn swip_bytes(&self) -> [u8; 8] {
        self.swip.value_bytes()
    }

    #[inline]
    pub(crate) fn upgrade_shared(&mut self) {
        match self.state {
            LockState::Exclusive => panic!("Cannot call upgrade_shared when exclusive lock is held"),
            LockState::Shared => (),
            LockState::Optimistic => {
                let frame = unsafe { self.swip.as_buffer_frame() };
                frame.latch_shared();
                self.state = LockState::Shared;
            }
        }
    }

    #[inline]
    pub(crate) fn upgrade_exclusive(&mut self) {
        let frame = unsafe { self.swip.as_buffer_frame() };

        match self.state {
            LockState::Exclusive => return,
            LockState::Shared => {
                unsafe { frame.upgrade_latch(); }
                
            }

            LockState::Optimistic => {
                frame.latch_exclusive();
            }
        }

        self.state = LockState::Exclusive;
    }

    pub(crate) fn downgrade(&mut self) {
        let frame = unsafe { self.swip.as_buffer_frame() };

        match self.state {
            LockState::Exclusive => {
                unsafe { frame.unlatch_exclusive() }
            },

            LockState::Shared => {
                unsafe { frame.unlatch_shared(); }
                
            }

            LockState::Optimistic => return (),
        }

        self.state = LockState::Optimistic;
    }
}

impl<T> Drop for PageGuard<T> {
    fn drop(&mut self) {
        // Safety: SharedGuard takes a shared latch when created, thus
        // we know we own a latch.
        unsafe {
            let frame = self.swip.as_buffer_frame();

            match self.state {
                LockState::Exclusive => frame.unlatch_exclusive(),
                LockState::Shared => frame.unlatch_shared(),
                _ => ()
            }
        }
    }
}

impl<T> Deref for PageGuard<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.data_structure()
    }
}

impl<T> DerefMut for PageGuard<T>  {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.data_structure_mut()
    }
}
