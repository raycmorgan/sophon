use super::{
    buffer_frame::{LatchStrategy, PageGuard},
    Swipable,
};
use crate::buffer_manager::buffer_frame::BufferFrame;
use std::fmt::Debug;

#[derive(Copy, Clone, Debug, PartialEq)]
pub(crate) enum OptimisticError {
    Conflict,
}

pub(crate) struct Swip<T: Swipable> {
    ptr: usize, // either pid or pointer to BufferFrame
    _marker: std::marker::PhantomData<T>,
}

impl<T: Swipable> From<usize> for Swip<T> {
    fn from(ptr: usize) -> Self {
        Swip {
            ptr,
            _marker: Default::default(),
        }
    }
}

impl<T: Swipable> From<Swip<T>> for usize {
    fn from(swip: Swip<T>) -> Self {
        swip.ptr
    }
}

impl<T: Swipable> Clone for Swip<T> {
    fn clone(&self) -> Self {
        Swip {
            ptr: self.ptr,
            _marker: self._marker,
        }
    }
}

impl<T: Swipable> Copy for Swip<T> {}

impl<T: Swipable> Debug for Swip<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Swip").field("ptr", &self.ptr).finish()
    }
}

impl<T: Swipable> PartialEq for Swip<T> {
    fn eq(&self, other: &Self) -> bool {
        self.ptr == other.ptr
    }
}

impl<T: Swipable> Swip<T> {
    pub(crate) fn new(ptr: usize) -> Self {
        Swip {
            ptr,
            _marker: Default::default(),
        }
    }

    #[inline]
    pub(crate) fn value(&self) -> u64 {
        self.ptr as u64
    }

    #[inline]
    pub(crate) fn value_bytes(&self) -> [u8; 8] {
        (self.ptr as u64).to_ne_bytes()
    }

    #[inline]
    pub(crate) fn is_swizzled(&self) -> bool {
        self.ptr & 1 == 0
    }

    /// Safety: Caller must ensure the underlying BufferFrame is live
    #[inline]
    pub(crate) unsafe fn as_buffer_frame(&self) -> &BufferFrame {
        // debug_assert swizzle
        (self.ptr as *const BufferFrame).as_ref().unwrap()
    }

    /// Safety: Caller must ensure the underlying BufferFrame is live
    #[inline]
    pub(crate) unsafe fn as_buffer_frame_mut(&self) -> &mut BufferFrame {
        // debug_assert swizzle
        (self.ptr as *mut BufferFrame).as_mut().unwrap()
    }

    #[inline]
    pub(crate) fn page_id(&self) -> u64 {
        // TODO: mask
        self.ptr as u64
    }

    #[inline]
    pub(crate) fn coupled_page_guard<T2: Swipable>(
        self,
        parent: Option<&PageGuard<T2>>,
        strategy: LatchStrategy,
    ) -> Result<PageGuard<T>, OptimisticError> {
        let guard = match strategy {
            LatchStrategy::OptimisticSpin => PageGuard::new_optimistic_spin(self),
            LatchStrategy::OptimisticOrShared => PageGuard::new_optimistic_or_shared(self),
            LatchStrategy::OptimisticOrExclusive => PageGuard::new_optimistic_or_exclusive(self),
            LatchStrategy::Exclusive => PageGuard::new_exclusive(self),
            LatchStrategy::Shared => PageGuard::new_shared(self),
            LatchStrategy::Yolo => PageGuard::new_yolo(self),
            LatchStrategy::OptimisticOrConflict => unimplemented!(),
        };

        if let Some(p_guard) = parent {
            if !p_guard.is_valid() {
                return Err(OptimisticError::Conflict);
            }
        }

        Ok(guard)
    }
}
