use std::ops::Deref;

use crate::buffer_manager::buffer_frame::BufferFrame;
use crate::buffer_manager::swip::Swip;

pub(crate) struct OptimisticGuard<'a, T> {
    frame: &'a BufferFrame<'a>,
    initial_version: u64,
    _marker: std::marker::PhantomData<T>,
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub(crate) enum OptimisticError {
    Conflict
}

/// This is a very unsafe by default type. The underlying data can be modified
/// while reading. Thus, it is very important to validate the read before
/// dereferencing any pointers.
impl<'a, T> OptimisticGuard<'a, T> {
    pub(crate) fn new(swip: &'a Swip<T>) -> Result<Self, OptimisticError> {
        let frame = unsafe { swip.as_buffer_frame() };
        let initial_version = frame.version();

        if initial_version & (1 << 63) > 0 {
            return Err(OptimisticError::Conflict);
        }

        Ok(OptimisticGuard {
            frame,
            initial_version,
            _marker: Default::default()
        })
    }

    pub(crate) fn validate(&self) -> Result<(), OptimisticError> {
        if self.initial_version == self.frame.version() {
            Ok(())
        } else {
            Err(OptimisticError::Conflict)
        }
    }

    #[inline]
    pub(crate) fn version(&self) -> u64 {
        self.initial_version
    }

    pub(crate) fn data_structure(&self) -> &T {
        let ptr = self.frame.page.data.as_ptr();
        unsafe { std::mem::transmute(ptr.as_ref().unwrap()) }
    }
}

impl<'a, T> Deref for OptimisticGuard<'a, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.data_structure()
    }
}
