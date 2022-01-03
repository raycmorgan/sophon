use stackvec::StackVec;
use crate::buffer_manager::{swip::Swip, buffer_frame::PageGuard};

struct NodePath<T> {
    stack: StackVec<[PathItem<T>; 32]>,
}

pub(crate) struct PathItem<T> {
    swip: Swip<T>,
    version: u64,
}

pub(crate) struct LockPath<T> {
    stack: StackVec<[PageGuard<T>; 32]>,
}
