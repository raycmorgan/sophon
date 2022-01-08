#![feature(int_log)]
#![feature(slice_as_chunks)]
#![feature(int_abs_diff)]

#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

// assert_eq!(8, std::mem::size_of::<parking_log::RWLock<()>>());

// mod btree;
mod btree2;
mod buffer_manager;

trait DiskManager: Sync + Send {
    fn capacity(&self) -> usize;
    fn base_page_size(&self) -> usize;

    fn allocate_page(&self, size_class: usize) -> Result<usize, DiskManagerAllocError>;
    fn deallocate_page(&self);

    fn read_page(&self);
    fn write_page(&self);
    fn fsync(&self);
}

#[derive(Debug, Copy, Clone)]
enum DiskManagerAllocError {
    OutOfSpace
}

// #[derive(Debug, Clone, Copy, PartialEq, Eq)]
// struct Swizzle(usize);

// impl Swizzle {
//     fn new(id: usize, size_class: usize) -> Self {
//         // assert!(ptr & 1 == 0, "Expected swizzled pointer, received unswizzled.");
//         assert!(size_class <= 63);
//         Swizzle((id << 7) | (size_class << 1))
//     }

//     // fn as_ptr(&self) -> *mut u8 {
//     //     self.0 as *mut u8
//     // }

//     fn page_id(&self) -> usize {
//         self.0 >> 7
//     }

//     fn size_class(&self) -> usize {
//         (self.0 & 0b1111110) >> 1
//     }

//     fn as_u64(&self) -> u64 {
//         self.0 as u64
//     }
// }

// impl From<Swizzle> for usize {
//     fn from(swizzle: Swizzle) -> usize {
//         swizzle.0
//     }
// }

// impl From<Swizzle> for u64 {
//     fn from(swizzle: Swizzle) -> u64 {
//         swizzle.as_u64()
//     }
// }

// impl From<(usize, usize)> for Swizzle {
//     fn from(parts: (usize, usize)) -> Self {
//         Swizzle::new(parts.0, parts.1)
//     }
// }

// impl From<Unswizzle> for Swizzle {
//     fn from(u: Unswizzle) -> Self {
//         Swizzle::new(u.page_id(), u.size_class())
//     }
// }

// impl From<Swizzle> for Unswizzle {
//     fn from(s: Swizzle) -> Self {
//         Unswizzle::from_parts(s.page_id(), s.size_class())
//     }
// }

// #[derive(Debug, Clone, Copy, PartialEq, Eq)]
// struct Unswizzle(usize);

// impl Unswizzle {
//     fn from_parts(id: usize, size_class: usize) -> Unswizzle {
//         assert!(size_class <= 63);
//         Unswizzle((id << 7) | (size_class << 1) | 1)
//     }

//     fn page_id(&self) -> usize {
//         self.0 >> 7
//     }

//     fn size_class(&self) -> usize {
//         (self.0 & 0b1111110) >> 1
//     }

//     fn as_u64(&self) -> u64 {
//         self.0 as u64
//     }
// }

// impl From<(usize, usize)> for Unswizzle {
//     fn from(parts: (usize, usize)) -> Self {
//         Unswizzle::from_parts(parts.0, parts.1)
//     }
// }

// #[derive(Debug, Clone, Copy, PartialEq, Eq)]
// enum Swip {
//     Swizzle(Swizzle),
//     Unswizzle(Unswizzle),
// }

// impl Swip {
//     fn page_id(&self) -> usize {
//         match self {
//             Swip::Swizzle(s) => s.page_id(),
//             Swip::Unswizzle(s) => s.page_id(),
//         }
//     }
// }

// impl From<u64> for Swip {
//     fn from(data: u64) -> Self {
//         if data & 1 == 1 {
//             Swip::Unswizzle(Unswizzle(data.try_into().unwrap()))
//         } else {
//             Swip::Swizzle(Swizzle(data.try_into().unwrap()))
//         }
//     }
// }

// impl From<&[u8]> for Swip {
//     fn from(input: &[u8]) -> Self {
//         let data = u64::from_le_bytes(input[0..8].try_into().unwrap());
//         data.into()
//     }
// }


#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    
    #[derive(Default)]
    pub struct FakeDiskManager {
        next_page: AtomicUsize,
    }

    impl DiskManager for FakeDiskManager {
        fn capacity(&self) -> usize { 4096 * 1000 }
        fn base_page_size(&self) -> usize { crate::buffer_manager::buffer_frame::PAGE_SIZE }

        fn allocate_page(&self, size_class: usize) -> Result<usize, DiskManagerAllocError> {
            let next = self.next_page.fetch_add(1, Ordering::SeqCst);
            // Ok(Unswizzle::from_parts(next, size_class))
            Ok(next)
        }

        fn deallocate_page(&self) {}

        fn read_page(&self) {}
        fn write_page(&self) {}
        fn fsync(&self) {}
    }

    #[test]
    fn it_works() {
        // println!("Unswizzle: {:?}", Unswizzle::from_parts(10, 0));
    }
}


