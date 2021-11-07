#![feature(int_log)]
#![feature(slice_as_chunks)]

#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

// assert_eq!(8, std::mem::size_of::<parking_log::RWLock<()>>());

mod btree;
mod buffer_manager;

trait DiskManager {
    fn capacity(&self) -> usize;
    fn base_page_size(&self) -> usize;

    fn allocate_page(&self, size_class: usize) -> Result<Unswizzle, u8>;
    fn deallocate_page();

    fn read_page();
    fn write_page();
    fn fsync();
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Swizzle(usize);

impl Swizzle {
    fn new(id: usize, size_class: usize) -> Self {
        // assert!(ptr & 1 == 0, "Expected swizzled pointer, received unswizzled.");
        assert!(size_class <= 63);
        Swizzle((id << 7) | (size_class << 1))
    }

    // fn as_ptr(&self) -> *mut u8 {
    //     self.0 as *mut u8
    // }

    fn page_id(&self) -> usize {
        self.0 >> 7
    }

    fn size_class(&self) -> usize {
        (self.0 & 0b1111110) >> 1
    }

    fn as_u64(&self) -> u64 {
        self.0 as u64
    }
}

impl From<Swizzle> for usize {
    fn from(swizzle: Swizzle) -> usize {
        swizzle.0
    }
}

impl From<Swizzle> for u64 {
    fn from(swizzle: Swizzle) -> u64 {
        swizzle.as_u64()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Unswizzle(usize);

impl Unswizzle {
    fn from_parts(id: usize, size_class: usize) -> Unswizzle {
        assert!(size_class <= 63);
        Unswizzle((id << 7) | (size_class << 1) | 1)
    }

    fn page_id(&self) -> usize {
        self.0 >> 7
    }

    fn size_class(&self) -> usize {
        (self.0 & 0b1111110) >> 1
    }

    fn as_u64(&self) -> u64 {
        self.0 as u64
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Swip {
    Swizzle(Swizzle),
    Unswizzle(Unswizzle),
}

impl From<u64> for Swip {
    fn from(data: u64) -> Self {
        if data & 1 == 1 {
            Swip::Unswizzle(Unswizzle(data.try_into().unwrap()))
        } else {
            Swip::Swizzle(Swizzle(data.try_into().unwrap()))
        }
    }
}


#[cfg(test)]
mod tests {
    use crate::Unswizzle;

    #[test]
    fn it_works() {
        println!("Unswizzle: {:?}", Unswizzle::from_parts(10, 0));
    }
}


