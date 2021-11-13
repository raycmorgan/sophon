

#[repr(C)]
struct NodeHeader {

}

struct Node {
    db_version: u8,
    upper: Option<Swip<Node>>,
}



struct Swip<T> {
    ptr: usize,
    _marker: std::marker::PhantomData<T>,
}

impl<T> Swip<T> {
    fn cast(&self) -> &T {
        unsafe { std::mem::transmute(&self) }
    }
}
