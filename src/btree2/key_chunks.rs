use std::ops::Range;

use stackvec::StackVec;


#[derive(Debug, Clone)]
pub(crate) struct KeyChunks<'a> {
    chunks: StackVec<[&'a [u8]; 4]>
}

impl<'a> KeyChunks<'a> {
    fn new(chunks: &[&'a [u8]]) -> Self {
        let mut v = StackVec::default();

        for c in chunks.iter() {
            v.try_push(*c).expect("Cannot have more than 4 chunks");
        }

        KeyChunks {
            chunks: v
        }
    }
}

impl<'a> KeyChunks<'a> {
    #[inline]
    pub(crate) fn len(&self) -> usize {
        // self.chunks.iter().fold(0, |a, i| a+i.len())
        self.chunks.iter().fold(0, |a, i| a+i.len())
    }

    #[inline]
    pub(crate) fn copy_to(&self, dst: &mut [u8], range: Range<usize>) {
        let mut s = range.start;
        let mut e = range.end - range.start;
        let mut dsti = 0;
        
        for c in self.chunks.iter() {
            if e == 0 {
                break;
            }

            if s > c.len() {
                s -= c.len();
                continue;
            }

            let chunk = &c[s..(s+e).min(c.len())];
            dst[dsti..dsti+chunk.len()].copy_from_slice(chunk);
            s = 0;
            e = e.saturating_sub(c.len());
        }
    }

    #[inline]
    pub(crate) fn to_vec(&self) -> Vec<u8> {
        let mut v = Vec::with_capacity(self.len());
        self.copy_to(&mut v, 0..self.len());
        v
    }
}

impl<'a> PartialEq<&[u8]> for KeyChunks<'a> {
    fn eq(&self, other: &&[u8]) -> bool {
        if other.len() != self.len() {
            return false;
        }

        let mut p = 0;
        self.chunks.iter().all(|c| {
            let s = p;
            p += c.len();
            *c == &other[s..p]
        })

        // self.prefix == &other[self.prefix_range()]
        //     && self.slot_key == &other[self.slot_key_range()]
        //     && self.suffix == &other[self.suffix_range()]
    }
}

impl<'a> PartialOrd<&[u8]> for KeyChunks<'a> {
    fn partial_cmp(&self, other: &&[u8]) -> Option<std::cmp::Ordering> {
        use std::cmp::Ordering;

        let mut p = 0;
        for c in self.chunks.iter() {

            let s = p;
            p = (p + c.len()).min(other.len());
            
            match (*c).partial_cmp(&other[s..p]) {
                None => unreachable!(),
                Some(Ordering::Greater) => return Some(Ordering::Greater),
                Some(Ordering::Less) => return Some(Ordering::Less),
                Some(Ordering::Equal) => (),
            }
        }

        Some(Ordering::Equal)
    }
}

// impl<'a> Index<Range<usize>> for KeyChunks<'a> {
//     type Output = KeyChunks<'a>;

//     fn index(&self, index: Range<usize>) -> &Self::Output {
//         let mut v = StackVec::default();
//         let mut s = index.start;
//         let mut e = index.end - index.start;
        
//         for c in self.chunks.iter() {
//             if e == 0 {
//                 break;
//             }

//             if s > c.len() {
//                 s -= c.len();
//                 continue;
//             }

//             let chunk = &c[s..(s+e).min(c.len())];
//             s = 0;
//             e = e.saturating_sub(c.len());
//             v.try_push(chunk);
//         }

//         &KeyChunks {
//             chunks: v,
//         }
//     }
// }



#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunks() {
        #[test]
    fn test_key_chunks() {
        let data = b"foobarbaz";
        let kc = KeyChunks::new(&[ &data[0..2], &data[2..6], &data[6..] ]);

        assert_eq!(kc, b"foobarbaz");
        assert!(kc < b"z");
        assert!(kc > b"a");
        assert!(kc < b"foobarbaza");
        assert!(kc > b"foobarbar");
        assert!(kc > b"foobar");
        assert!(kc > b"fo");

        // let sub = &kc[3..6];
        // assert_eq!(kc, b"bar");
    }
    }
}