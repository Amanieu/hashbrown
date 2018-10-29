// NOTE: This is an experimental implementation for ARM NEON which we don't use
// since it has no performance advantage over the scalar variant. It is only
// kept around as a reference.

extern crate packed_simd;

use core::mem;
use core::slice;
use self::packed_simd::{m8x16, u16x8, i8x16, u8x8, Cast, IntoBits};
use raw::bitmask::BitMask;
use raw::EMPTY;

pub type BitMaskWord = u64;
pub const BITMASK_SHIFT: u64 = 2;
pub const BITMASK_MASK: u64 = 0x8888888888888888;

#[inline]
fn narrow_mask(mask: m8x16) -> BitMask {
    // NEON can perform right-shift & narrow in a single instruction
    let mask16: u16x8 = mask.into_bits();
    let narrowed: u8x8 = (mask16 >> 4).cast();
    let result: u64 = unsafe { mem::transmute(narrowed) };
    BitMask(result & BITMASK_MASK)
}

/// Abstraction over a group of control bytes which can be scanned in
/// parallel.
///
/// This implementation uses a 128-bit NEON value.
pub struct Group(i8x16);

impl Group {
    /// Number of bytes in the group.
    pub const WIDTH: usize = mem::size_of::<Self>();

    /// Returns a full group of empty bytes, suitable for use as the initial
    /// value for an empty hash table.
    ///
    /// This is guaranteed to be aligned to the group size.
    #[inline]
    pub fn static_empty() -> &'static [u8] {
        #[repr(C)]
        struct Dummy {
            _align: [i8x16; 0],
            bytes: [u8; Group::WIDTH],
        };
        const DUMMY: Dummy = Dummy {
            _align: [],
            bytes: [EMPTY; Group::WIDTH],
        };
        &DUMMY.bytes
    }

    /// Loads a group of bytes starting at the given address.
    #[inline]
    pub unsafe fn load(ptr: *const u8) -> Group {
        let slice = slice::from_raw_parts(ptr as *const i8, 16);
        Group(i8x16::from_slice_unaligned_unchecked(slice))
    }

    /// Loads a group of bytes starting at the given address, which must be
    /// aligned to `WIDTH`.
    #[inline]
    pub unsafe fn load_aligned(ptr: *const u8) -> Group {
        let slice = slice::from_raw_parts(ptr as *const i8, 16);
        Group(i8x16::from_slice_aligned_unchecked(slice))
    }

    /// Stores the group of bytes to the given address, which must be
    /// aligned to `WIDTH`.
    #[inline]
    pub unsafe fn store_aligned(&self, ptr: *mut u8) {
        let slice = slice::from_raw_parts_mut(ptr as *mut i8, 16);
        self.0.write_to_slice_aligned_unchecked(slice);
    }

    /// Returns a `BitMask` indicating all bytes in the group which have
    /// the given value.
    #[inline]
    pub fn match_byte(&self, byte: u8) -> BitMask {
        narrow_mask(self.0.eq(i8x16::splat(byte as i8)))
    }

    /// Returns a `BitMask` indicating all bytes in the group which are
    /// `EMPTY`.
    #[inline]
    pub fn match_empty(&self) -> BitMask {
        self.match_byte(EMPTY)
    }

    /// Returns a `BitMask` indicating all bytes in the group which are
    /// `EMPTY` pr `DELETED`.
    #[inline]
    pub fn match_empty_or_deleted(&self) -> BitMask {
        narrow_mask(self.0.lt(i8x16::splat(0)))
    }

    /// Performs the following transformation on all bytes in the group:
    /// - `EMPTY => EMPTY`
    /// - `DELETED => EMPTY`
    /// - `FULL => DELETED`
    #[inline]
    pub fn convert_special_to_empty_and_full_to_deleted(&self) -> Group {
        let special: i8x16 = self.0.lt(i8x16::splat(0)).into_bits();
        Group(special | i8x16::splat(0x80u8 as i8))
    }
}
