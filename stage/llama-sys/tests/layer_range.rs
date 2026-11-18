//! Check that the native ABI exposes writable layer bounds and that their
//! defaults select the full layer range.

use publicompute_llama_sys::*;

#[test]
fn default_model_params_cover_the_full_range() {
    let p: llama_model_params = unsafe { llama_model_default_params() };
    let (a, b): (i32, i32) = (p.layer_start, p.layer_end);
    assert_eq!(a, 0, "default layer_start");
    assert_eq!(b, -1, "default layer_end (-1 selects the full range)");
}

#[test]
fn layer_bounds_are_writable() {
    let mut p: llama_model_params = unsafe { llama_model_default_params() };
    p.layer_start = 3;
    p.layer_end = 7;
    assert_eq!((p.layer_start, p.layer_end), (3, 7));
}
