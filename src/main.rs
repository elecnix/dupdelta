//! Thin entry point.
//!
//! Every behaviour lives in the library so that it is testable; this shim
//! exists only to give the library a command line. It is excluded from the
//! coverage gate for exactly that reason — see `scripts/coverage.sh`. Keep it
//! trivial enough that the exclusion stays honest.

fn main() {
    println!("dupdelta {}", env!("CARGO_PKG_VERSION"));
}
