//! Thin entry point.
//!
//! Every behaviour lives in the library so that it is testable; this shim only
//! gives the library a command line and turns an error into an exit status. It
//! is excluded from the coverage gate for exactly that reason — see
//! `scripts/coverage.sh`. Keep it trivial enough that the exclusion stays
//! honest.

use std::process::ExitCode;

use clap::Parser;
use dupdelta::cli::Cli;

fn main() -> ExitCode {
    // `parse` handles `--help`, `--version` and bad arguments itself, exiting
    // with clap's conventional status.
    let cli = Cli::parse();
    let mut stdout = std::io::stdout().lock();

    match cli.run(&mut stdout) {
        // Findings are reported, never fatal: a hard gate on a similarity
        // heuristic gets worked around or switched off. See the README.
        Ok(_) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("dupdelta: {error}");
            ExitCode::FAILURE
        }
    }
}
