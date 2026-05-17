use anyhow::Result;
use clap::{Parser, Subcommand};
use gec_aligner::apply_edits;
use gec_core::Edit;
use std::io::Read;

#[derive(Parser)]
#[command(name = "gec-cli", about = "GEC engine command-line interface")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Read text on stdin and a JSON edits array from --edits, write
    /// the corrected text to stdout.
    ApplyEdits {
        #[arg(long)]
        edits: String,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::ApplyEdits { edits } => {
            let edits: Vec<Edit> = serde_json::from_str(&edits)?;
            let mut input = String::new();
            std::io::stdin().read_to_string(&mut input)?;
            print!("{}", apply_edits(&input, &edits));
            Ok(())
        }
    }
}
