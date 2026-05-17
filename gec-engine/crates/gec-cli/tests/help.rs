use std::process::Command;

#[test]
fn cli_help_prints_subcommands() {
    let exe = env!("CARGO_BIN_EXE_gec-cli");
    let out = Command::new(exe).arg("--help").output().expect("ran");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("apply-edits"), "stdout: {stdout}");
}
