//! `editor-ctl` — command-line client for the editor's remote-control channel
//! (`docs/REMOTE.md`).
//!
//! Connects to the port published in `.redlilium/editor.port` (the editor must
//! run with `REDLILIUM_REMOTE=1`), sends one command, prints the response line.
//!
//! Subcommands are sugar over the RON protocol; `raw` sends an arbitrary
//! envelope:
//!
//! ```text
//! editor-ctl state
//! editor-ctl inspect 12@3
//! editor-ctl edit 12@3 Transform '(translation: (5.0, 0.7, -2.0))'
//! editor-ctl select 12@3
//! editor-ctl screenshot /tmp/scene.png
//! editor-ctl wait-assets 5000
//! editor-ctl logs 0
//! editor-ctl undo
//! editor-ctl raw '(id: 1, cmd: "hello")'
//! ```

use std::io::{BufRead, BufReader, Write};
use std::net::TcpStream;
use std::time::Duration;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: editor-ctl <command> [args…]  (see docs/REMOTE.md)");
        std::process::exit(2);
    }

    let envelope = match build_envelope(&args) {
        Ok(e) => e,
        Err(msg) => {
            eprintln!("{msg}");
            std::process::exit(2);
        }
    };

    let port_file = std::env::var("REDLILIUM_PORT_FILE")
        .unwrap_or_else(|_| ".redlilium/editor.port".to_string());
    let port: u16 = match std::fs::read_to_string(&port_file) {
        Ok(text) => match text.trim().parse() {
            Ok(p) => p,
            Err(_) => {
                eprintln!("{port_file}: not a port number");
                std::process::exit(1);
            }
        },
        Err(e) => {
            eprintln!("{port_file}: {e}\n(is the editor running with REDLILIUM_REMOTE=1?)");
            std::process::exit(1);
        }
    };

    let mut stream = match TcpStream::connect(("127.0.0.1", port)) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("connect 127.0.0.1:{port}: {e}");
            std::process::exit(1);
        }
    };
    let _ = stream.set_read_timeout(Some(Duration::from_secs(30)));

    if stream.write_all(envelope.as_bytes()).is_err() || stream.write_all(b"\n").is_err() {
        eprintln!("send failed");
        std::process::exit(1);
    }

    let mut reader = BufReader::new(stream);
    let mut line = String::new();
    match reader.read_line(&mut line) {
        Ok(0) => {
            eprintln!("connection closed without a response");
            std::process::exit(1);
        }
        Ok(_) => {
            print!("{line}");
            if line.contains("ok: false") {
                std::process::exit(1);
            }
        }
        Err(e) => {
            eprintln!("read: {e}");
            std::process::exit(1);
        }
    }
}

/// Escape a string for RON.
fn q(s: &str) -> String {
    format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""))
}

fn build_envelope(args: &[String]) -> Result<String, String> {
    let cmd = args[0].as_str();
    let rest = &args[1..];
    let need = |n: usize, usage: &str| -> Result<(), String> {
        if rest.len() < n {
            Err(format!("usage: editor-ctl {usage}"))
        } else {
            Ok(())
        }
    };
    Ok(match cmd {
        "raw" => {
            need(1, "raw '<RON envelope>'")?;
            rest[0].clone()
        }
        "hello" | "state" | "undo" | "redo" | "shutdown" => {
            format!("(id: 1, cmd: {})", q(cmd))
        }
        "step" => {
            let n = rest.first().map(String::as_str).unwrap_or("1");
            format!("(id: 1, cmd: \"step\", n: {n})")
        }
        "inspect" => {
            need(1, "inspect <entity>")?;
            format!("(id: 1, cmd: \"inspect\", entity: {})", q(&rest[0]))
        }
        "select" => {
            need(1, "select <entity>…")?;
            let list: Vec<String> = rest.iter().map(|e| q(e)).collect();
            format!("(id: 1, cmd: \"select\", entities: [{}])", list.join(", "))
        }
        "edit" => {
            need(3, "edit <entity> <Component> '<natural RON>'")?;
            format!(
                "(id: 1, cmd: \"edit_component\", entity: {}, component: {}, data: {})",
                q(&rest[0]),
                q(&rest[1]),
                rest[2]
            )
        }
        "add" | "remove" => {
            need(2, "add|remove <entity> <Component>")?;
            let wire = if cmd == "add" {
                "add_component"
            } else {
                "remove_component"
            };
            format!(
                "(id: 1, cmd: {}, entity: {}, component: {})",
                q(wire),
                q(&rest[0]),
                q(&rest[1])
            )
        }
        "screenshot" => {
            need(1, "screenshot <path.png>")?;
            format!(
                "(id: 1, cmd: \"screenshot\", target: \"scene\", path: {})",
                q(&rest[0])
            )
        }
        "wait-assets" => {
            let timeout = rest.first().map(String::as_str).unwrap_or("10000");
            format!("(id: 1, cmd: \"wait\", for: \"assets_idle\", timeout_ms: {timeout})")
        }
        "wait-frames" => {
            need(1, "wait-frames <n>")?;
            format!("(id: 1, cmd: \"wait\", for: \"frames\", n: {})", rest[0])
        }
        "logs" => {
            let since = rest.first().map(String::as_str).unwrap_or("0");
            format!("(id: 1, cmd: \"logs\", since: {since})")
        }
        other => return Err(format!("unknown subcommand '{other}'")),
    })
}
