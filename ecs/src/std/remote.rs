//! Remote-control transport: a line-based TCP channel served by the engine's
//! own async IO runtime (`docs/REMOTE.md`).
//!
//! This module is deliberately **dumb**: it accepts localhost connections,
//! frames newline-delimited text both ways, and hands `(connection, line)`
//! pairs to whoever pumps the [`RemoteTransport`] resource. Protocol semantics
//! (parsing commands, building actions, replying) live in the consumer — the
//! editor's command pump — so this same transport can later serve game debug
//! consoles.
//!
//! Networking runs entirely on the ECS IO runtime (the same tokio runtime that
//! drives asset IO): [`RemoteServe`] spawns the accept loop once via
//! `ctx.io()`; each connection gets a read task (lines → the shared inbound
//! channel) and a write task (per-connection outbound channel). No dedicated
//! threads.

use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::mpsc;
use std::time::Duration;

use crate::system::SystemError;
use crate::{IoRunner, System, SystemContext};

/// An inbound line from a remote client.
#[derive(Debug)]
pub struct RemoteLine {
    /// Connection id (unique per accepted connection).
    pub conn: u64,
    /// The raw line (one request), without the trailing newline.
    pub line: String,
}

/// The remote-control channel (an ECS resource). Insert it configured with
/// [`new`](Self::new), register [`RemoteServe`] in the `Startup` schedule, and
/// pump [`try_recv`](Self::try_recv) / [`send`](Self::send) from the protocol
/// layer once per frame.
pub struct RemoteTransport {
    port_file: PathBuf,
    /// Inbound lines from all connections (async reader tasks → pump).
    incoming_rx: Mutex<mpsc::Receiver<RemoteLine>>,
    /// Lines pulled off the channel by [`wait_incoming`](Self::wait_incoming)
    /// while blocking; drained ahead of the channel by `try_recv`.
    buffered: Mutex<VecDeque<RemoteLine>>,
    /// Kept to clone into the accept loop at serve time.
    incoming_tx: mpsc::Sender<RemoteLine>,
    /// Outbound lines to route to per-connection writers (pump → router task).
    outgoing_tx: tokio::sync::mpsc::UnboundedSender<(u64, String)>,
    /// Taken by `RemoteServe` when it spawns the router.
    outgoing_rx: Mutex<Option<tokio::sync::mpsc::UnboundedReceiver<(u64, String)>>>,
    started: Mutex<bool>,
}

impl RemoteTransport {
    /// Create the (not yet listening) transport. `port_file` is where the
    /// bound port is published for clients (e.g. `.redlilium/editor.port`).
    pub fn new(port_file: impl Into<PathBuf>) -> Self {
        let (incoming_tx, incoming_rx) = mpsc::channel();
        let (outgoing_tx, outgoing_rx) = tokio::sync::mpsc::unbounded_channel();
        Self {
            port_file: port_file.into(),
            incoming_rx: Mutex::new(incoming_rx),
            buffered: Mutex::new(VecDeque::new()),
            incoming_tx,
            outgoing_tx,
            outgoing_rx: Mutex::new(Some(outgoing_rx)),
            started: Mutex::new(false),
        }
    }

    /// Drain inbound lines received since the last pump.
    pub fn try_recv(&self) -> Vec<RemoteLine> {
        let mut lines: Vec<RemoteLine> = self
            .buffered
            .lock()
            .expect("remote buffered poisoned")
            .drain(..)
            .collect();
        let rx = self.incoming_rx.lock().expect("remote incoming poisoned");
        while let Ok(line) = rx.try_recv() {
            lines.push(line);
        }
        lines
    }

    /// Block up to `timeout` for an inbound line. Returns `true` if at least
    /// one line is available (it stays queued for the next [`try_recv`]).
    /// Lets a headless shell sleep between remote commands instead of
    /// free-running frames (tick-on-demand).
    pub fn wait_incoming(&self, timeout: Duration) -> bool {
        if !self
            .buffered
            .lock()
            .expect("remote buffered poisoned")
            .is_empty()
        {
            return true;
        }
        let received = {
            let rx = self.incoming_rx.lock().expect("remote incoming poisoned");
            rx.recv_timeout(timeout)
        };
        match received {
            Ok(line) => {
                self.buffered
                    .lock()
                    .expect("remote buffered poisoned")
                    .push_back(line);
                true
            }
            Err(_) => false,
        }
    }

    /// Queue a response line for a connection (newline appended on the wire).
    pub fn send(&self, conn: u64, line: String) {
        let _ = self.outgoing_tx.send((conn, line));
    }
}

/// One-shot startup system: binds a localhost listener on the IO runtime,
/// publishes the port to the transport's port file, and spawns the accept /
/// read / write tasks. No-op without a [`RemoteTransport`] resource.
pub struct RemoteServe;

impl System for RemoteServe {
    type Result = ();
    fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) -> Result<Self::Result, SystemError> {
        let world = ctx.raw_world();
        if !world.has_resource::<RemoteTransport>() {
            return Ok(());
        }
        let transport = world.resource::<RemoteTransport>();
        {
            let mut started = transport.started.lock().expect("remote started poisoned");
            if *started {
                return Ok(());
            }
            *started = true;
        }
        let incoming = transport.incoming_tx.clone();
        let Some(outgoing) = transport
            .outgoing_rx
            .lock()
            .expect("remote outgoing poisoned")
            .take()
        else {
            return Ok(());
        };
        let port_file = transport.port_file.clone();

        drop(ctx.io().run(serve(port_file, incoming, outgoing)));
        Ok(())
    }
}

/// The server task: bind, publish the port, route outbound lines, accept
/// connections and spawn their read/write loops.
async fn serve(
    port_file: PathBuf,
    incoming: mpsc::Sender<RemoteLine>,
    mut outgoing: tokio::sync::mpsc::UnboundedReceiver<(u64, String)>,
) {
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

    let listener = match tokio::net::TcpListener::bind("127.0.0.1:0").await {
        Ok(l) => l,
        Err(e) => {
            log::error!("remote: failed to bind: {e}");
            return;
        }
    };
    let port = match listener.local_addr() {
        Ok(addr) => addr.port(),
        Err(e) => {
            log::error!("remote: no local addr: {e}");
            return;
        }
    };
    if let Some(dir) = port_file.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    if let Err(e) = std::fs::write(&port_file, port.to_string()) {
        log::error!("remote: failed to publish port file: {e}");
        return;
    }
    log::info!(
        "remote: listening on 127.0.0.1:{port} ({})",
        port_file.display()
    );

    // Per-connection outbound senders, shared between the router and acceptor.
    let writers: std::sync::Arc<
        tokio::sync::Mutex<HashMap<u64, tokio::sync::mpsc::UnboundedSender<String>>>,
    > = Default::default();

    // Router: fan outbound lines to their connection's writer.
    let router_writers = writers.clone();
    tokio::spawn(async move {
        while let Some((conn, line)) = outgoing.recv().await {
            let writers = router_writers.lock().await;
            if let Some(tx) = writers.get(&conn) {
                let _ = tx.send(line);
            }
        }
    });

    // Accept loop.
    let mut next_conn: u64 = 1;
    loop {
        let (stream, _) = match listener.accept().await {
            Ok(pair) => pair,
            Err(e) => {
                log::warn!("remote: accept failed: {e}");
                continue;
            }
        };
        let conn = next_conn;
        next_conn += 1;
        let (read_half, mut write_half) = stream.into_split();

        // Writer task: connection's outbound queue → socket.
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<String>();
        writers.lock().await.insert(conn, tx);
        tokio::spawn(async move {
            while let Some(line) = rx.recv().await {
                if write_half.write_all(line.as_bytes()).await.is_err()
                    || write_half.write_all(b"\n").await.is_err()
                {
                    break;
                }
            }
        });

        // Reader task: socket lines → the shared inbound channel.
        let incoming = incoming.clone();
        let cleanup = writers.clone();
        tokio::spawn(async move {
            let mut lines = BufReader::new(read_half).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                if line.trim().is_empty() {
                    continue;
                }
                if incoming.send(RemoteLine { conn, line }).is_err() {
                    break; // transport dropped — editor shut down
                }
            }
            cleanup.lock().await.remove(&conn);
        });
    }
}
