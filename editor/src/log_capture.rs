use std::collections::VecDeque;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

/// A single captured log entry.
pub struct LogEntry {
    pub level: log::Level,
    pub target: String,
    pub message: String,
    pub timestamp: Instant,
}

/// Ring buffer of captured log entries.
pub struct LogBuffer {
    entries: VecDeque<LogEntry>,
    max_capacity: usize,
}

impl LogBuffer {
    fn new(max_capacity: usize) -> Self {
        Self {
            entries: VecDeque::with_capacity(max_capacity.min(1024)),
            max_capacity,
        }
    }

    pub fn entries(&self) -> &VecDeque<LogEntry> {
        &self.entries
    }

    fn push(&mut self, entry: LogEntry) {
        if self.entries.len() >= self.max_capacity {
            self.entries.pop_front();
        }
        self.entries.push_back(entry);
    }

    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

/// Global shared log buffer.
static LOG_BUFFER: OnceLock<Arc<Mutex<LogBuffer>>> = OnceLock::new();

/// Returns the shared log buffer handle.
pub fn log_buffer() -> Arc<Mutex<LogBuffer>> {
    LOG_BUFFER
        .get()
        .expect("log_capture::install() must be called first")
        .clone()
}

/// Custom logger that wraps `env_logger` and captures entries to the ring buffer.
struct LogCapture {
    inner: env_logger::Logger,
    buffer: Arc<Mutex<LogBuffer>>,
}

impl log::Log for LogCapture {
    fn enabled(&self, metadata: &log::Metadata) -> bool {
        self.inner.enabled(metadata)
    }

    fn log(&self, record: &log::Record) {
        if self.inner.enabled(record.metadata()) {
            // Forward to env_logger (prints to stderr)
            self.inner.log(record);

            // Capture to ring buffer
            let entry = LogEntry {
                level: record.level(),
                target: record.target().to_owned(),
                message: format!("{}", record.args()),
                timestamp: Instant::now(),
            };
            if let Ok(mut buf) = self.buffer.lock() {
                buf.push(entry);
            }
        }
    }

    fn flush(&self) {
        self.inner.flush();
    }
}

/// Install the capturing logger. Must be called once before `App::run()`.
pub fn install() {
    let buffer = Arc::new(Mutex::new(LogBuffer::new(10_000)));
    assert!(
        LOG_BUFFER.set(buffer.clone()).is_ok(),
        "install() called twice"
    );

    let inner =
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).build();
    let max_level = inner.filter();

    let logger = LogCapture { inner, buffer };

    log::set_boxed_logger(Box::new(logger)).expect("logger already set");
    log::set_max_level(max_level);
}
