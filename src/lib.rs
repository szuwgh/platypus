mod document;
mod field;
mod memory;

use crate::document::Document;
use crate::memory::mem;

struct Engine {
    mem_table: mem::MemTable,
}

impl Engine {
    fn index(doc: &Document) {}

    fn search() {}
}
