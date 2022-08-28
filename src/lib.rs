mod document;
mod field;
mod memory;

use crate::document::Document;
use crate::memory::mem;

struct IndexMemoryWriter {
    mem_table: mem::MemTable,
}

impl IndexMemoryWriter {
    fn new() -> IndexMemoryWriter {
        Self {
            mem_table: mem::MemTable::default(),
        }
    }

    fn index(doc: &Document) {}

    fn search() {}
}

struct IndexWriter {}

struct IndexReader {}
