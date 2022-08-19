//域接口定义
pub(crate) struct Field {
    id: u32,
    value: Value,
}

impl Field {
    fn new() {}
}

pub(crate) enum Value {
    Str(String),
}
