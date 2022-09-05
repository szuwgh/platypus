use super::error::GyResult;
use image::{self, ImageFormat};
use std::io::{BufRead, BufReader, BufWriter, Seek};

pub(crate) struct ModelConfig {
    model_path: String,
}

pub(crate) struct ImageEmbed {}

impl ImageEmbed {
    pub(crate) fn load_model(m: ModelConfig) {}

    pub(crate) fn embed<R: BufRead + Seek>(r: R, f: ImageFormat) -> GyResult<()> {
        let im = image::load(r, f)?;
        Ok(())
    }
}
