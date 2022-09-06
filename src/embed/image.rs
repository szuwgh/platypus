use crate::util::error::GyResult;
use image::{self, ImageFormat};
use std::io::{BufRead, BufReader, BufWriter, Seek};
use tract_onnx::prelude::*;

type TractSimplePlan = SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

struct ImageSize {
    pub width: usize,
    pub height: usize,
}

pub(crate) struct ModelConfig {
    model_path: String,
    image_size: ImageSize,
    layer_name: Option<String>,
}

pub(crate) struct DefaultImageEmbed {
    model: TractSimplePlan,
}

impl DefaultImageEmbed {
    pub(crate) fn new(m: ModelConfig) -> Self {
        let model = Self::load_model(m);
        Self { model }
    }

    fn load_model(m: ModelConfig) -> TractSimplePlan {
        let mut model = tract_onnx::onnx()
            .model_for_path(m.model_path.clone())
            .expect("not found file")
            .with_input_fact(
                0,
                InferenceFact::dt_shape(
                    f32::datum_type(),
                    tvec!(1, 3, m.image_size.width, m.image_size.height),
                ),
            )
            .unwrap();
        if let Some(layer_name) = m.layer_name.clone() {
            model = model.with_output_names(vec![layer_name]).unwrap()
        }
        model.into_optimized().unwrap().into_runnable().unwrap()
    }

    pub(crate) fn embed<R: BufRead + Seek>(r: R, image_ext: &str) -> GyResult<()> {
        let image_format = ImageFormat::from_extension(image_ext).ok_or("not surrport")?;
        let im = image::load(r, image_format)?;

        Ok(())
    }
}
