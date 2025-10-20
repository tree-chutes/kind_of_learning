//Copyright (c) 2025, tree-chutes

use num_traits::Float;

pub trait Aggregator<F: Float> {
    fn forward(&self, input: F, output: &mut F);
    fn backward(&self, input: F) -> F;
}

pub fn aggregator_functions_factory<F: Float + 'static>(_f: F) -> Box<dyn Aggregator<F>>{
    Box::new(Sum{})
}

pub(in super) struct Sum{
}

impl<F: Float> Aggregator<F> for Sum{
    fn forward(&self, input: F, output: &mut F){
        *output = *output + input;
    }

    fn backward(&self, input: F) -> F{
        input / input
    }
}