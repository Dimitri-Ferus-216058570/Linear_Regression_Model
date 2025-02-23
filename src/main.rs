//imports
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use burn::nn::Linear;//for linear layering
use burn::nn::loss::MseLoss;
use burn::optim::Adam;
use rand::Rng;//generates random numbers
use textplots::{Chart, Plot, Shape};//for plotting data

//generate a dataset x,y pairs where y=2x+1
fn generate_data(samples: usize) -> (Vec<f32>, Vec<f32>) {
    let mut rng = rand::thread_rng();
    let mut x_data = Vec::new();
    let  mut y_data = Vec::new();

    for _ in 0..samples {
        let x: f32 = rng.gen_range(0.0..10.0);
        let noise: f32 = rng.gen_range(-0.5..0.5);
        let y = 2.0 * x + 1.0 + noise;
        x_data.push(x);
        y_data.push(y);
    }
    (x_data, y_data)
}

//defines the model structure of a single linear layer
#[derive(Debug, burn::module::Module)]
struct LinearRegression<B:Backend> {
    linear: Linear<B>,
}
//computes predictions
fn main() {
    println!("Hello, world!");
}
