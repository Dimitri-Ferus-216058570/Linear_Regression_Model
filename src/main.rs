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
impl<B:Backend> LinearRegression<B> {
    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        self.linear.forward(x)
    }
}
//main call to execute the program: create the dataset and display the results
fn main() {
    let (x_data, y_data) = generate_data(100);
    let mut points: Vec<(f32, f32)> = x_data.iter().zip(y_data.iter()).map(|(x, y)| (*x, *y)).collect();

    println!("Training data plot");
    Chart::new(100, 30, 0.0, 10.0)
        .lineplot(&Shape::Points(&points))
        .display();
}
