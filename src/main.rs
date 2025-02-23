//imports
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use burn::nn::{Linear, LinearConfig};
use burn::nn::loss::{MseLoss, Reduction};
use burn_ndarray::NdArray;
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
    fn compute_loss(&self, x: Tensor<B, 2>, y_true: Tensor<B, 2>) -> Tensor<B, 1> {
        let y_pred = self.forward(x);
        MseLoss::new().forward(y_pred, y_true, Reduction::Mean) // Compute MSE loss
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

    // Convert data into tensors
    let x_data_cloned = x_data.clone();
    let y_data_cloned = y_data.clone();

    let x_tensor = Tensor::<burn_ndarray::NdArray, 2>::from_data(&[x_data.clone()], &Default::default());
    let y_tensor = Tensor::<burn_ndarray::NdArray, 2>::from_data(&[y_data.clone()], &Default::default());

    // Initialize model
    let model = LinearRegression {
        linear: LinearConfig::new(1, 1).init(&Default::default()),
    };

    // Compute loss
    let loss = model.compute_loss(x_tensor, y_tensor);
    println!("Initial Loss: {:?}", loss);

    println!("\nLinear Regression Model Implemented!");
}
