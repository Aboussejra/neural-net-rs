use ndarray::{Array, Ix2};
use ndarray_rand::{rand_distr::Uniform, RandomExt};

use super::utilities::{linear_forward, sigmoid};

#[derive(Debug)]
pub struct DeepNeuralNet {
    pub weights: Vec<Array<f32, Ix2>>,
    pub bias: Vec<Array<f32, Ix2>>,
}

impl DeepNeuralNet {
    /*
    Inits a deep neural net
    weights[i] -- weight matrix of shape (layer_dims[i], layer_dims[i-1])
    bias[i] -- bias vector of shape (layer_dims[i], 1)
     */
    pub fn init(layer_dims: Vec<usize>) -> DeepNeuralNet {
        let mut weights = vec![];
        let mut bias = vec![];
        let depth = layer_dims.len();
        for i in 1..depth {
            weights.push(Array::<f32, Ix2>::random(
                (layer_dims[i], layer_dims[i - 1]),
                Uniform::new(-1., 1.),
            ));
            bias.push(Array::<f32, Ix2>::zeros((layer_dims[i], 1)));
        }
        DeepNeuralNet { weights, bias }
    }
    pub fn forward(
        &self,
        input_data: Array<f32, Ix2>,
    ) -> (
        ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<[usize; 2]>>,
        Vec<(
            ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<[usize; 2]>>,
            ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<[usize; 2]>>,
            ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<[usize; 2]>>,
        )>,
    ) {
        let depth = self.weights.len();
        let mut caches = vec![];
        let mut current_layer_input = input_data;
        for i in 0..depth {
            let layer_input = current_layer_input.clone();
            let (output, cache) =
                linear_forward(layer_input, self.weights[i].clone(), self.bias[i].clone());
            current_layer_input = output;
            caches.push(cache)
        }
        let forward_pass_output = sigmoid(current_layer_input);
        (forward_pass_output, caches)
    }
    // fn sigmoid_backward(dA, cache) {}
    // """
    // Implement the backward propagation for a single SIGMOID unit.
    // Arguments:
    // dA -- post-activation gradient, of any shape
    // cache -- 'Z' where we store for computing backward propagation efficiently
    // Returns:
    // dZ -- Gradient of the cost with respect to Z
    // """

    // Z = cache

    // s = 1/(1+np.exp(-Z))
    // dZ = dA * s * (1-s)

    // assert (dZ.shape == Z.shape)

    // return dZ
}
