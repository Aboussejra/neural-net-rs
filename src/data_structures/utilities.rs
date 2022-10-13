use ndarray::{Array, Ix2};

/*
    activations -- activations from previous layer (or input data): (size of previous layer, number of examples)
    weights_matrix -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    bias -- bias vector, numpy array of shape (size of the current layer, 1)

Returns:
out -- the input of the activation function, also called pre-activation parameter
cache -- a tuple containing "activations", "weights_matrix" and "bias" ; stored for computing the backward pass efficiently
 */
pub fn linear_forward(
    activations: Array<f32, Ix2>,
    weights_matrix: Array<f32, Ix2>,
    bias: Array<f32, Ix2>,
) -> (
    Array<f32, Ix2>,
    (Array<f32, Ix2>, Array<f32, Ix2>, Array<f32, Ix2>),
) {
    let out = weights_matrix.dot(&activations) + bias.clone();
    let cache = (activations, weights_matrix, bias);
    (out, cache)
}

pub fn sigmoid(input: Array<f32, Ix2>) -> Array<f32, Ix2> {
    input.mapv(|x| 1. / (1. + (-x).exp()))
}
/*
forward_pass_output -- probability vector corresponding to your label predictions, shape (number of examples, 1)
Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (number of examples, 1)

Returns:
cost -- cross-entropy cost
 */
pub fn compute_cost(
    forward_pass_output: Array<f32, Ix2>,
    true_label_vector: Array<f32, Ix2>,
) -> f32 {
    let num_samples = true_label_vector.len();

    let first_product = true_label_vector.dot(&forward_pass_output.mapv(|x| x.ln()).t());
    let second_product = true_label_vector
        .mapv(|x| 1. - x)
        .dot(&forward_pass_output.mapv(|x| (1. - x).ln()).t());
    let result = -(first_product + second_product);
    let cost = Array::from_iter(result.iter().cloned())[0] / (num_samples as f32);
    cost
}

/*
    Implement the linear portion of backward propagation for a single layer (layer i)

    Arguments:
    grad_output -- Gradient of the cost with respect to the linear output (of layer i)
    cache_forward_prop -- tuple of values (A_previous_layer, W, b) coming from the forward propagation in the current layer

    Returns:
    grad_prev_layer -- Gradient of the cost with respect to the activation (of the previous layer i-1)
    grad_cost_weights -- Gradient of the cost with respect to weights (current layer l)
    grad_cost_bias -- Gradient of the cost with respect to vias (current layer l)
*/
pub fn linear_backward(
    grad_output: Array<f32, Ix2>,
    cache_forward_prop: (Array<f32, Ix2>, Array<f32, Ix2>, Array<f32, Ix2>),
) -> (Array<f32, Ix2>, Array<f32, Ix2>, Array<f32, Ix2>) {
    let (activations_previous_layer, weights_matrix, bias) = cache_forward_prop;
    let m = activations_previous_layer.shape()[0];
    let grad_cost_weights =
        (grad_output.dot(&activations_previous_layer.t())).mapv(|x| x / (m as f32));
    let bias_weights = activations_previous_layer
        .sum_axis(ndarray::Axis(1))
        .mapv(|x| x / (m as f32))
        .into_shape((m, 1))
        .expect("Should work, equivalent to numpy keep_dims");
    let grad_prev_layer = (weights_matrix.t()).dot(&grad_output);
    return (grad_prev_layer, grad_cost_weights, bias_weights);
}
#[allow(non_snake_case)]
mod tests {
    use ndarray::{array, Array, Ix2};
    use ndarray_rand::{rand_distr::Uniform, RandomExt};

    use crate::data_structures::{
        utilities::{compute_cost, linear_forward},
        DeepNeuralNet,
    };

    #[test]
    fn test_forward_propagation() {
        let X = array![
            [-1.02387576, 1.12397796],
            [-1.62328545, 0.64667545],
            [-1.74314104, -0.59664964]
        ];
        let W = array![[0.74505627, 1.97611078, -1.24412333]];
        let b = array![[1.]];
        let (out, _cache) = linear_forward(X, W, b);
        let real_out = array![[-0.8019544, 3.857635]];
        assert_eq!(real_out, out);
    }
    #[test]
    fn test_whole_model_forward_propagation() {
        let X = array![[-1.02387576, 1.12397796]];
        let W1 = array![[0.01788628], [0.0043651]];
        let b1 = array![[0.], [0.]];
        let W2 = array![[0.00096497, -0.01863493]];
        let b2 = array![[0.]];
        let model = DeepNeuralNet {
            weights: vec![W1, W2],
            bias: vec![b1, b2],
        };
        let model_output = model.forward(X).0;
        let test_output: Array<f32, Ix2> = array![[0.5000164, 0.49998199]];
        assert_eq!(model_output, test_output);
    }
    #[test]
    fn test_compute_cost() {
        let X = array![[0.5, 0.5]];
        let Y = array![[0., 1.]];
        assert_eq!(compute_cost(X, Y), 0.6931472)
    }
}
