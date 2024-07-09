extern crate rand;
use rand::Rng;
use std::iter::zip;

// Definição da função de ativação sigmoidal
fn sigmoid(z: f64) -> f64 {
    1.0 / (1.0 + (-z).exp())
}

// Derivada da função de ativação sigmoidal
fn sigmoid_prime(z: f64) -> f64 {
    sigmoid(z) * (1.0 - sigmoid(z))
}

// Estrutura de um neurônio
#[derive(Clone)]
struct Neuron {
    bias: f64,
    weights: Vec<f64>,
}

// Estrutura de uma camada de neurônios
struct Layer {
    neurons: Vec<Neuron>,
}

// Função de custo quadrático
fn cost_derivative(output_activations: &[f64], y: &[f64]) -> Vec<f64> {
    zip(output_activations, y)
        .map(|(output, &target)| output - target)
        .collect()
}

fn main() {
    // Dados de entrada
    let x1 = 0.5;
    let x2 = 0.8;

    // Saída desejada (target)
    let y = vec![0.1];

    // Definição das camadas da rede neural
    let mut layers = vec![
        Layer {
            neurons: vec![
                Neuron {
                    bias: 0.1,
                    weights: vec![0.3, 0.5],
                },
                Neuron {
                    bias: -0.2,
                    weights: vec![-0.4, 0.6],
                },
            ],
        },
        Layer {
            neurons: vec![
                Neuron {
                    bias: 0.2,
                    weights: vec![0.7, -0.3],
                },
                Neuron {
                    bias: 0.5,
                    weights: vec![-0.1, 0.4],
                },
                Neuron {
                    bias: 0.3,
                    weights: vec![-0.2, 0.3],
                },
                Neuron {
                    bias: 0.2,
                    weights: vec![0.7, -0.3],
                },
                Neuron {
                    bias: 0.5,
                    weights: vec![-0.1, 0.4],
                },
                Neuron {
                    bias: 0.3,
                    weights: vec![-0.2, 0.3],
                },
            ],
        },
        Layer {
            neurons: vec![
                Neuron {
                    bias: -0.3,
                    weights: vec![0.2, 0.4],
                },
            ],
        },
    ];

    // Taxa de aprendizado
    let learning_rate = 0.1;
    // Número de épocas
    let epochs = 10000;

    for epoch in 0..epochs {
        // Dados de entrada para a primeira camada
        let mut inputs = vec![x1, x2];

        // Forward pass
        let mut activations = vec![inputs.clone()];
        let mut zs = vec![];

        for layer in &layers {
            let mut z = vec![];
            let mut activation = vec![];

            for neuron in &layer.neurons {
                let z_value: f64 = neuron.bias + neuron.weights.iter().zip(&inputs).map(|(w, x)| w * x).sum::<f64>();
                z.push(z_value);
                activation.push(sigmoid(z_value));
            }

            zs.push(z);
            inputs = activation.clone();
            activations.push(activation);
        }

        // Backward pass
        let mut delta = cost_derivative(&activations.pop().unwrap(), &y);
        let mut nabla_b = vec![vec![0.0; delta.len()]];
        let mut nabla_w = vec![vec![vec![0.0; layers[0].neurons[0].weights.len()]; delta.len()]];

        // Última camada
        for (i, z) in zs.pop().unwrap().iter().enumerate() {
            delta[i] *= sigmoid_prime(*z);
        }

        for (j, delta_j) in delta.iter().enumerate() {
            for (k, activation) in activations.last().unwrap().iter().enumerate() {
                nabla_w[0][j][k] += delta_j * activation;
            }
            nabla_b[0][j] += delta_j;
        }

        // Camadas intermediárias
        for l in (1..layers.len()).rev() {
            let z = zs.pop().unwrap();
            let mut sp = vec![];
            for z_value in &z {
                sp.push(sigmoid_prime(*z_value));
            }
            let layer = &layers[l];
            let mut new_delta = vec![0.0; layer.neurons.len()];

            for (j, neuron) in layer.neurons.iter().enumerate() {
                for (k, weight) in neuron.weights.iter().enumerate() {
                    new_delta[j] += weight * delta[k];
                }
                new_delta[j] *= sp[j];
            }

            delta = new_delta;
            nabla_b.push(vec![0.0; delta.len()]);
            nabla_w.push(vec![vec![0.0; layers[l-1].neurons[0].weights.len()]; delta.len()]);

            for (j, delta_j) in delta.iter().enumerate() {
                for (k, activation) in activations[l-1].iter().enumerate() {
                    nabla_w[l][j][k] += delta_j * activation;
                }
                nabla_b[l][j] += delta_j;
            }
        }

        // Atualizar pesos e bias
        for (l, layer) in layers.iter_mut().enumerate() {
            for (j, neuron) in layer.neurons.iter_mut().enumerate() {
                neuron.bias -= learning_rate * nabla_b[l][j];
                for k in 0..neuron.weights.len() {
                    neuron.weights[k] -= learning_rate * nabla_w[l][j][k];
                }
            }
        }

        // Opcional: Imprimir o erro atual ou o progresso a cada N épocas
        if epoch % 1000 == 0 {
            let cost = cost_derivative(&inputs, &y).iter().map(|&x| x.powi(2)).sum::<f64>() / 2.0;
            println!("Epoch {}: Erro = {:.6}", epoch, cost);
        }
    }

    // Dados de entrada para a última camada (novamente)
    let mut inputs = vec![x1, x2];

    // Forward pass final
    for layer in &layers {
        let mut activation = vec![];

        for neuron in &layer.neurons {
            let z_value: f64 = neuron.bias + neuron.weights.iter().zip(&inputs).map(|(w, x)| w * x).sum::<f64>();
            activation.push(sigmoid(z_value));
        }

        inputs = activation.clone();
    }

    // Resultado final (saída da rede neural)
    println!("Saída final da rede neural:");
    for (i, output) in inputs.iter().enumerate() {
        println!("Z{}_{}: {:.4}", layers.len(), i + 1, output);
    }
}
