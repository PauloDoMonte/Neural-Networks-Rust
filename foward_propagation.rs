// Definição da função de ativação sigmoidal
fn sigmoid(z: f64) -> f64 {
    1.0 / (1.0 + (-z).exp())
}

// Estrutura de um neurônio
struct Neuron {
    bias: f64,
    weights: Vec<f64>,
}

// Estrutura de uma camada de neurônios
struct Layer {
    neurons: Vec<Neuron>,
}

fn main() {
    // Dados de entrada
    let x1 = 0.5;
    let x2 = 0.8;

    // Definição das camadas da rede neural
    let layers = vec![
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
            ],
        },
        Layer {
            neurons: vec![
                Neuron {
                    bias: -0.3,
                    weights: vec![0.2, 0.4],
                },
                Neuron {
                    bias: 0.4,
                    weights: vec![0.6, -0.5],
                },
            ],
        },
    ];

    // Dados de entrada para a primeira camada
    let mut inputs = vec![x1, x2];

    // Iterar sobre cada camada
    for (i, layer) in layers.iter().enumerate() {
        // Armazenar as saídas desta camada
        let mut outputs = vec![];

        // Iterar sobre cada neurônio na camada
        for neuron in &layer.neurons {
            // Calcular o valor de Z para o neurônio
            let z: f64 = neuron.bias + neuron.weights.iter().zip(&inputs).map(|(w, x)| w * x).sum::<f64>();

            // Aplicar a função de ativação sigmoidal
            let a: f64 = sigmoid(z);

            // Adicionar a saída do neurônio aos resultados desta camada
            outputs.push(a);
        }

        // A saída desta camada se torna a entrada para a próxima
        inputs = outputs;

        // Imprimir saída intermediária (opcional)
        println!("Saída da camada {}:", i + 1);
        for (j, output) in outputs.iter().enumerate() {
            println!("Z{}_{}: {:.4}", i + 1, j + 1, output);
        }
    }

    // Resultado final (saída da rede neural)
    println!("Saída final da rede neural:");
    for (i, output) in inputs.iter().enumerate() {
        println!("Z{}: {:.4}", layers.len(), i + 1, output);
    }
}
