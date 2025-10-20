// Copyright (c) 2025, tree-chutes

pub mod aggregator_functions;
mod conv2d;
pub mod layers;
mod linear_layer;
pub mod loss_functions;
mod mean_squares;
pub mod register;

#[cfg(test)]
mod tests {

    use super::{
        layers::{Layers, layer_factory},
        loss_functions::{LossFunctions, loss_function_factory},
    };

    #[test]
    fn test_18_x_18_f64() {
        let mut x: Vec<Vec<f64>> = vec![vec![
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7,
            1.8,
        ]];
        let w: Vec<Vec<f64>> = vec![
            vec![0.1],
            vec![0.2],
            vec![0.3],
            vec![0.4],
            vec![0.5],
            vec![0.6],
            vec![0.7],
            vec![0.8],
            vec![0.9],
            vec![0.1],
            vec![0.2],
            vec![0.3],
            vec![0.4],
            vec![0.5],
            vec![0.6],
            vec![0.7],
            vec![0.8],
            vec![0.9],
        ];
        let y = vec![vec![2.0]];
        let l = layer_factory::<f64>(
            Layers::Linear,
            1, //configuration value. Vector already flattened from previous layer
            x[0].len(),
            1, //configuration value. Vector already flattened from previous layer
            None,
            0.0,
        );
        let b = vec![];

        let LINEAR_OUTPUT = [9.75];
        let LOSS = [60.0625];
        let GRADIENTS_FROM_LINEAR_TO_PREVIOUS = [
            1.55,
            3.1,
            4.6499999999999995,
            6.2,
            7.75,
            9.299999999999999,
            10.85,
            12.4,
            13.950000000000001,
            1.55,
            3.1,
            4.6499999999999995,
            6.2,
            7.75,
            9.299999999999999,
            10.85,
            12.4,
            13.950000000000001,
        ];
        let LINEAR_UPDATED_WEIGHTS = [
            0.0845,
            0.169,
            0.2535,
            0.338,
            0.4225,
            0.507,
            0.5914999999999999,
            0.676,
            0.7605,
            -0.055,
            0.029500000000000002,
            0.114,
            0.1985,
            0.28300000000000003,
            0.3675,
            0.45199999999999996,
            0.5365000000000001,
            0.621,
        ];

        let (mut flat_x, mut flat_w, mut flat_b) = l.flatten(x, w, b);
        let forward_linear = (flat_x.as_slice(), flat_w.as_mut_slice(), flat_b.as_slice());
        let mut z_linear: Vec<f64> = l.forward(forward_linear, None);
        assert!(
            LINEAR_OUTPUT[0] - z_linear[0] < f64::EPSILON,
            "LINEAR_OUTPUT truth {} prediction {}",
            LINEAR_OUTPUT[0],
            z_linear[0]
        );
        let (flat_y, squared) = loss_function_factory(LossFunctions::MeanSquares, y, 1.0);
        let loss = squared.forward(&flat_y, &z_linear);
        assert!(
            LOSS[0] - loss[0] < f64::EPSILON,
            "LOSS truth {} prediction {}",
            LOSS[0],
            loss[0]
        );
        let mut from_loss_to_linear_grads = squared.backward(z_linear[0], &mut flat_x);
        let l_backward = (
            from_loss_to_linear_grads.as_mut_slice(),
            flat_w.as_mut_slice(),
            flat_x.as_mut_slice(),
        );

        let (from_linear_to_previous_grads, _dummy_bias) =
            l.backward(l_backward, 0.01, z_linear[0]);

        for i in 0..GRADIENTS_FROM_LINEAR_TO_PREVIOUS.len() {
            assert!(
                (GRADIENTS_FROM_LINEAR_TO_PREVIOUS[i] - from_linear_to_previous_grads[i]).abs()
                    < f64::EPSILON,
                "GRADIENTS_FROM_LINEAR_TO_PREVIOUS {} truth {} prediction {}",
                i,
                GRADIENTS_FROM_LINEAR_TO_PREVIOUS[i],
                from_linear_to_previous_grads[i]
            );
        }
        for i in 0..LINEAR_UPDATED_WEIGHTS.len() {
            assert!(
                (LINEAR_UPDATED_WEIGHTS[i] - flat_w[i]).abs() < f64::EPSILON,
                "LINEAR_UPDATED_WEIGHTS {} truth {} prediction {}",
                i,
                LINEAR_UPDATED_WEIGHTS[i],
                flat_w[i]
            );
        }
    }

    #[test]
    fn test_18_x_18_f32() {
        let mut x: Vec<Vec<f32>> = vec![vec![
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7,
            1.8,
        ]];
        let w: Vec<Vec<f32>> = vec![
            vec![0.1],
            vec![0.2],
            vec![0.3],
            vec![0.4],
            vec![0.5],
            vec![0.6],
            vec![0.7],
            vec![0.8],
            vec![0.9],
            vec![0.1],
            vec![0.2],
            vec![0.3],
            vec![0.4],
            vec![0.5],
            vec![0.6],
            vec![0.7],
            vec![0.8],
            vec![0.9],
        ];
        let y: Vec<Vec<f32>> = vec![vec![2.0]];
        let l = layer_factory::<f32>(
            Layers::Linear,
            1, //configuration value. Vector already flattened from previous layer
            x[0].len(),
            1, //configuration value. Vector already flattened from previous layer
            None,
            0.0,
        );
        let b: Vec<Vec<f32>> = vec![];

        let LINEAR_OUTPUT: [f32; 1] = [9.75];
        let LOSS: [f32; 1] = [60.0625];
        let GRADIENTS_FROM_LINEAR_TO_PREVIOUS: [f32; 18] = [
            1.5500001, 3.1000001, 4.65, 6.2000003, 7.75, 9.3, 10.849999, 12.400001, 13.95,
            1.5500001, 3.1000001, 4.65, 6.2000003, 7.75, 9.3, 10.849999, 12.400001, 13.95,
        ];
        let LINEAR_UPDATED_WEIGHTS: [f32; 18] = [
            0.0845,
            0.169,
            0.2535,
            0.338,
            0.4225,
            0.507,
            0.5915,
            0.676,
            0.76049995,
            -0.054999996,
            0.029499995,
            0.114000015,
            0.1985,
            0.28300002,
            0.36750004,
            0.452,
            0.53650004,
            0.621,
        ];

        let (mut flat_x, mut flat_w, mut flat_b) = l.flatten(x, w, b);
        let forward_linear = (flat_x.as_slice(), flat_w.as_mut_slice(), flat_b.as_slice());
        let mut z_linear: Vec<f32> = l.forward(forward_linear, None);
        assert!(
            LINEAR_OUTPUT[0] - z_linear[0] < f32::EPSILON,
            "LINEAR_OUTPUT truth {} prediction {}",
            LINEAR_OUTPUT[0],
            z_linear[0]
        );
        let (flat_y, squared) = loss_function_factory(LossFunctions::MeanSquares, y, 1.0);
        let loss = squared.forward(&flat_y, &z_linear);
        assert!(
            LOSS[0] - loss[0] < f32::EPSILON,
            "LOSS truth {} prediction {}",
            LOSS[0],
            loss[0]
        );
        let mut from_loss_to_linear_grads = squared.backward(z_linear[0], &mut flat_x);
        let l_backward = (
            from_loss_to_linear_grads.as_mut_slice(),
            flat_w.as_mut_slice(),
            flat_x.as_mut_slice(),
        );

        let (from_linear_to_previous_grads, _dummy_bias) =
            l.backward(l_backward, 0.01, z_linear[0]);

        for i in 0..GRADIENTS_FROM_LINEAR_TO_PREVIOUS.len() {
            assert!(
                (GRADIENTS_FROM_LINEAR_TO_PREVIOUS[i] - from_linear_to_previous_grads[i]).abs()
                    < f32::EPSILON,
                "GRADIENTS_FROM_LINEAR_TO_PREVIOUS {} truth {} prediction {}",
                i,
                GRADIENTS_FROM_LINEAR_TO_PREVIOUS[i],
                from_linear_to_previous_grads[i]
            );
        }
        for i in 0..LINEAR_UPDATED_WEIGHTS.len() {
            assert!(
                (LINEAR_UPDATED_WEIGHTS[i] - flat_w[i]).abs() < f32::EPSILON,
                "LINEAR_UPDATED_WEIGHTS {} truth {} prediction {}",
                i,
                LINEAR_UPDATED_WEIGHTS[i],
                flat_w[i]
            );
        }
    }

    #[test]
    fn test_conv2d_conv2d_linear_7_5_3_64() {
        let mut input_layer = vec![
            vec![1.0, 0.5, 1.2, 0.8, 1.5, 0.9, 1.3, 0.7, 1.1],
            vec![0.6, 1.4, 0.8, 1.7, 1.0, 1.6, 0.9, 1.2, 0.5],
            vec![1.3, 0.7, 1.8, 1.1, 1.4, 0.6, 1.9, 1.0, 1.5],
            vec![0.9, 1.2, 0.8, 1.6, 1.3, 1.1, 0.7, 1.4, 0.9],
            vec![1.1, 0.8, 1.5, 1.0, 1.7, 1.2, 1.4, 0.6, 1.3],
            vec![0.7, 1.3, 0.9, 1.4, 1.1, 1.8, 1.0, 1.5, 0.8],
            vec![1.2, 0.6, 1.4, 0.9, 1.3, 1.0, 1.6, 1.1, 1.7],
            vec![0.8, 1.1, 0.7, 1.5, 1.2, 1.4, 0.9, 1.3, 1.0],
            vec![1.0, 1.5, 0.8, 1.2, 0.9, 1.3, 1.1, 0.7, 1.4],
        ];

        let conv_0_weights = vec![
            vec![0.1, 0.2, 0.3, 0.2, 0.1],
            vec![0.2, 0.4, 0.6, 0.4, 0.2],
            vec![0.3, 0.6, 0.9, 0.6, 0.3],
            vec![0.2, 0.4, 0.6, 0.4, 0.2],
            vec![0.1, 0.2, 0.3, 0.2, 0.1],
        ];

        let conv_1_weights = vec![
            vec![1.0, 0.5, 0.2],
            vec![0.5, 1.0, 0.5],
            vec![0.2, 0.5, 1.0],
        ];

        let linear_weights: Vec<Vec<f64>> = vec![
            vec![0.1],
            vec![0.2],
            vec![0.3],
            vec![0.4],
            vec![0.5],
            vec![0.6],
            vec![0.7],
            vec![0.8],
            vec![0.9],
        ];

        //Pytorch matches
        let CONV_0_OUTPUT = [
            9.55,
            9.96,
            10.090000000000002,
            9.63,
            9.41,
            9.59,
            10.169999999999998,
            10.270000000000001,
            9.870000000000001,
            9.44,
            9.430000000000001,
            10.01,
            10.4,
            10.110000000000001,
            9.780000000000001,
            9.04,
            9.82,
            10.28,
            10.34,
            9.97,
            8.81,
            9.450000000000001,
            9.930000000000003,
            10.100000000000001,
            10.000000000000002,
        ];

        let CONV_1_OUTPUT = [
            53.93899999999999,
            54.533,
            53.42700000000001,
            53.651999999999994,
            55.18300000000001,
            54.489000000000004,
            52.412000000000006,
            54.547000000000004,
            54.912000000000006,
        ];

        let LINEAR_OUTPUT = [243.82110000000003];

        let LOSS = [58477.444405210015];

        let LINEAR_WEIGHTS_GRADIENTS = [
            26087.1766258,
            26374.460092600002,
            25839.551819400007,
            25948.3713144,
            26688.827522600008,
            26353.179835800005,
            25348.654986400004,
            26381.231083400005,
            26557.760486400006,
        ];

        let LINEAR_UPDATED_WEIGHTS = [
            -260.771766258,
            -263.54460092600004,
            -258.0955181940001,
            -259.083713144,
            -266.3882752260001,
            -262.93179835800004,
            -252.78654986400005,
            -263.01231083400006,
            -264.67760486400005,
        ];

        let GRADIENTS_FROM_LINEAR_TO_CONV_1 = [
            48.36422000000001,
            96.72844000000002,
            145.09266000000002,
            193.45688000000004,
            241.82110000000003,
            290.18532000000005,
            338.54954000000004,
            386.9137600000001,
            435.27798000000007,
        ];

        let CONV_1_UPDATED_WEIGHTS = [
            -216.76473697200004,
            -219.33472558800005,
            -215.67369597000004,
            -214.85136239400003,
            -220.053503932,
            -218.86559265400007,
            -209.81195250600007,
            -217.30826477000005,
            -218.04638880200008,
        ];

        let GRADIENTS_FROM_CONV_1_TO_CONV_0 = [
            48.36422000000001,
            120.91055000000003,
            203.12972400000004,
            91.89201800000002,
            29.018532000000008,
            217.63899000000004,
            435.27798000000007,
            643.244126,
            386.9137600000001,
            130.58339400000003,
            444.95082400000007,
            914.0837580000002,
            1305.8339400000004,
            875.3923820000002,
            377.2409160000001,
            207.96614600000004,
            677.0990800000001,
            1146.2320140000002,
            1015.6486200000002,
            507.8243100000001,
            67.70990800000001,
            246.65752200000003,
            619.0620160000001,
            604.5527500000001,
            435.27798000000007,
        ];

        let CONV_0_UPDATED_WEIGHTS = [
            -143.09194615400003,
            -147.60589274200004,
            -146.68370100200005,
            -136.883545168,
            -135.508436458,
            -139.321101856,
            -150.684986858,
            -150.33022135400003,
            -143.82694046200004,
            -133.96234628000002,
            -135.86462498800003,
            -145.00048431000002,
            -152.83050969200005,
            -149.04373310200003,
            -144.53632963400003,
            -130.15608216600003,
            -141.902044506,
            -147.46705953000006,
            -150.64145906000002,
            -146.89977513000005,
            -126.53687364800004,
            -135.708294622,
            -142.09393652400001,
            -142.39222982600003,
            -142.84045221000002,
        ];
        //Pytorch matches END

        let bias: Vec<Vec<f64>> = vec![];
        let p_s = (0 as u16, 1 as u16);
        let mut conv_layer_0 = layer_factory::<f64>(
            Layers::Conv2D,
            input_layer.len(),
            conv_0_weights.len(),
            input_layer[0].len(),
            Some(p_s),
            0.0,
        );

        conv_layer_0.set_first_layer_flag();

        let (mut flat_input_layer, mut flat_conv_0_kernel, mut flat_conv_0_bias) =
            conv_layer_0.flatten(input_layer, conv_0_weights, bias);
        let forward_conv_0 = (
            flat_input_layer.as_slice(),
            flat_conv_0_kernel.as_mut_slice(),
            flat_conv_0_bias.as_slice(),
        );

        let mut z_conv_0: Vec<f64> = conv_layer_0.forward(forward_conv_0, None);
        for i in 0..CONV_0_OUTPUT.len() {
            assert!(
                (CONV_0_OUTPUT[i] - z_conv_0[i]).abs() < f64::EPSILON,
                "CONV_0_OUTPUT {} truth {} prediction {}",
                i,
                CONV_0_OUTPUT[i],
                z_conv_0[i]
            );
        }

        let conv_layer_1 = layer_factory::<f64>(
            Layers::Conv2D,
            5, //configuration value
            conv_1_weights.len(),
            5, //configuration value
            Some(p_s),
            0.0,
        );
        let mut flat_conv_1_kernel = conv_layer_1.flatten_kernel(conv_1_weights);
        let forward_conv_1 = (
            z_conv_0.as_slice(),
            flat_conv_1_kernel.as_mut_slice(),
            flat_conv_0_bias.as_slice(),
        );

        let mut z_conv_1: Vec<f64> = conv_layer_1.forward(forward_conv_1, None);
        for i in 0..CONV_1_OUTPUT.len() {
            assert!(
                (CONV_1_OUTPUT[i] - z_conv_1[i]).abs() < f64::EPSILON,
                "CONV_1_OUTPUT {} truth {} prediction {}",
                i,
                CONV_1_OUTPUT[i],
                z_conv_1[i]
            );
        }

        let linear_layer = layer_factory::<f64>(
            Layers::Linear,
            1, //configuration value. Vector already flattened from previous layer
            linear_weights.len(),
            1, //configuration value. Vector already flattened from previous layer
            Some(p_s),
            0.0,
        );

        let mut flat_linear_weights = linear_layer.flatten_kernel(linear_weights);

        let forward_linear = (
            z_conv_1.as_slice(),
            flat_linear_weights.as_mut_slice(),
            flat_conv_0_bias.as_slice(),
        );

        let mut z_linear: Vec<f64> = linear_layer.forward(forward_linear, None);
        assert!(
            LINEAR_OUTPUT[0] - z_linear[0] < f64::EPSILON,
            "LINEAR_OUTPUT truth {} prediction {}",
            LINEAR_OUTPUT[0],
            z_linear[0]
        );

        let (flat_loss, squared) =
            loss_function_factory(LossFunctions::MeanSquares, vec![vec![2.0]], 1.0);
        let mut loss = squared.forward(&flat_loss, &z_linear);
        assert!(
            LOSS[0] - loss[0] < f64::EPSILON,
            "LOSS truth {} prediction {}",
            LOSS[0],
            loss[0]
        );

        //BACKPASS
        let mut from_loss_to_linear_grads = squared.backward(z_linear[0], z_conv_1.as_mut_slice());

        for i in 0..LINEAR_WEIGHTS_GRADIENTS.len() {
            assert!(
                (LINEAR_WEIGHTS_GRADIENTS[i] - from_loss_to_linear_grads[i]).abs() < f64::EPSILON,
                "LINEAR_WEIGHTS_GRADIENTS {} truth {} prediction {}",
                i,
                LINEAR_WEIGHTS_GRADIENTS[i],
                from_loss_to_linear_grads[i]
            );
        }

        let linear_backward = (
            from_loss_to_linear_grads.as_mut_slice(),
            flat_linear_weights.as_mut_slice(),
            z_conv_1.as_mut_slice(),
        );

        let (mut from_linear_to_conv_1_grads, dummy_bias) =
            linear_layer.backward(linear_backward, 0.01, z_linear[0]);

        for i in 0..LINEAR_UPDATED_WEIGHTS.len() {
            assert!(
                (LINEAR_UPDATED_WEIGHTS[i] - flat_linear_weights[i]).abs() < f64::EPSILON,
                "LINEAR_UPDATED_WEIGHTS {} truth {} prediction {}",
                i,
                LINEAR_UPDATED_WEIGHTS[i],
                flat_linear_weights[i]
            );
        }

        for i in 0..GRADIENTS_FROM_LINEAR_TO_CONV_1.len() {
            assert!(
                (GRADIENTS_FROM_LINEAR_TO_CONV_1[i] - from_linear_to_conv_1_grads[i]).abs()
                    < f64::EPSILON,
                "GRADIENTS_FROM_LINEAR_TO_CONV_1 {} truth {} prediction {}",
                i,
                GRADIENTS_FROM_LINEAR_TO_CONV_1[i],
                from_linear_to_conv_1_grads[i]
            );
        }

        let conv_1_backward = (
            z_conv_0.as_mut_slice(),
            from_linear_to_conv_1_grads.as_mut_slice(),
            flat_conv_1_kernel.as_mut_slice(),
        );

        let (flat_conv_1_kernel, mut from_conv_1_to_conv_0_grads) =
            conv_layer_1.backward(conv_1_backward, 0.01, 0.0);

        for i in 0..CONV_1_UPDATED_WEIGHTS.len() {
            assert!(
                (CONV_1_UPDATED_WEIGHTS[i] - flat_conv_1_kernel[i]).abs() < f64::EPSILON,
                "CONV_1_UPDATED_WEIGHTS {} truth {} prediction {}",
                i,
                CONV_1_UPDATED_WEIGHTS[i],
                flat_conv_1_kernel[i]
            );
        }

        for i in 0..GRADIENTS_FROM_CONV_1_TO_CONV_0.len() {
            assert!(
                (GRADIENTS_FROM_CONV_1_TO_CONV_0[i] - from_conv_1_to_conv_0_grads[i]).abs()
                    < f64::EPSILON,
                "GRADIENTS_FROM_CONV_1_TO_CONV_0 {} truth {} prediction {}",
                i,
                GRADIENTS_FROM_CONV_1_TO_CONV_0[i],
                from_conv_1_to_conv_0_grads[i]
            );
        }

        let conv_0_backward = (
            flat_input_layer.as_mut_slice(),
            from_conv_1_to_conv_0_grads.as_mut_slice(),
            flat_conv_0_kernel.as_mut_slice(),
        );

        let (flat_conv_0_kernel, _not_needed) = conv_layer_0.backward(conv_0_backward, 0.01, 0.0);

        for i in 0..CONV_0_UPDATED_WEIGHTS.len() {
            assert!(
                (CONV_0_UPDATED_WEIGHTS[i] - flat_conv_0_kernel[i]).abs() < f64::EPSILON,
                "GRADIENTS_FROM_CONV_1_TO_CONV_0 {} truth {} prediction {}",
                i,
                CONV_0_UPDATED_WEIGHTS[i],
                flat_conv_0_kernel[i]
            );
        }
    }

    #[test]
    fn test_conv2d_conv2d_linear_7_5_3_32() {
        let mut input_layer: Vec<Vec<f32>> = vec![
            vec![1.0, 0.5, 1.2, 0.8, 1.5, 0.9, 1.3, 0.7, 1.1],
            vec![0.6, 1.4, 0.8, 1.7, 1.0, 1.6, 0.9, 1.2, 0.5],
            vec![1.3, 0.7, 1.8, 1.1, 1.4, 0.6, 1.9, 1.0, 1.5],
            vec![0.9, 1.2, 0.8, 1.6, 1.3, 1.1, 0.7, 1.4, 0.9],
            vec![1.1, 0.8, 1.5, 1.0, 1.7, 1.2, 1.4, 0.6, 1.3],
            vec![0.7, 1.3, 0.9, 1.4, 1.1, 1.8, 1.0, 1.5, 0.8],
            vec![1.2, 0.6, 1.4, 0.9, 1.3, 1.0, 1.6, 1.1, 1.7],
            vec![0.8, 1.1, 0.7, 1.5, 1.2, 1.4, 0.9, 1.3, 1.0],
            vec![1.0, 1.5, 0.8, 1.2, 0.9, 1.3, 1.1, 0.7, 1.4],
        ];

        let conv_0_weights: Vec<Vec<f32>> = vec![
            vec![0.1, 0.2, 0.3, 0.2, 0.1],
            vec![0.2, 0.4, 0.6, 0.4, 0.2],
            vec![0.3, 0.6, 0.9, 0.6, 0.3],
            vec![0.2, 0.4, 0.6, 0.4, 0.2],
            vec![0.1, 0.2, 0.3, 0.2, 0.1],
        ];

        let conv_1_weights: Vec<Vec<f32>> = vec![
            vec![1.0, 0.5, 0.2],
            vec![0.5, 1.0, 0.5],
            vec![0.2, 0.5, 1.0],
        ];

        let linear_weights: Vec<Vec<f32>> = vec![
            vec![0.1],
            vec![0.2],
            vec![0.3],
            vec![0.4],
            vec![0.5],
            vec![0.6],
            vec![0.7],
            vec![0.8],
            vec![0.9],
        ];

        let linear_bias: Vec<Vec<f32>> = vec![vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]];

        //Pytorch matches
        let CONV_0_OUTPUT: [f32; 25] = [
            9.55, 9.960001, 10.09, 9.63, 9.41, 9.59, 10.17, 10.27, 9.87, 9.44, 9.430001, 10.01,
            10.4, 10.11, 9.78, 9.040001, 9.82, 10.280001, 10.339999, 9.97, 8.81, 9.450001, 9.93,
            10.1, 10.000001,
        ];

        let CONV_1_OUTPUT: [f32; 9] = [
            53.939003, 54.533, 53.427002, 53.652, 55.183, 54.489, 52.412003, 54.547005, 54.912003,
        ];

        let LINEAR_OUTPUT: [f32; 1] = [243.821121];

        let LOSS: [f32; 1] = [58477.4531];

        let LINEAR_WEIGHTS_GRADIENTS: [f32; 9] = [
            26087.18, 26374.463, 25839.555, 25948.373, 26688.83, 26353.182, 25348.658, 26381.236,
            26557.764,
        ];

        let LINEAR_UPDATED_WEIGHTS: [f32; 9] = [
            -260.7718, -263.54462, -258.09555, -259.0837, -266.3883, -262.93182, -252.78658,
            -263.01236, -264.67764,
        ];

        let GRADIENTS_FROM_LINEAR_TO_CONV_1: [f32; 9] = [
            48.364223, 96.72845, 145.09268, 193.4569, 241.82112, 290.18536, 338.54956, 386.9138,
            435.278,
        ];

        let CONV_1_UPDATED_WEIGHTS: [f32; 9] = [
            -216.76476, -219.33475, -215.6737, -214.85138, -220.05351, -218.86562, -209.81198,
            -217.30827, -218.0464,
        ];

        let GRADIENTS_FROM_CONV_1_TO_CONV_0: [f32; 25] = [
            48.364223, 120.91056, 203.12976, 91.89203, 29.018538, 217.639, 435.278, 643.2442,
            386.91382, 130.58342, 444.95087, 914.08386, 1305.8341, 875.39246, 377.24097, 207.96616,
            677.0991, 1146.2322, 1015.64874, 507.82437, 67.709915, 246.65753, 619.0621, 604.5528,
            435.278,
        ];

        let CONV_0_UPDATED_WEIGHTS: [f32; 25] = [
            -143.09196, -147.6059, -146.6837, -136.88354, -135.50844, -139.3211, -150.68501,
            -150.33023, -143.82695, -133.96236, -135.86462, -145.00049, -152.83052, -149.04375,
            -144.53635, -130.1561, -141.90205, -147.46707, -150.64146, -146.89978, -126.53689,
            -135.7083, -142.09395, -142.39224, -142.84047,
        ];
        //Pytorch matches END

        let bias: Vec<Vec<f32>> = vec![];
        let p_s = (0 as u16, 1 as u16);
        let mut conv_layer_0 = layer_factory::<f32>(
            Layers::Conv2D,
            input_layer.len(),
            conv_0_weights.len(),
            input_layer[0].len(),
            Some(p_s),
            0.0,
        );

        conv_layer_0.set_first_layer_flag();

        let (mut flat_input_layer, mut flat_conv_0_kernel, mut flat_conv_0_bias) =
            conv_layer_0.flatten(input_layer, conv_0_weights, bias);
        let forward_conv_0 = (
            flat_input_layer.as_slice(),
            flat_conv_0_kernel.as_mut_slice(),
            flat_conv_0_bias.as_slice(),
        );

        let mut z_conv_0: Vec<f32> = conv_layer_0.forward(forward_conv_0, None);
        for i in 0..CONV_0_OUTPUT.len() {
            assert!(
                (CONV_0_OUTPUT[i] - z_conv_0[i]).abs() < f32::EPSILON,
                "CONV_0_OUTPUT {} truth {} prediction {}",
                i,
                CONV_0_OUTPUT[i],
                z_conv_0[i]
            );
        }

        let conv_layer_1 = layer_factory::<f32>(
            Layers::Conv2D,
            5, //configuration value
            conv_1_weights.len(),
            5, //configuration value
            Some(p_s),
            0.0,
        );
        let mut flat_conv_1_kernel = conv_layer_1.flatten_kernel(conv_1_weights);
        let forward_conv_1 = (
            z_conv_0.as_slice(),
            flat_conv_1_kernel.as_mut_slice(),
            flat_conv_0_bias.as_slice(),
        );

        let mut z_conv_1: Vec<f32> = conv_layer_1.forward(forward_conv_1, None);

        for i in 0..CONV_1_OUTPUT.len() {
            assert!(
                (CONV_1_OUTPUT[i] - z_conv_1[i]).abs() < f32::EPSILON,
                "CONV_1_OUTPUT {} truth {} prediction {}",
                i,
                CONV_1_OUTPUT[i],
                z_conv_1[i]
            );
        }

        let linear_layer = layer_factory::<f32>(
            Layers::Linear,
            1, //configuration value. Vector already flattened from previous layer
            linear_weights.len(),
            1, //configuration value. Vector already flattened from previous layer
            Some(p_s),
            0.0,
        );

        let mut flat_linear_weights = linear_layer.flatten_kernel(linear_weights);

        let forward_linear = (
            z_conv_1.as_slice(),
            flat_linear_weights.as_mut_slice(),
            flat_conv_0_bias.as_slice(),
        );

        let mut z_linear: Vec<f32> = linear_layer.forward(forward_linear, None);

        assert!(
            LINEAR_OUTPUT[0] - z_linear[0] < f32::EPSILON,
            "LINEAR_OUTPUT truth {} prediction {}",
            LINEAR_OUTPUT[0],
            z_linear[0]
        );

        let (flat_loss, squared) =
            loss_function_factory(LossFunctions::MeanSquares, vec![vec![2.0]], 1.0);
        let mut loss = squared.forward(&flat_loss, &z_linear);
        assert!(
            LOSS[0] - loss[0] < f32::EPSILON,
            "LOSS truth {} prediction {}",
            LOSS[0],
            loss[0]
        );

        //BACKPASS
        let mut from_loss_to_linear_grads = squared.backward(z_linear[0], z_conv_1.as_mut_slice());

        for i in 0..LINEAR_WEIGHTS_GRADIENTS.len() {
            assert!(
                (LINEAR_WEIGHTS_GRADIENTS[i] - from_loss_to_linear_grads[i]).abs() < f32::EPSILON,
                "LINEAR_WEIGHTS_GRADIENTS {} truth {} prediction {}",
                i,
                LINEAR_WEIGHTS_GRADIENTS[i],
                from_loss_to_linear_grads[i]
            );
        }

        let linear_backward = (
            from_loss_to_linear_grads.as_mut_slice(),
            flat_linear_weights.as_mut_slice(),
            z_conv_1.as_mut_slice(),
        );

        let (mut from_linear_to_conv_1_grads, dummy_bias) =
            linear_layer.backward(linear_backward, 0.01, z_linear[0]);
        for i in 0..LINEAR_UPDATED_WEIGHTS.len() {
            assert!(
                (LINEAR_UPDATED_WEIGHTS[i] - flat_linear_weights[i]).abs() < f32::EPSILON,
                "LINEAR_UPDATED_WEIGHTS {} truth {} prediction {}",
                i,
                LINEAR_UPDATED_WEIGHTS[i],
                flat_linear_weights[i]
            );
        }

        for i in 0..GRADIENTS_FROM_LINEAR_TO_CONV_1.len() {
            assert!(
                (GRADIENTS_FROM_LINEAR_TO_CONV_1[i] - from_linear_to_conv_1_grads[i]).abs()
                    < f32::EPSILON,
                "LINEAR_UPDATED_WEIGHTS {} truth {} prediction {}",
                i,
                GRADIENTS_FROM_LINEAR_TO_CONV_1[i],
                from_linear_to_conv_1_grads[i]
            );
        }

        let conv_1_backward = (
            z_conv_0.as_mut_slice(),
            from_linear_to_conv_1_grads.as_mut_slice(),
            flat_conv_1_kernel.as_mut_slice(),
        );

        let (flat_conv_1_kernel, mut from_conv_1_to_conv_0_grads) =
            conv_layer_1.backward(conv_1_backward, 0.01, 0.0);

        for i in 0..CONV_1_UPDATED_WEIGHTS.len() {
            assert!(
                (CONV_1_UPDATED_WEIGHTS[i] - flat_conv_1_kernel[i]).abs() < f32::EPSILON,
                "CONV_1_UPDATED_WEIGHTS {} truth {} prediction {}",
                i,
                CONV_1_UPDATED_WEIGHTS[i],
                flat_conv_1_kernel[i]
            );
        }

        for i in 0..GRADIENTS_FROM_CONV_1_TO_CONV_0.len() {
            assert!(
                (GRADIENTS_FROM_CONV_1_TO_CONV_0[i] - from_conv_1_to_conv_0_grads[i]).abs()
                    < f32::EPSILON,
                "GRADIENTS_FROM_CONV_1_TO_CONV_0 {} truth {} prediction {}",
                i,
                GRADIENTS_FROM_CONV_1_TO_CONV_0[i],
                from_conv_1_to_conv_0_grads[i]
            );
        }

        let conv_0_backward = (
            flat_input_layer.as_mut_slice(),
            from_conv_1_to_conv_0_grads.as_mut_slice(),
            flat_conv_0_kernel.as_mut_slice(),
        );

        let (flat_conv_0_kernel, _not_needed) = conv_layer_0.backward(conv_0_backward, 0.01, 0.0);

        for i in 0..CONV_0_UPDATED_WEIGHTS.len() {
            assert!(
                (CONV_0_UPDATED_WEIGHTS[i] - flat_conv_0_kernel[i]).abs() < f32::EPSILON,
                "GRADIENTS_FROM_CONV_1_TO_CONV_0 {} truth {} prediction {}",
                i,
                CONV_0_UPDATED_WEIGHTS[i],
                flat_conv_0_kernel[i]
            );
        }
    }
}

// #[test]
// fn test_3_x_3_f64() {
//     let w0: Vec<f64> = vec![0.7, 0.8, 0.9];
//     let w1: Vec<f64> = vec![1.0, 1.1, 1.2];
//     let W = vec![w0, w1];

//     let x0: Vec<f64> = vec![0.1, 0.2];
//     let x1: Vec<f64> = vec![0.3, 0.4];
//     let x2: Vec<f64> = vec![0.5, 0.6];
//     let X = vec![x0, x1, x2];

//     let b0: Vec<f64> = vec![0.0; 3];
//     let b1: Vec<f64> = vec![0.0; 3];
//     let b2: Vec<f64> = vec![0.0; 3];
//     let B = vec![b0, b1, b2];

//     let y0: Vec<f64> = vec![0.5, 0.6, 0.7];
//     let y1: Vec<f64> = vec![0.8, 0.9, 1.0];
//     let y2: Vec<f64> = vec![1.1, 1.2, 1.3];
//     let Y = vec![y0, y1, y2];

//     let Z: Vec<f64> = vec![
//         -0.22999999999999998,
//         -0.29999999999999993,
//         -0.36999999999999994,
//         -0.19000000000000006,
//         -0.21999999999999997,
//         -0.25,
//         -0.15000000000000013,
//         -0.1399999999999999,
//         -0.13000000000000012,
//     ];

//     let D_Z = vec![
//         -0.051111111111111107,
//         -0.066666666666666652,
//         -0.08222222222222221,
//         -0.04222222222222223,
//         -0.048888888888888878,
//         -0.055555555555555552,
//         -0.033333333333333361,
//         -0.031111111111111089,
//         -0.028888888888888912,
//     ];

//     let U_X = vec![
//         0.11046666666666667,
//         0.21342222222222224,
//         0.3,
//         0.41204444444444444,
//         0.5154666666666666,
//         0.6,
//     ];

//     let U_W = vec![
//         0.7034444444444444,
//         0.8036888888888889,
//         0.9039333333333334,
//         1.004711111111111,
//         1.1051555555555557,
//         1.2056,
//     ];
//     let (mut data, l) = layer_factory::<f64>(Layers::Linear, X, W, B, None, 0.0);
//     let data1 = (data.0.as_slice(), data.1.as_mut_slice(), data.2.as_slice());
//     let mut z = l.forward(data1, None);

//     let (flattened, squared) = loss_function_factory(LossFunctions::MeanSquares, Y, 1.0);
//     let loss = squared.forward(&flattened, &z);
//     for i in 0..Z.len() {
//         assert!((Z[i] - z[i]).abs() < f64::EPSILON);
//     }
//     assert!((loss[0] - 0.05397777777777777).abs() < f64::EPSILON);
//     // squared.backward(&z);
//     for i in 0..D_Z.len() {
//         assert!((D_Z[i] - z[i]).abs() < f64::EPSILON);
//     }
//     let data2 = (
//         data.0.as_mut_slice(),
//         data.1.as_mut_slice(),
//         data.2.as_slice(),
//     );
//     let updated = l.backward(data2, &mut z, 0.1, 0.0);
//     for i in 0..U_W.len() {
//         assert!((U_W[i] - updated.1[i]).abs() < f64::EPSILON);
//     }
//     for i in 0..U_X.len() {
//         assert!((U_X[i] - updated.0[i]).abs() < f64::EPSILON);
//     }
// }

// #[test]
// fn test_3_x_3_f32() {
//     let w0: Vec<f32> = vec![0.7, 0.8, 0.9];
//     let w1: Vec<f32> = vec![1.0, 1.1, 1.2];
//     let W = vec![w0, w1];

//     let x0: Vec<f32> = vec![0.1, 0.2];
//     let x1: Vec<f32> = vec![0.3, 0.4];
//     let x2: Vec<f32> = vec![0.5, 0.6];
//     let X = vec![x0, x1, x2];

//     let b0: Vec<f32> = vec![0.0; 3];
//     let b1: Vec<f32> = vec![0.0; 3];
//     let b2: Vec<f32> = vec![0.0; 3];
//     let B = vec![b0, b1, b2];

//     let y0: Vec<f32> = vec![0.5, 0.6, 0.7];
//     let y1: Vec<f32> = vec![0.8, 0.9, 1.0];
//     let y2: Vec<f32> = vec![1.1, 1.2, 1.3];
//     let Y = vec![y0, y1, y2];

//     let Z: Vec<f32> = vec![
//         -0.22999999,
//         -0.3,
//         -0.36999997,
//         -0.19,
//         -0.21999991,
//         -0.25,
//         -0.14999998,
//         -0.13999999,
//         -0.12999988,
//     ];

//     let D_Z: Vec<f32> = vec![
//         -0.051111111111111107,
//         -0.066666666666666652,
//         -0.08222222222222221,
//         -0.04222222222222223,
//         -0.048888888888888878,
//         -0.055555555555555552,
//         -0.033333333333333361,
//         -0.031111111111111089,
//         -0.028888888888888912,
//     ];

//     let D_X: Vec<f32> = vec![
//         0.11046666666666667,
//         0.21342222222222224,
//         0.3,
//         0.41204444444444444,
//         0.5154666666666666,
//         0.6,
//     ];

//     let D_W: Vec<f32> = vec![
//         0.7034444444444444,
//         0.8036888888888889,
//         0.9039333333333334,
//         1.004711111111111,
//         1.1051555555555557,
//         1.2056,
//     ];
//     let (mut data, l) = layer_factory::<f32>(Layers::Linear, X, W, B, None, 0.0);
//     let data1 = (data.0.as_slice(), data.1.as_mut_slice(), data.2.as_slice());
//     let mut z = l.forward(data1, None);

//     let (flattened, squared) = loss_function_factory(LossFunctions::MeanSquares, Y, 1.0);
//     let loss = squared.forward(&flattened, &z);
//     for i in 0..Z.len() {
//         assert!((Z[i] - z[i]).abs() < f32::EPSILON);
//     }
//     assert!((loss[0] - 0.05397777777777777).abs() < f32::EPSILON);
//     //squared.backward(&z);
//     for i in 0..D_Z.len() {
//         assert!((D_Z[i] - z[i]).abs() < f32::EPSILON);
//     }
//     let data2 = (
//         data.0.as_mut_slice(),
//         data.1.as_mut_slice(),
//         data.2.as_slice(),
//     );
//     let updated = l.backward(data2, &mut z, 0.1, 0.0);
//     for i in 0..D_W.len() {
//         assert!((D_W[i] - updated.1[i]).abs() < f32::EPSILON);
//     }
//     for i in 0..D_X.len() {
//         assert!((D_X[i] - updated.0[i]).abs() < f32::EPSILON);
//     }
// }

// #[test]
// fn test_conv2d_5_3_f32() {
//     let features: Vec<Vec<f32>> = vec![
//         vec![1.0, 1.0, 1.0, 0.0, 0.0],
//         vec![0.0, 1.0, 1.0, 1.0, 0.0],
//         vec![0.0, 0.0, 1.0, 1.0, 1.0],
//         vec![0.0, 0.0, 1.0, 1.0, 0.0],
//         vec![0.0, 1.0, 1.0, 0.0, 0.0],
//     ];

//     let kernel: Vec<Vec<f32>> = vec![
//         vec![1.0, 0.0, 1.0],
//         vec![0.0, 1.0, 0.0],
//         vec![1.0, 0.0, 1.0],
//     ];

//     let Z: Vec<f32> = vec![4.0, 3.0, 4.0, 2.0, 4.0, 3.0, 2.0, 3.0, 4.0];
//     let D_X: [f32; 25] = [
//         4.0, 3.0, 8.0, 3.0, 4.0, 2.0, 8.0, 8.0, 8.0, 3.0, 6.0, 8.0, 18.0, 9.0, 8.0, 2.0, 6.0,
//         8.0, 8.0, 3.0, 2.0, 3.0, 6.0, 3.0, 4.0,
//     ];
//     let D_W: [f32; 9] = [22.0, 23.0, 19.0, 14.0, 25.0, 21.0, 14.0, 19.0, 19.0];

//     let bias: Vec<Vec<f32>> = vec![];
//     let p_s = (0 as u16, 1 as u16);
//     let (mut data, l) =
//         layer_factory::<f32>(Layers::Conv2D, features, kernel, bias, Some(p_s), 0.0);
//     let data1 = (data.0.as_slice(), data.1.as_mut_slice(), data.2.as_slice());
//     let mut z = l.forward(data1, None);
//     for i in 0..Z.len() {
//         assert!(Z[i] == z[i], "{} truth {} prediction {}", i, Z[i], z[i]);
//     }
//     let data2 = (
//         data.0.as_mut_slice(),
//         data.1.as_mut_slice(),
//         data.2.as_slice(),
//     );
//     let updated = l.backward(data2, z.as_mut_slice(), 0.1, 0.0);
//     for i in 0..D_W.len() {
//         assert!(
//             (D_W[i] - updated.1[i]).abs() < f32::EPSILON,
//             "{} truth {} prediction {}",
//             i,
//             D_W[i],
//             updated.1[i]
//         );
//     }

//     for i in 0..D_X.len() {
//         assert!(
//             (D_X[i] - updated.0[i]).abs() < f32::EPSILON,
//             "{} truth {} prediction {}",
//             i,
//             D_X[i],
//             updated.0[i]
//         );
//     }
// }

// #[test]
// fn test_conv2d_5_3_f64() {
//     let features = vec![
//         vec![1.0, 1.0, 1.0, 0.0, 0.0],
//         vec![0.0, 1.0, 1.0, 1.0, 0.0],
//         vec![0.0, 0.0, 1.0, 1.0, 1.0],
//         vec![0.0, 0.0, 1.0, 1.0, 0.0],
//         vec![0.0, 1.0, 1.0, 0.0, 0.0],
//     ];

//     let kernel = vec![
//         vec![1.0, 0.0, 1.0],
//         vec![0.0, 1.0, 0.0],
//         vec![1.0, 0.0, 1.0],
//     ];

//     let Z = vec![4.0, 3.0, 4.0, 2.0, 4.0, 3.0, 2.0, 3.0, 4.0];
//     let D_X = [
//         4.0, 3.0, 8.0, 3.0, 4.0, 2.0, 8.0, 8.0, 8.0, 3.0, 6.0, 8.0, 18.0, 9.0, 8.0, 2.0, 6.0,
//         8.0, 8.0, 3.0, 2.0, 3.0, 6.0, 3.0, 4.0,
//     ];
//     let D_W = [22.0, 23.0, 19.0, 14.0, 25.0, 21.0, 14.0, 19.0, 19.0];

//     let bias: Vec<Vec<f64>> = vec![];
//     let p_s = (0 as u16, 1 as u16);
//     let (mut data, l) =
//         layer_factory::<f64>(Layers::Conv2D, features, kernel, bias, Some(p_s), 0.0);
//     let data1 = (data.0.as_slice(), data.1.as_mut_slice(), data.2.as_slice());
//     let mut z = l.forward(data1, None);
//     for i in 0..Z.len() {
//         assert!(Z[i] == z[i], "{} truth {} prediction {}", i, Z[i], z[i]);
//     }
//     let data2 = (
//         data.0.as_mut_slice(),
//         data.1.as_mut_slice(),
//         data.2.as_slice(),
//     );
//     let updated = l.backward(data2, z.as_mut_slice(), 0.1, 0.0);
//     for i in 0..D_W.len() {
//         assert!(
//             (D_W[i] - updated.1[i]).abs() < f64::EPSILON,
//             "{} truth {} prediction {}",
//             i,
//             D_W[i],
//             updated.1[i]
//         );
//     }

//     for i in 0..D_X.len() {
//         assert!(
//             (D_X[i] - updated.0[i]).abs() < f64::EPSILON,
//             "{} truth {} prediction {}",
//             i,
//             D_X[i],
//             updated.0[i]
//         );
//     }
// }

// #[test]
// fn test_conv2d_20_17_f64() {
//     let features = vec![
//         vec![
//             6.0, 3.0, 7.0, 4.0, 6.0, 9.0, 2.0, 6.0, 7.0, 4.0, 3.0, 7.0, 7.0, 2.0, 5.0, 4.0,
//             1.0, 7.0, 5.0, 1.0,
//         ],
//         vec![
//             4.0, 0.0, 9.0, 5.0, 8.0, 0.0, 9.0, 2.0, 6.0, 3.0, 8.0, 2.0, 4.0, 2.0, 6.0, 4.0,
//             8.0, 6.0, 1.0, 3.0,
//         ],
//         vec![
//             8.0, 1.0, 9.0, 8.0, 9.0, 4.0, 1.0, 3.0, 6.0, 7.0, 2.0, 0.0, 3.0, 1.0, 7.0, 3.0,
//             1.0, 5.0, 5.0, 9.0,
//         ],
//         vec![
//             3.0, 5.0, 1.0, 9.0, 1.0, 9.0, 3.0, 7.0, 6.0, 8.0, 7.0, 4.0, 1.0, 4.0, 7.0, 9.0,
//             8.0, 8.0, 0.0, 8.0,
//         ],
//         vec![
//             6.0, 8.0, 7.0, 0.0, 7.0, 7.0, 2.0, 0.0, 7.0, 2.0, 2.0, 0.0, 4.0, 9.0, 6.0, 9.0,
//             8.0, 6.0, 8.0, 7.0,
//         ],
//         vec![
//             1.0, 0.0, 6.0, 6.0, 7.0, 4.0, 2.0, 7.0, 5.0, 2.0, 0.0, 2.0, 4.0, 2.0, 0.0, 4.0,
//             9.0, 6.0, 6.0, 8.0,
//         ],
//         vec![
//             9.0, 9.0, 2.0, 6.0, 0.0, 3.0, 3.0, 4.0, 6.0, 6.0, 3.0, 6.0, 2.0, 5.0, 1.0, 9.0,
//             8.0, 4.0, 5.0, 3.0,
//         ],
//         vec![
//             9.0, 6.0, 8.0, 6.0, 0.0, 0.0, 8.0, 8.0, 3.0, 8.0, 2.0, 6.0, 5.0, 7.0, 8.0, 4.0,
//             0.0, 2.0, 9.0, 7.0,
//         ],
//         vec![
//             5.0, 7.0, 8.0, 3.0, 0.0, 0.0, 9.0, 3.0, 6.0, 1.0, 2.0, 0.0, 4.0, 0.0, 7.0, 0.0,
//             0.0, 1.0, 1.0, 5.0,
//         ],
//         vec![
//             6.0, 4.0, 0.0, 0.0, 2.0, 1.0, 4.0, 9.0, 5.0, 6.0, 3.0, 6.0, 7.0, 0.0, 5.0, 7.0,
//             4.0, 3.0, 1.0, 5.0,
//         ],
//         vec![
//             5.0, 0.0, 8.0, 5.0, 2.0, 3.0, 3.0, 2.0, 9.0, 2.0, 2.0, 3.0, 6.0, 3.0, 8.0, 0.0,
//             7.0, 6.0, 1.0, 7.0,
//         ],
//         vec![
//             0.0, 8.0, 8.0, 1.0, 6.0, 9.0, 2.0, 6.0, 9.0, 8.0, 3.0, 0.0, 1.0, 0.0, 4.0, 4.0,
//             6.0, 8.0, 8.0, 2.0,
//         ],
//         vec![
//             2.0, 2.0, 3.0, 7.0, 5.0, 7.0, 0.0, 7.0, 3.0, 0.0, 7.0, 3.0, 5.0, 7.0, 3.0, 2.0,
//             8.0, 2.0, 8.0, 1.0,
//         ],
//         vec![
//             1.0, 1.0, 5.0, 2.0, 8.0, 3.0, 0.0, 3.0, 0.0, 4.0, 3.0, 7.0, 7.0, 6.0, 2.0, 0.0,
//             0.0, 2.0, 5.0, 6.0,
//         ],
//         vec![
//             5.0, 5.0, 5.0, 2.0, 5.0, 7.0, 1.0, 4.0, 0.0, 0.0, 4.0, 2.0, 3.0, 2.0, 0.0, 0.0,
//             4.0, 5.0, 2.0, 8.0,
//         ],
//         vec![
//             4.0, 7.0, 0.0, 4.0, 2.0, 0.0, 3.0, 4.0, 6.0, 0.0, 2.0, 1.0, 8.0, 9.0, 5.0, 9.0,
//             2.0, 7.0, 7.0, 1.0,
//         ],
//         vec![
//             5.0, 6.0, 1.0, 9.0, 1.0, 9.0, 0.0, 7.0, 0.0, 8.0, 5.0, 6.0, 9.0, 6.0, 9.0, 2.0,
//             1.0, 8.0, 7.0, 9.0,
//         ],
//         vec![
//             6.0, 8.0, 3.0, 3.0, 0.0, 7.0, 2.0, 6.0, 1.0, 1.0, 6.0, 5.0, 2.0, 8.0, 9.0, 5.0,
//             9.0, 9.0, 5.0, 0.0,
//         ],
//         vec![
//             3.0, 9.0, 5.0, 5.0, 4.0, 0.0, 7.0, 4.0, 4.0, 6.0, 3.0, 5.0, 3.0, 2.0, 6.0, 7.0,
//             3.0, 1.0, 9.0, 2.0,
//         ],
//         vec![
//             0.0, 7.0, 2.0, 9.0, 6.0, 9.0, 4.0, 9.0, 4.0, 6.0, 8.0, 4.0, 0.0, 9.0, 9.0, 0.0,
//             1.0, 5.0, 8.0, 7.0,
//         ],
//     ];

//     let kernel = vec![
//         vec![
//             4.0, 0.0, 6.0, 4.0, 5.0, 6.0, 2.0, 9.0, 2.0, 4.0, 5.0, 8.0, 4.0, 0.0, 3.0, 4.0, 9.0,
//         ],
//         vec![
//             9.0, 4.0, 6.0, 3.0, 0.0, 4.0, 6.0, 9.0, 9.0, 5.0, 4.0, 3.0, 1.0, 3.0, 9.0, 9.0, 2.0,
//         ],
//         vec![
//             9.0, 0.0, 7.0, 4.0, 3.0, 7.0, 6.0, 1.0, 0.0, 3.0, 7.0, 1.0, 2.0, 0.0, 0.0, 2.0, 4.0,
//         ],
//         vec![
//             2.0, 0.0, 0.0, 7.0, 9.0, 1.0, 2.0, 1.0, 2.0, 6.0, 0.0, 9.0, 7.0, 9.0, 9.0, 9.0, 1.0,
//         ],
//         vec![
//             2.0, 8.0, 6.0, 3.0, 9.0, 4.0, 1.0, 7.0, 3.0, 8.0, 4.0, 8.0, 3.0, 9.0, 4.0, 8.0, 7.0,
//         ],
//         vec![
//             2.0, 0.0, 2.0, 3.0, 1.0, 0.0, 6.0, 7.0, 6.0, 4.0, 0.0, 6.0, 6.0, 8.0, 2.0, 8.0, 0.0,
//         ],
//         vec![
//             0.0, 3.0, 8.0, 5.0, 2.0, 0.0, 3.0, 8.0, 2.0, 8.0, 6.0, 3.0, 2.0, 9.0, 4.0, 4.0, 2.0,
//         ],
//         vec![
//             8.0, 3.0, 4.0, 3.0, 4.0, 6.0, 8.0, 6.0, 4.0, 9.0, 9.0, 6.0, 9.0, 4.0, 2.0, 6.0, 1.0,
//         ],
//         vec![
//             8.0, 9.0, 9.0, 0.0, 5.0, 6.0, 7.0, 9.0, 8.0, 1.0, 9.0, 1.0, 4.0, 4.0, 5.0, 2.0, 7.0,
//         ],
//         vec![
//             0.0, 5.0, 3.0, 0.0, 6.0, 8.0, 3.0, 3.0, 5.0, 2.0, 5.0, 6.0, 9.0, 9.0, 2.0, 6.0, 2.0,
//         ],
//         vec![
//             1.0, 9.0, 3.0, 7.0, 8.0, 6.0, 0.0, 2.0, 8.0, 0.0, 8.0, 7.0, 0.0, 5.0, 4.0, 5.0, 9.0,
//         ],
//         vec![
//             4.0, 5.0, 4.0, 4.0, 3.0, 2.0, 2.0, 3.0, 8.0, 1.0, 8.0, 0.0, 0.0, 4.0, 5.0, 5.0, 2.0,
//         ],
//         vec![
//             6.0, 8.0, 9.0, 7.0, 5.0, 7.0, 4.0, 7.0, 9.0, 3.0, 9.0, 7.0, 9.0, 1.0, 4.0, 8.0, 3.0,
//         ],
//         vec![
//             5.0, 0.0, 8.0, 0.0, 4.0, 3.0, 2.0, 5.0, 1.0, 2.0, 4.0, 8.0, 1.0, 9.0, 7.0, 1.0, 4.0,
//         ],
//         vec![
//             6.0, 7.0, 0.0, 5.0, 0.0, 1.0, 0.0, 4.0, 9.0, 8.0, 5.0, 0.0, 0.0, 1.0, 8.0, 2.0, 0.0,
//         ],
//         vec![
//             4.0, 6.0, 5.0, 0.0, 4.0, 4.0, 5.0, 2.0, 4.0, 6.0, 4.0, 4.0, 4.0, 9.0, 9.0, 2.0, 0.0,
//         ],
//         vec![
//             4.0, 8.0, 0.0, 2.0, 3.0, 0.0, 0.0, 7.0, 1.0, 7.0, 6.0, 9.0, 9.0, 1.0, 5.0, 5.0, 2.0,
//         ],
//     ];

//     let Z = vec![
//         5828.0, 5589.0, 5940.0, 5524.0, 5584.0, 5518.0, 5278.0, 5559.0, 5662.0, 5239.0, 5492.0,
//         5496.0, 5533.0, 5497.0, 5665.0, 5656.0,
//     ];

//     let D_W = [
//         457628.0, 470051.0, 544435.0, 486607.0, 439072.0, 448728.0, 445622.0, 473486.0,
//         444122.0, 391674.0, 319477.0, 346131.0, 386283.0, 399896.0, 494353.0, 418066.0,
//         422179.0, 461522.0, 482502.0, 517802.0, 456132.0, 401716.0, 399779.0, 401173.0,
//         419723.0, 390356.0, 317090.0, 297867.0, 335017.0, 439144.0, 510474.0, 563319.0,
//         495460.0, 505599.0, 437303.0, 466869.0, 528284.0, 437920.0, 406245.0, 406620.0,
//         378363.0, 393174.0, 333868.0, 267321.0, 251396.0, 296998.0, 403823.0, 481649.0,
//         531513.0, 526338.0, 567632.0, 431805.0, 413563.0, 413582.0, 385529.0, 367043.0,
//         421716.0, 388633.0, 403703.0, 370927.0, 298754.0, 306128.0, 315694.0, 423407.0,
//         547576.0, 567863.0, 595664.0, 565302.0, 495970.0, 433656.0, 383330.0, 341493.0,
//         347173.0, 382666.0, 405939.0, 360194.0, 334627.0, 300051.0, 327247.0, 375748.0,
//         441140.0, 498261.0, 465158.0, 540700.0, 537621.0, 506176.0, 409340.0, 330616.0,
//         320357.0, 324181.0, 394202.0, 448959.0, 366909.0, 320049.0, 293833.0, 276613.0,
//         326540.0, 343343.0, 352537.0, 350297.0, 380121.0, 413449.0, 487312.0, 340052.0,
//         217541.0, 253069.0, 299422.0, 397589.0, 496735.0, 416770.0, 386045.0, 374989.0,
//         322625.0, 383267.0, 394003.0, 363162.0, 354509.0, 325825.0, 323482.0, 448290.0,
//         331639.0, 257911.0, 256519.0, 299750.0, 406504.0, 477510.0, 397989.0, 355083.0,
//         354022.0, 311217.0, 419619.0, 397736.0, 334684.0, 345969.0, 256432.0, 330204.0,
//         382609.0, 344987.0, 313124.0, 278314.0, 340307.0, 443054.0, 470290.0, 419607.0,
//         362329.0, 297406.0, 220967.0, 297934.0, 313725.0, 301849.0, 390729.0, 314820.0,
//         359950.0, 325254.0, 337180.0, 372845.0, 314068.0, 376214.0, 442363.0, 414493.0,
//         451474.0, 384731.0, 347698.0, 314282.0, 339807.0, 345511.0, 379130.0, 428316.0,
//         415955.0, 427072.0, 323624.0, 395469.0, 455448.0, 348332.0, 365545.0, 366998.0,
//         321684.0, 377002.0, 352809.0, 339082.0, 352617.0, 361409.0, 323681.0, 331897.0,
//         346666.0, 370225.0, 428389.0, 318847.0, 404452.0, 464038.0, 361068.0, 407686.0,
//         342047.0, 262524.0, 344114.0, 295235.0, 316737.0, 330881.0, 284788.0, 254827.0,
//         268070.0, 279886.0, 358987.0, 418848.0, 304124.0, 351497.0, 360870.0, 313020.0,
//         327978.0, 271462.0, 194928.0, 261830.0, 235599.0, 309647.0, 424990.0, 390880.0,
//         380313.0, 331062.0, 287024.0, 350237.0, 380277.0, 343704.0, 351636.0, 352184.0,
//         311344.0, 318405.0, 261135.0, 221140.0, 279793.0, 267763.0, 386213.0, 447186.0,
//         456932.0, 428445.0, 316780.0, 308232.0, 338907.0, 410981.0, 406868.0, 339788.0,
//         321765.0, 309945.0, 320567.0, 319938.0, 234686.0, 301316.0, 260075.0, 343653.0,
//         431708.0, 463750.0, 476536.0, 440787.0, 468498.0, 455629.0, 462705.0, 432393.0,
//         373407.0, 293776.0, 310005.0, 310775.0, 333196.0, 329639.0, 349758.0, 327982.0,
//         385801.0, 443875.0, 518095.0, 549644.0, 514228.0, 511404.0, 505861.0, 444550.0,
//         448279.0, 437844.0, 405910.0, 420443.0, 416644.0, 407402.0, 382163.0, 437657.0,
//         399217.0, 429596.0, 453205.0, 513844.0, 480449.0, 476044.0, 465500.0, 445689.0,
//         468719.0,
//     ];

//     let D_X = [
//         23312.0, 22356.0, 58728.0, 78942.0, 87136.0, 119817.0, 96986.0, 126890.0, 106981.0,
//         98998.0, 113092.0, 109377.0, 119820.0, 97496.0, 85436.0, 62175.0, 92628.0, 90633.0,
//         75556.0, 49716.0, 74788.0, 95685.0, 165400.0, 202174.0, 156163.0, 189836.0, 166798.0,
//         230501.0, 265229.0, 258037.0, 265012.0, 223748.0, 189081.0, 155080.0, 174067.0,
//         187223.0, 220151.0, 201805.0, 131334.0, 61079.0, 125356.0, 143255.0, 253274.0,
//         309220.0, 234096.0, 306626.0, 278462.0, 323858.0, 336154.0, 308552.0, 320260.0,
//         282467.0, 260446.0, 208498.0, 189091.0, 204364.0, 247889.0, 228412.0, 166807.0,
//         82678.0, 135002.0, 152627.0, 258684.0, 360427.0, 321641.0, 402755.0, 385986.0,
//         392398.0, 369598.0, 371806.0, 371761.0, 379771.0, 382434.0, 356535.0, 383326.0,
//         398495.0, 405848.0, 339864.0, 221943.0, 89656.0, 123579.0, 187594.0, 297349.0,
//         391722.0, 387424.0, 405623.0, 382576.0, 389257.0, 349991.0, 388944.0, 386088.0,
//         402741.0, 396711.0, 391376.0, 427921.0, 469582.0, 473381.0, 356613.0, 236275.0,
//         77523.0, 83945.0, 126837.0, 212440.0, 305053.0, 343298.0, 364374.0, 365074.0, 361837.0,
//         304745.0, 352435.0, 326683.0, 373426.0, 410025.0, 440378.0, 460459.0, 477776.0,
//         439990.0, 299103.0, 214538.0, 67033.0, 33558.0, 95288.0, 183313.0, 283969.0, 368258.0,
//         331452.0, 312117.0, 341595.0, 292670.0, 413760.0, 397881.0, 419331.0, 444077.0,
//         495898.0, 542471.0, 563477.0, 516533.0, 320355.0, 217429.0, 55176.0, 69014.0, 144684.0,
//         259637.0, 334419.0, 355959.0, 334381.0, 327019.0, 404116.0, 395620.0, 500817.0,
//         507408.0, 481483.0, 508897.0, 513363.0, 472859.0, 489541.0, 430856.0, 268012.0,
//         200747.0, 56234.0, 96534.0, 139655.0, 246421.0, 326538.0, 294392.0, 288920.0, 330747.0,
//         365091.0, 417598.0, 546191.0, 524931.0, 517789.0, 527632.0, 478074.0, 449631.0,
//         484594.0, 426135.0, 307254.0, 214084.0, 60743.0, 84384.0, 154431.0, 264387.0, 332884.0,
//         339984.0, 351248.0, 368353.0, 400562.0, 414201.0, 490082.0, 509928.0, 525265.0,
//         562818.0, 528121.0, 473148.0, 499334.0, 425304.0, 305004.0, 209701.0, 72328.0, 89726.0,
//         194533.0, 279242.0, 358620.0, 390723.0, 403160.0, 429309.0, 414449.0, 435555.0,
//         426710.0, 479298.0, 549466.0, 539835.0, 528411.0, 460980.0, 466614.0, 450984.0,
//         347522.0, 257991.0, 110458.0, 67627.0, 185125.0, 268204.0, 349883.0, 402068.0,
//         377229.0, 369423.0, 337332.0, 386254.0, 354933.0, 436235.0, 489089.0, 405944.0,
//         436450.0, 393272.0, 428119.0, 466078.0, 367073.0, 254981.0, 117319.0, 62966.0,
//         214012.0, 317555.0, 419886.0, 480028.0, 456137.0, 399484.0, 387963.0, 432052.0,
//         333551.0, 454614.0, 478334.0, 412415.0, 486495.0, 403796.0, 424065.0, 411747.0,
//         304495.0, 222537.0, 88466.0, 90825.0, 210285.0, 344940.0, 441571.0, 467384.0, 442477.0,
//         356604.0, 351375.0, 383315.0, 313937.0, 433130.0, 460326.0, 371640.0, 449064.0,
//         393973.0, 380439.0, 422949.0, 313818.0, 207319.0, 100669.0, 118992.0, 228303.0,
//         343924.0, 434569.0, 379079.0, 345402.0, 268975.0, 288683.0, 372207.0, 377260.0,
//         482477.0, 455654.0, 360373.0, 361783.0, 356153.0, 358597.0, 354915.0, 271571.0,
//         137773.0, 50036.0, 118324.0, 232961.0, 357247.0, 420753.0, 381620.0, 343947.0,
//         286106.0, 318973.0, 374030.0, 395636.0, 456419.0, 462084.0, 410197.0, 421725.0,
//         453250.0, 418183.0, 377859.0, 265979.0, 111873.0, 38952.0, 107285.0, 223109.0,
//         292826.0, 334752.0, 287053.0, 217807.0, 182908.0, 245892.0, 265675.0, 352507.0,
//         415078.0, 434235.0, 430844.0, 414789.0, 470823.0, 402690.0, 361492.0, 251282.0,
//         89926.0, 33672.0, 78182.0, 193385.0, 219437.0, 258015.0, 222421.0, 132052.0, 131963.0,
//         165033.0, 204221.0, 291341.0, 348613.0, 348118.0, 342620.0, 285781.0, 322793.0,
//         301413.0, 242047.0, 182247.0, 60655.0, 11118.0, 44780.0, 121438.0, 147187.0, 161343.0,
//         155825.0, 99101.0, 99781.0, 139957.0, 126410.0, 178113.0, 203695.0, 227066.0, 270249.0,
//         252286.0, 276995.0, 243609.0, 183358.0, 127652.0, 49776.0, 10992.0, 22132.0, 66252.0,
//         66636.0, 79010.0, 72841.0, 27821.0, 28307.0, 55699.0, 44012.0, 83883.0, 116934.0,
//         128090.0, 172852.0, 139927.0, 135051.0, 111719.0, 72532.0, 67599.0, 39610.0, 11312.0,
//     ];

//     let bias: Vec<Vec<f64>> = vec![];
//     let p_s = (0 as u16, 1 as u16);
//     let (mut data, l) =
//         layer_factory::<f64>(Layers::Conv2D, features, kernel, bias, Some(p_s), 0.0);
//     let data1 = (data.0.as_slice(), data.1.as_mut_slice(), data.2.as_slice());
//     let mut z = l.forward(data1, None);
//     for i in 0..Z.len() {
//         assert!(Z[i] == z[i]);
//     }
//     let data2 = (
//         data.0.as_mut_slice(),
//         data.1.as_mut_slice(),
//         data.2.as_slice(),
//     );
//     let updated = l.backward(data2, z.as_mut_slice(), 0.1, 0.0);
//     for i in 0..D_W.len() {
//         assert!(
//             (D_W[i] - updated.1[i]).abs() < f64::EPSILON,
//             "{} truth {} prediction {}",
//             i,
//             D_W[i],
//             updated.1[i]
//         );
//     }

//     for i in 0..D_X.len() {
//         assert!(
//             (D_X[i] - updated.0[i]).abs() < f64::EPSILON,
//             "{} truth {} prediction {}",
//             i,
//             D_X[i],
//             updated.0[i]
//         );
//     }
// }

// #[test]
// fn test_conv2d_20_17_f32() {
//     let features: Vec<Vec<f32>> = vec![
//         vec![
//             6.0, 3.0, 7.0, 4.0, 6.0, 9.0, 2.0, 6.0, 7.0, 4.0, 3.0, 7.0, 7.0, 2.0, 5.0, 4.0,
//             1.0, 7.0, 5.0, 1.0,
//         ],
//         vec![
//             4.0, 0.0, 9.0, 5.0, 8.0, 0.0, 9.0, 2.0, 6.0, 3.0, 8.0, 2.0, 4.0, 2.0, 6.0, 4.0,
//             8.0, 6.0, 1.0, 3.0,
//         ],
//         vec![
//             8.0, 1.0, 9.0, 8.0, 9.0, 4.0, 1.0, 3.0, 6.0, 7.0, 2.0, 0.0, 3.0, 1.0, 7.0, 3.0,
//             1.0, 5.0, 5.0, 9.0,
//         ],
//         vec![
//             3.0, 5.0, 1.0, 9.0, 1.0, 9.0, 3.0, 7.0, 6.0, 8.0, 7.0, 4.0, 1.0, 4.0, 7.0, 9.0,
//             8.0, 8.0, 0.0, 8.0,
//         ],
//         vec![
//             6.0, 8.0, 7.0, 0.0, 7.0, 7.0, 2.0, 0.0, 7.0, 2.0, 2.0, 0.0, 4.0, 9.0, 6.0, 9.0,
//             8.0, 6.0, 8.0, 7.0,
//         ],
//         vec![
//             1.0, 0.0, 6.0, 6.0, 7.0, 4.0, 2.0, 7.0, 5.0, 2.0, 0.0, 2.0, 4.0, 2.0, 0.0, 4.0,
//             9.0, 6.0, 6.0, 8.0,
//         ],
//         vec![
//             9.0, 9.0, 2.0, 6.0, 0.0, 3.0, 3.0, 4.0, 6.0, 6.0, 3.0, 6.0, 2.0, 5.0, 1.0, 9.0,
//             8.0, 4.0, 5.0, 3.0,
//         ],
//         vec![
//             9.0, 6.0, 8.0, 6.0, 0.0, 0.0, 8.0, 8.0, 3.0, 8.0, 2.0, 6.0, 5.0, 7.0, 8.0, 4.0,
//             0.0, 2.0, 9.0, 7.0,
//         ],
//         vec![
//             5.0, 7.0, 8.0, 3.0, 0.0, 0.0, 9.0, 3.0, 6.0, 1.0, 2.0, 0.0, 4.0, 0.0, 7.0, 0.0,
//             0.0, 1.0, 1.0, 5.0,
//         ],
//         vec![
//             6.0, 4.0, 0.0, 0.0, 2.0, 1.0, 4.0, 9.0, 5.0, 6.0, 3.0, 6.0, 7.0, 0.0, 5.0, 7.0,
//             4.0, 3.0, 1.0, 5.0,
//         ],
//         vec![
//             5.0, 0.0, 8.0, 5.0, 2.0, 3.0, 3.0, 2.0, 9.0, 2.0, 2.0, 3.0, 6.0, 3.0, 8.0, 0.0,
//             7.0, 6.0, 1.0, 7.0,
//         ],
//         vec![
//             0.0, 8.0, 8.0, 1.0, 6.0, 9.0, 2.0, 6.0, 9.0, 8.0, 3.0, 0.0, 1.0, 0.0, 4.0, 4.0,
//             6.0, 8.0, 8.0, 2.0,
//         ],
//         vec![
//             2.0, 2.0, 3.0, 7.0, 5.0, 7.0, 0.0, 7.0, 3.0, 0.0, 7.0, 3.0, 5.0, 7.0, 3.0, 2.0,
//             8.0, 2.0, 8.0, 1.0,
//         ],
//         vec![
//             1.0, 1.0, 5.0, 2.0, 8.0, 3.0, 0.0, 3.0, 0.0, 4.0, 3.0, 7.0, 7.0, 6.0, 2.0, 0.0,
//             0.0, 2.0, 5.0, 6.0,
//         ],
//         vec![
//             5.0, 5.0, 5.0, 2.0, 5.0, 7.0, 1.0, 4.0, 0.0, 0.0, 4.0, 2.0, 3.0, 2.0, 0.0, 0.0,
//             4.0, 5.0, 2.0, 8.0,
//         ],
//         vec![
//             4.0, 7.0, 0.0, 4.0, 2.0, 0.0, 3.0, 4.0, 6.0, 0.0, 2.0, 1.0, 8.0, 9.0, 5.0, 9.0,
//             2.0, 7.0, 7.0, 1.0,
//         ],
//         vec![
//             5.0, 6.0, 1.0, 9.0, 1.0, 9.0, 0.0, 7.0, 0.0, 8.0, 5.0, 6.0, 9.0, 6.0, 9.0, 2.0,
//             1.0, 8.0, 7.0, 9.0,
//         ],
//         vec![
//             6.0, 8.0, 3.0, 3.0, 0.0, 7.0, 2.0, 6.0, 1.0, 1.0, 6.0, 5.0, 2.0, 8.0, 9.0, 5.0,
//             9.0, 9.0, 5.0, 0.0,
//         ],
//         vec![
//             3.0, 9.0, 5.0, 5.0, 4.0, 0.0, 7.0, 4.0, 4.0, 6.0, 3.0, 5.0, 3.0, 2.0, 6.0, 7.0,
//             3.0, 1.0, 9.0, 2.0,
//         ],
//         vec![
//             0.0, 7.0, 2.0, 9.0, 6.0, 9.0, 4.0, 9.0, 4.0, 6.0, 8.0, 4.0, 0.0, 9.0, 9.0, 0.0,
//             1.0, 5.0, 8.0, 7.0,
//         ],
//     ];

//     let kernel: Vec<Vec<f32>> = vec![
//         vec![
//             4.0, 0.0, 6.0, 4.0, 5.0, 6.0, 2.0, 9.0, 2.0, 4.0, 5.0, 8.0, 4.0, 0.0, 3.0, 4.0, 9.0,
//         ],
//         vec![
//             9.0, 4.0, 6.0, 3.0, 0.0, 4.0, 6.0, 9.0, 9.0, 5.0, 4.0, 3.0, 1.0, 3.0, 9.0, 9.0, 2.0,
//         ],
//         vec![
//             9.0, 0.0, 7.0, 4.0, 3.0, 7.0, 6.0, 1.0, 0.0, 3.0, 7.0, 1.0, 2.0, 0.0, 0.0, 2.0, 4.0,
//         ],
//         vec![
//             2.0, 0.0, 0.0, 7.0, 9.0, 1.0, 2.0, 1.0, 2.0, 6.0, 0.0, 9.0, 7.0, 9.0, 9.0, 9.0, 1.0,
//         ],
//         vec![
//             2.0, 8.0, 6.0, 3.0, 9.0, 4.0, 1.0, 7.0, 3.0, 8.0, 4.0, 8.0, 3.0, 9.0, 4.0, 8.0, 7.0,
//         ],
//         vec![
//             2.0, 0.0, 2.0, 3.0, 1.0, 0.0, 6.0, 7.0, 6.0, 4.0, 0.0, 6.0, 6.0, 8.0, 2.0, 8.0, 0.0,
//         ],
//         vec![
//             0.0, 3.0, 8.0, 5.0, 2.0, 0.0, 3.0, 8.0, 2.0, 8.0, 6.0, 3.0, 2.0, 9.0, 4.0, 4.0, 2.0,
//         ],
//         vec![
//             8.0, 3.0, 4.0, 3.0, 4.0, 6.0, 8.0, 6.0, 4.0, 9.0, 9.0, 6.0, 9.0, 4.0, 2.0, 6.0, 1.0,
//         ],
//         vec![
//             8.0, 9.0, 9.0, 0.0, 5.0, 6.0, 7.0, 9.0, 8.0, 1.0, 9.0, 1.0, 4.0, 4.0, 5.0, 2.0, 7.0,
//         ],
//         vec![
//             0.0, 5.0, 3.0, 0.0, 6.0, 8.0, 3.0, 3.0, 5.0, 2.0, 5.0, 6.0, 9.0, 9.0, 2.0, 6.0, 2.0,
//         ],
//         vec![
//             1.0, 9.0, 3.0, 7.0, 8.0, 6.0, 0.0, 2.0, 8.0, 0.0, 8.0, 7.0, 0.0, 5.0, 4.0, 5.0, 9.0,
//         ],
//         vec![
//             4.0, 5.0, 4.0, 4.0, 3.0, 2.0, 2.0, 3.0, 8.0, 1.0, 8.0, 0.0, 0.0, 4.0, 5.0, 5.0, 2.0,
//         ],
//         vec![
//             6.0, 8.0, 9.0, 7.0, 5.0, 7.0, 4.0, 7.0, 9.0, 3.0, 9.0, 7.0, 9.0, 1.0, 4.0, 8.0, 3.0,
//         ],
//         vec![
//             5.0, 0.0, 8.0, 0.0, 4.0, 3.0, 2.0, 5.0, 1.0, 2.0, 4.0, 8.0, 1.0, 9.0, 7.0, 1.0, 4.0,
//         ],
//         vec![
//             6.0, 7.0, 0.0, 5.0, 0.0, 1.0, 0.0, 4.0, 9.0, 8.0, 5.0, 0.0, 0.0, 1.0, 8.0, 2.0, 0.0,
//         ],
//         vec![
//             4.0, 6.0, 5.0, 0.0, 4.0, 4.0, 5.0, 2.0, 4.0, 6.0, 4.0, 4.0, 4.0, 9.0, 9.0, 2.0, 0.0,
//         ],
//         vec![
//             4.0, 8.0, 0.0, 2.0, 3.0, 0.0, 0.0, 7.0, 1.0, 7.0, 6.0, 9.0, 9.0, 1.0, 5.0, 5.0, 2.0,
//         ],
//     ];

//     let Z: Vec<f32> = vec![
//         5828.0, 5589.0, 5940.0, 5524.0, 5584.0, 5518.0, 5278.0, 5559.0, 5662.0, 5239.0, 5492.0,
//         5496.0, 5533.0, 5497.0, 5665.0, 5656.0,
//     ];

//     let D_W: [f32; 289] = [
//         457628.0, 470051.0, 544435.0, 486607.0, 439072.0, 448728.0, 445622.0, 473486.0,
//         444122.0, 391674.0, 319477.0, 346131.0, 386283.0, 399896.0, 494353.0, 418066.0,
//         422179.0, 461522.0, 482502.0, 517802.0, 456132.0, 401716.0, 399779.0, 401173.0,
//         419723.0, 390356.0, 317090.0, 297867.0, 335017.0, 439144.0, 510474.0, 563319.0,
//         495460.0, 505599.0, 437303.0, 466869.0, 528284.0, 437920.0, 406245.0, 406620.0,
//         378363.0, 393174.0, 333868.0, 267321.0, 251396.0, 296998.0, 403823.0, 481649.0,
//         531513.0, 526338.0, 567632.0, 431805.0, 413563.0, 413582.0, 385529.0, 367043.0,
//         421716.0, 388633.0, 403703.0, 370927.0, 298754.0, 306128.0, 315694.0, 423407.0,
//         547576.0, 567863.0, 595664.0, 565302.0, 495970.0, 433656.0, 383330.0, 341493.0,
//         347173.0, 382666.0, 405939.0, 360194.0, 334627.0, 300051.0, 327247.0, 375748.0,
//         441140.0, 498261.0, 465158.0, 540700.0, 537621.0, 506176.0, 409340.0, 330616.0,
//         320357.0, 324181.0, 394202.0, 448959.0, 366909.0, 320049.0, 293833.0, 276613.0,
//         326540.0, 343343.0, 352537.0, 350297.0, 380121.0, 413449.0, 487312.0, 340052.0,
//         217541.0, 253069.0, 299422.0, 397589.0, 496735.0, 416770.0, 386045.0, 374989.0,
//         322625.0, 383267.0, 394003.0, 363162.0, 354509.0, 325825.0, 323482.0, 448290.0,
//         331639.0, 257911.0, 256519.0, 299750.0, 406504.0, 477510.0, 397989.0, 355083.0,
//         354022.0, 311217.0, 419619.0, 397736.0, 334684.0, 345969.0, 256432.0, 330204.0,
//         382609.0, 344987.0, 313124.0, 278314.0, 340307.0, 443054.0, 470290.0, 419607.0,
//         362329.0, 297406.0, 220967.0, 297934.0, 313725.0, 301849.0, 390729.0, 314820.0,
//         359950.0, 325254.0, 337180.0, 372845.0, 314068.0, 376214.0, 442363.0, 414493.0,
//         451474.0, 384731.0, 347698.0, 314282.0, 339807.0, 345511.0, 379130.0, 428316.0,
//         415955.0, 427072.0, 323624.0, 395469.0, 455448.0, 348332.0, 365545.0, 366998.0,
//         321684.0, 377002.0, 352809.0, 339082.0, 352617.0, 361409.0, 323681.0, 331897.0,
//         346666.0, 370225.0, 428389.0, 318847.0, 404452.0, 464038.0, 361068.0, 407686.0,
//         342047.0, 262524.0, 344114.0, 295235.0, 316737.0, 330881.0, 284788.0, 254827.0,
//         268070.0, 279886.0, 358987.0, 418848.0, 304124.0, 351497.0, 360870.0, 313020.0,
//         327978.0, 271462.0, 194928.0, 261830.0, 235599.0, 309647.0, 424990.0, 390880.0,
//         380313.0, 331062.0, 287024.0, 350237.0, 380277.0, 343704.0, 351636.0, 352184.0,
//         311344.0, 318405.0, 261135.0, 221140.0, 279793.0, 267763.0, 386213.0, 447186.0,
//         456932.0, 428445.0, 316780.0, 308232.0, 338907.0, 410981.0, 406868.0, 339788.0,
//         321765.0, 309945.0, 320567.0, 319938.0, 234686.0, 301316.0, 260075.0, 343653.0,
//         431708.0, 463750.0, 476536.0, 440787.0, 468498.0, 455629.0, 462705.0, 432393.0,
//         373407.0, 293776.0, 310005.0, 310775.0, 333196.0, 329639.0, 349758.0, 327982.0,
//         385801.0, 443875.0, 518095.0, 549644.0, 514228.0, 511404.0, 505861.0, 444550.0,
//         448279.0, 437844.0, 405910.0, 420443.0, 416644.0, 407402.0, 382163.0, 437657.0,
//         399217.0, 429596.0, 453205.0, 513844.0, 480449.0, 476044.0, 465500.0, 445689.0,
//         468719.0,
//     ];

//     let D_X: [f32; 400] = [
//         23312.0, 22356.0, 58728.0, 78942.0, 87136.0, 119817.0, 96986.0, 126890.0, 106981.0,
//         98998.0, 113092.0, 109377.0, 119820.0, 97496.0, 85436.0, 62175.0, 92628.0, 90633.0,
//         75556.0, 49716.0, 74788.0, 95685.0, 165400.0, 202174.0, 156163.0, 189836.0, 166798.0,
//         230501.0, 265229.0, 258037.0, 265012.0, 223748.0, 189081.0, 155080.0, 174067.0,
//         187223.0, 220151.0, 201805.0, 131334.0, 61079.0, 125356.0, 143255.0, 253274.0,
//         309220.0, 234096.0, 306626.0, 278462.0, 323858.0, 336154.0, 308552.0, 320260.0,
//         282467.0, 260446.0, 208498.0, 189091.0, 204364.0, 247889.0, 228412.0, 166807.0,
//         82678.0, 135002.0, 152627.0, 258684.0, 360427.0, 321641.0, 402755.0, 385986.0,
//         392398.0, 369598.0, 371806.0, 371761.0, 379771.0, 382434.0, 356535.0, 383326.0,
//         398495.0, 405848.0, 339864.0, 221943.0, 89656.0, 123579.0, 187594.0, 297349.0,
//         391722.0, 387424.0, 405623.0, 382576.0, 389257.0, 349991.0, 388944.0, 386088.0,
//         402741.0, 396711.0, 391376.0, 427921.0, 469582.0, 473381.0, 356613.0, 236275.0,
//         77523.0, 83945.0, 126837.0, 212440.0, 305053.0, 343298.0, 364374.0, 365074.0, 361837.0,
//         304745.0, 352435.0, 326683.0, 373426.0, 410025.0, 440378.0, 460459.0, 477776.0,
//         439990.0, 299103.0, 214538.0, 67033.0, 33558.0, 95288.0, 183313.0, 283969.0, 368258.0,
//         331452.0, 312117.0, 341595.0, 292670.0, 413760.0, 397881.0, 419331.0, 444077.0,
//         495898.0, 542471.0, 563477.0, 516533.0, 320355.0, 217429.0, 55176.0, 69014.0, 144684.0,
//         259637.0, 334419.0, 355959.0, 334381.0, 327019.0, 404116.0, 395620.0, 500817.0,
//         507408.0, 481483.0, 508897.0, 513363.0, 472859.0, 489541.0, 430856.0, 268012.0,
//         200747.0, 56234.0, 96534.0, 139655.0, 246421.0, 326538.0, 294392.0, 288920.0, 330747.0,
//         365091.0, 417598.0, 546191.0, 524931.0, 517789.0, 527632.0, 478074.0, 449631.0,
//         484594.0, 426135.0, 307254.0, 214084.0, 60743.0, 84384.0, 154431.0, 264387.0, 332884.0,
//         339984.0, 351248.0, 368353.0, 400562.0, 414201.0, 490082.0, 509928.0, 525265.0,
//         562818.0, 528121.0, 473148.0, 499334.0, 425304.0, 305004.0, 209701.0, 72328.0, 89726.0,
//         194533.0, 279242.0, 358620.0, 390723.0, 403160.0, 429309.0, 414449.0, 435555.0,
//         426710.0, 479298.0, 549466.0, 539835.0, 528411.0, 460980.0, 466614.0, 450984.0,
//         347522.0, 257991.0, 110458.0, 67627.0, 185125.0, 268204.0, 349883.0, 402068.0,
//         377229.0, 369423.0, 337332.0, 386254.0, 354933.0, 436235.0, 489089.0, 405944.0,
//         436450.0, 393272.0, 428119.0, 466078.0, 367073.0, 254981.0, 117319.0, 62966.0,
//         214012.0, 317555.0, 419886.0, 480028.0, 456137.0, 399484.0, 387963.0, 432052.0,
//         333551.0, 454614.0, 478334.0, 412415.0, 486495.0, 403796.0, 424065.0, 411747.0,
//         304495.0, 222537.0, 88466.0, 90825.0, 210285.0, 344940.0, 441571.0, 467384.0, 442477.0,
//         356604.0, 351375.0, 383315.0, 313937.0, 433130.0, 460326.0, 371640.0, 449064.0,
//         393973.0, 380439.0, 422949.0, 313818.0, 207319.0, 100669.0, 118992.0, 228303.0,
//         343924.0, 434569.0, 379079.0, 345402.0, 268975.0, 288683.0, 372207.0, 377260.0,
//         482477.0, 455654.0, 360373.0, 361783.0, 356153.0, 358597.0, 354915.0, 271571.0,
//         137773.0, 50036.0, 118324.0, 232961.0, 357247.0, 420753.0, 381620.0, 343947.0,
//         286106.0, 318973.0, 374030.0, 395636.0, 456419.0, 462084.0, 410197.0, 421725.0,
//         453250.0, 418183.0, 377859.0, 265979.0, 111873.0, 38952.0, 107285.0, 223109.0,
//         292826.0, 334752.0, 287053.0, 217807.0, 182908.0, 245892.0, 265675.0, 352507.0,
//         415078.0, 434235.0, 430844.0, 414789.0, 470823.0, 402690.0, 361492.0, 251282.0,
//         89926.0, 33672.0, 78182.0, 193385.0, 219437.0, 258015.0, 222421.0, 132052.0, 131963.0,
//         165033.0, 204221.0, 291341.0, 348613.0, 348118.0, 342620.0, 285781.0, 322793.0,
//         301413.0, 242047.0, 182247.0, 60655.0, 11118.0, 44780.0, 121438.0, 147187.0, 161343.0,
//         155825.0, 99101.0, 99781.0, 139957.0, 126410.0, 178113.0, 203695.0, 227066.0, 270249.0,
//         252286.0, 276995.0, 243609.0, 183358.0, 127652.0, 49776.0, 10992.0, 22132.0, 66252.0,
//         66636.0, 79010.0, 72841.0, 27821.0, 28307.0, 55699.0, 44012.0, 83883.0, 116934.0,
//         128090.0, 172852.0, 139927.0, 135051.0, 111719.0, 72532.0, 67599.0, 39610.0, 11312.0,
//     ];

//     let bias: Vec<Vec<f32>> = vec![];
//     let p_s = (0 as u16, 1 as u16);
//     let (mut data, l) =
//         layer_factory::<f32>(Layers::Conv2D, features, kernel, bias, Some(p_s), 0.0);
//     let data1 = (data.0.as_slice(), data.1.as_mut_slice(), data.2.as_slice());
//     let mut z = l.forward(data1, None);
//     for i in 0..Z.len() {
//         assert!(Z[i] == z[i]);
//     }
//     let data2 = (
//         data.0.as_mut_slice(),
//         data.1.as_mut_slice(),
//         data.2.as_slice(),
//     );
//     let updated = l.backward(data2, z.as_mut_slice(), 0.1, 0.0);
//     for i in 0..D_W.len() {
//         assert!(
//             (D_W[i] - updated.1[i]).abs() < f32::EPSILON,
//             "{} truth {} prediction {}",
//             i,
//             D_W[i],
//             updated.1[i]
//         );
//     }

//     for i in 0..D_X.len() {
//         assert!(
//             (D_X[i] - updated.0[i]).abs() < f32::EPSILON,
//             "{} truth {} prediction {}",
//             i,
//             D_X[i],
//             updated.0[i]
//         );
//     }
// }

// #[test]
// fn test_conv2d_linear_5_3() {
//     let conv_weights = vec![
//         vec![1.0, 0.5, 0.2],
//         vec![0.5, 1.0, 0.5],
//         vec![0.2, 0.5, 1.0],
//     ];

//     let linear_weights: Vec<Vec<f64>> = vec![
//         vec![0.1],
//         vec![0.2],
//         vec![0.3],
//         vec![0.4],
//         vec![0.5],
//         vec![0.6],
//         vec![0.7],
//         vec![0.8],
//         vec![0.9],
//     ];

//     let dummy_linear_inputs = vec![vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]];

//     let input = vec![
//         vec![2.0, 1.0, 3.0, 4.0, 1.0],
//         vec![1.0, 2.0, 1.0, 3.0, 2.0],
//         vec![3.0, 1.0, 4.0, 2.0, 3.0],
//         vec![2.0, 3.0, 1.0, 4.0, 2.0],
//         vec![1.0, 2.0, 3.0, 1.0, 4.0],
//     ];

//     let linear_bias = vec![vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]];

//     //As produced by Pytorch/Colab
//     let CONV_Z = [
//         11.2000,
//         11.0000,
//         14.5000,
//         9.6000,
//         13.7000,
//         12.600000000000001,
//         13.0000,
//         10.8000,
//         16.2000,
//     ];
//     let LINEAR_Z = [58.239999999999995];
//     let LOSS = [2834.4975999999992];
//     let D_Z_LINEAR = [
//         1192.5759999999998,
//         1171.28,
//         1543.9599999999998,
//         1022.2079999999999,
//         1458.7759999999998,
//         1341.648,
//         1384.2399999999998,
//         1149.984,
//         1724.9759999999997,
//     ];
//     let U_W_LINEAR = [
//         -11.825759999999999,
//         -11.5128,
//         -15.139599999999998,
//         -9.822079999999998,
//         -14.08776,
//         -12.81648,
//         -13.142399999999999,
//         -10.699839999999998,
//         -16.34976,
//     ];
//     let D_Z_CONV = [
//         10.648,
//         21.296,
//         31.943999999999996,
//         42.592,
//         53.239999999999995,
//         63.88799999999999,
//         74.53599999999999,
//         85.184,
//         95.832,
//     ];
//     let U_W_CONV = [
//         -9.435039999999999,
//         -10.893359999999998,
//         -12.151679999999997,
//         -9.722079999999998,
//         -11.13872,
//         -11.63872,
//         -10.022079999999999,
//         -10.46744,
//         -11.564639999999999,
//     ];

//     let bias: Vec<Vec<f64>> = vec![];
//     let p_s = (0 as u16, 1 as u16);
//     let (mut data_conv, l_conv) =
//         layer_factory::<f64>(Layers::Conv2D, input, conv_weights, bias, Some(p_s), 0.0);
//     let data1_conv = (
//         data_conv.0.as_slice(),
//         data_conv.1.as_mut_slice(),
//         data_conv.2.as_slice(),
//     );
//     let mut z_conv: Vec<f64> = l_conv.forward(data1_conv, None);
//     for i in 0..CONV_Z.len() {
//         assert!(
//             (CONV_Z[i] - z_conv[i]).abs() < f64::EPSILON,
//             "{} truth {} prediction {}",
//             i,
//             CONV_Z[i],
//             z_conv[i]
//         );
//     }

//     let (mut data_linear, linear_layer) = layer_factory::<f64>(
//         Layers::Linear,
//         dummy_linear_inputs,
//         linear_weights,
//         linear_bias,
//         None,
//         0.0,
//     );
//     let data1_linear = (
//         z_conv.as_slice(),
//         data_linear.1.as_mut_slice(),
//         data_linear.2.as_slice(),
//     );
//     let mut linear_z = linear_layer.forward(data1_linear, None);
//     let data1_linear = (
//         z_conv.as_mut_slice(),
//         data_linear.1.as_mut_slice(),
//         data_linear.2.as_slice(),
//     );

//     assert!((LINEAR_Z[0] - linear_z[0]).abs() < f64::EPSILON);
//     let (flattened, squared) =
//         loss_function_factory(LossFunctions::MeanSquares, vec![vec![5.0]], 1.0);
//     let mut loss = squared.forward(&flattened, &linear_z);
//     assert!((LOSS[0] - loss[0]).abs() < f64::EPSILON);
//     squared.backward(linear_z[0], data1_linear.0);
//     for i in 0..D_Z_LINEAR.len() {
//         assert!(
//             (D_Z_LINEAR[i] - data1_linear.0[i]).abs() < f64::EPSILON,
//             "{} truth {} prediction {}",
//             i,
//             D_Z_LINEAR[i],
//             data1_linear.0[i]
//         );
//     }

//     let data2_linear = (
//         data_linear.0.as_mut_slice(),
//         data1_linear.1,
//         data_linear.2.as_slice(),
//     );
//     let mut updated = linear_layer.backward(data2_linear, data1_linear.0, 0.01, linear_z[0]);
//     for i in 0..D_Z_CONV.len() {
//         assert!(
//             (D_Z_CONV[i] - updated.0[i]).abs() < f64::EPSILON,
//             "{} truth {} prediction {}",
//             i,
//             D_Z_CONV[i],
//             updated.0[i]
//         );
//     }
//     for i in 0..U_W_LINEAR.len() {
//         assert!(
//             (U_W_LINEAR[i] - updated.1[i]).abs() < f64::EPSILON,
//             "{} truth {} prediction {}",
//             i,
//             U_W_LINEAR[i],
//             updated.1[i]
//         );
//     }
//     let data2_conv = (
//         data_conv.0.as_mut_slice(),
//         data_conv.1.as_mut_slice(),
//         data_conv.2.as_slice(),
//     );
//     let updated = l_conv.backward(data2_conv, updated.0.as_mut_slice(), 0.01, 0.0);
//     for i in 0..U_W_CONV.len() {
//         assert!(
//             (U_W_CONV[i] - updated.1[i]).abs() < f64::EPSILON,
//             "{} truth {} prediction {}",
//             i,
//             U_W_CONV[i],
//             updated.1[i]
//         );
//     }
// }
