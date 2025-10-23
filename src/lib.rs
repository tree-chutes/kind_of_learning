// Copyright (c) 2025, tree-chutes

#![feature(downcast_unchecked)]
#![allow(non_snake_case)]
mod mlp;

use std::sync::Arc;

use co5_backflow_public::{
    errors::{authorize::CO5AuthorizationError, workunit::CO5BuildWorkUnitError},
    stage_descriptor::StageDescriptor,
    workunit::WorkUnit,
};
use mlp::{
    layers::{Layer, Layers, layer_factory},
    loss_functions::{LossFunctions, loss_function_factory},
    register::REGISTER_WIDTH,
};
use num_traits::Float;
use uuid::Uuid;

struct Vectors<F: Float> {
    input: Vec<F>,
    conv_0_weights: Vec<F>,
    conv_1_weights: Vec<F>,
    linear_weights: Vec<F>,
    target: Vec<F>,
}
unsafe impl<F: Float> Send for Vectors<F> {}

#[unsafe(no_mangle)]
pub fn work_unit_factory(s: bool, i: &Uuid) -> Result<WorkUnit, CO5BuildWorkUnitError> {
    let ret = WorkUnit::new(
        s,
        i,
        80,
        Box::new(Vectors::<f64> {
            input: vec![],
            conv_0_weights: vec![],
            conv_1_weights: vec![],
            linear_weights: vec![],
            target: vec![],
        }),
    );
    Ok(ret)
}

#[unsafe(no_mangle)]
pub fn authorize(_t: &str) -> Result<bool, CO5AuthorizationError> {
    return Ok(true);
}

#[unsafe(no_mangle)]
pub fn nn_workflow() -> Vec<StageDescriptor> {
    let mut s = Vec::<StageDescriptor>::new();
    let s1: StageDescriptor = StageDescriptor {
        full_time_agents: 10,
        part_time_agents: 5,
        task: Some(full_pass),
        ctask: None,
        c_free_memory: None,
        agents_pause: 10,
        master_pause: 25,
        work_in_progress: 5,
        label: "FULL PASS".to_string(),
        logging: true,
    };
    s.push(s1);
    return s;
}

fn full_pass(mut wu: WorkUnit) -> WorkUnit {
    let input_data_len: usize;
    let conv_0_weights_len: usize;
    let conv_1_weights_len: usize;
    let linear_weights_len: usize;
    let bias: Vec<f64> = vec![];
    let p_s = (0 as u16, 1 as u16);
    let vectors: &mut Vectors<f64> = wu.custom_struct.downcast_mut().expect("msg");
    let mut layers: Vec<(Box<dyn Layer<f64>>, &mut [f64], &[f64])> = vec![];
    let ret: Arc<Vec<u8>>;

    unsafe {
        let payload = wu.payload.unwrap_unchecked();
        (
            input_data_len,
            conv_0_weights_len,
            conv_1_weights_len,
            linear_weights_len,
        ) = read_payload(payload.as_slice(), vectors);
    }
    wu.payload = None;
    let mut conv_layer_0 = layer_factory::<f64>(
        Layers::Conv2D,
        input_data_len,
        conv_0_weights_len,
        input_data_len,
        Some(p_s),
        0.0,
    );

    conv_layer_0.set_first_layer_flag(); //this skips the second convolution if it isnot needed
    layers.push((conv_layer_0, vectors.conv_0_weights.as_mut_slice(), &bias));

    let conv_layer_1 = layer_factory::<f64>(
        Layers::Conv2D,
        conv_0_weights_len,
        conv_1_weights_len,
        conv_0_weights_len,
        Some(p_s),
        0.0,
    );

    layers.push((conv_layer_1, vectors.conv_1_weights.as_mut_slice(), &bias));

    let linear_layer =
        layer_factory::<f64>(Layers::Linear, 1, linear_weights_len, 1, Some(p_s), 0.0);

    layers.push((linear_layer, vectors.linear_weights.as_mut_slice(), &bias));

    //FORWARD PASS
    let mut z: Vec<Vec<f64>> = network_forward(layers.as_mut_slice(), &vectors.input);

    //LOSS STEP
    let count = z.len();
    let (flat_loss, squared) =
        loss_function_factory(LossFunctions::MeanSquares, vec![vec![2.0]], 1.0);
    let _loss = squared.forward(&flat_loss, &z[z.len() - 1]);
    let from_loss_to_linear_grads =
        squared.backward(z[count - 1][0], z[count - 2].as_mut_slice());
    let d_loss = z.pop().unwrap()[0];

    //Backward pass can be executed in another endpoint since it does require more
    //computations. Layers are stateless and lightweight. Just load the bytes in the payload 
    //return the data or daisy chain service calls
    let success = network_backward(
        layers,
        from_loss_to_linear_grads,
        z,
        0.01,
        d_loss

    );
    wu.done = true;
    wu
}

fn network_forward(
    layers: &mut [(Box<dyn Layer<f64>>, &mut [f64], &[f64])],
    input_layer: &[f64],
) -> Vec<Vec<f64>> {
    let mut counter = 0;
    let mut ret: Vec<Vec<f64>> = vec![];
    ret.push(input_layer.to_vec());
    let mut current_input = input_layer;
    
    loop {
        ret.push(
            layers[counter]
                .0
                .forward((current_input, layers[counter].1, layers[counter].2), None),
        );
        current_input = &ret[ret.len() - 1];
        counter += 1;

        if counter == layers.len() {
            break;
        }
    }
    ret
}

fn network_backward(
    mut layers: Vec<(Box<dyn Layer<f64>>, &mut [f64], &[f64])>,
    loss_gradient: Vec<f64>,
    mut z: Vec<Vec<f64>>,
    learning_rate: f64,
    d_loss: f64
) -> Vec<Vec<f64>> {
    let mut idx = layers.len();
    let mut updated_weights: Vec<Vec<f64>> = vec![vec![]; layers.len()];
    //It is a regression, linear layer
    let mut current_gradient = loss_gradient;
    let (layer, weights, _bias) = layers.pop().unwrap();
    let mut z_previous = z.pop().unwrap();
    (current_gradient, _) = layer.backward((current_gradient.as_mut_slice(), weights, z_previous.as_mut_slice()), learning_rate, d_loss);
    idx -= 1;
    updated_weights[idx] = weights.to_vec();
    while layers.len() > 0 {
        let (layer, weights, _bias) = layers.pop().unwrap();
        z_previous = z.pop().unwrap();
        let (updated, back) = layer.backward((z_previous.as_mut_slice(), current_gradient.as_mut_slice(), weights), learning_rate, d_loss);
        idx -= 1;
        updated_weights[idx] = updated;
        current_gradient = back;
    }
    updated_weights
}

fn read_payload(payload: &[u8], vectors: &mut Vectors<f64>) -> (usize, usize, usize, usize) {
    let mut tmp: f64;
    let mut counter: usize = 0;
    let input_data_len = usize::from_be_bytes(payload[0..size_of::<usize>()].try_into().unwrap());
    let conv_0_weights_len = usize::from_be_bytes(
        payload[size_of::<usize>()..size_of::<usize>() * 2]
            .try_into()
            .unwrap(),
    );
    let conv_1_weights_len = usize::from_be_bytes(
        payload[size_of::<usize>() * 2..size_of::<usize>() * 3]
            .try_into()
            .unwrap(),
    );

    let linear_weights_len = usize::from_be_bytes(
        payload[size_of::<usize>() * 3..size_of::<usize>() * 4]
            .try_into()
            .unwrap(),
    );

    let required_buffer =
        (input_data_len + conv_0_weights_len + conv_1_weights_len + linear_weights_len + 1)
            * size_of::<f64>();
    let offset = size_of::<usize>() * 4;
    assert!(
        payload.len() - size_of::<usize>() * 4 == required_buffer,
        "payload size = {}, required_buffer = {}",
        payload.len() - size_of::<usize>() * 4,
        required_buffer
    );

    vectors.input = Vec::<f64>::with_capacity(input_data_len);
    vectors.conv_0_weights = Vec::<f64>::with_capacity(conv_0_weights_len);
    vectors.conv_1_weights = Vec::<f64>::with_capacity(conv_1_weights_len);
    vectors.linear_weights = Vec::<f64>::with_capacity(linear_weights_len);
    vectors.target = vec![0.0; 1];

    loop {
        tmp = f64::from_be_bytes(
            payload[offset + counter * size_of::<f64>()..offset + size_of::<f64>() * (counter + 1)]
                .try_into()
                .unwrap(),
        );
        if counter < input_data_len {
            vectors.input.push(tmp);
        } else if counter - input_data_len < conv_0_weights_len {
            vectors.conv_0_weights.push(tmp);
        } else if counter - (input_data_len + conv_0_weights_len) < conv_1_weights_len {
            vectors.conv_1_weights.push(tmp);
        } else if counter - (input_data_len + conv_0_weights_len + conv_1_weights_len)
            < linear_weights_len
        {
            vectors.linear_weights.push(tmp);
        } else {
            vectors.target[0] = tmp;
            break;
        }
        counter += 1;
    }
    let ret = (
        input_data_len.isqrt(),
        conv_0_weights_len.isqrt(),
        conv_1_weights_len.isqrt(),
        linear_weights_len,
    );

    vectors.input.resize(
        vectors.input.len() + (REGISTER_WIDTH / (size_of::<f64>() * 8))
            - vectors.input.len() % (REGISTER_WIDTH / (size_of::<f64>() * 8)),
        0.0,
    );
    vectors.conv_0_weights.resize(
        vectors.conv_0_weights.len() + (REGISTER_WIDTH / (size_of::<f64>() * 8))
            - vectors.conv_0_weights.len() % (REGISTER_WIDTH / (size_of::<f64>() * 8)),
        0.0,
    );
    vectors.conv_1_weights.resize(
        vectors.conv_1_weights.len() + (REGISTER_WIDTH / (size_of::<f64>() * 8))
            - vectors.conv_1_weights.len() % (REGISTER_WIDTH / (size_of::<f64>() * 8)),
        0.0,
    );
    vectors.linear_weights.resize(
        vectors.linear_weights.len() + (REGISTER_WIDTH / (size_of::<f64>() * 8))
            - vectors.linear_weights.len() % (REGISTER_WIDTH / (size_of::<f64>() * 8)),
        0.0,
    );
    ret
}

fn write_payload(
    input_layer: &[f64],
    conv_0_weights: &[f64],
    conv_1_weights: &[f64],
    linear_weights: &[f64],
    target: f64,
) -> Vec<u8> {
    let mut ret: Vec<u8> = vec![];
    ret.append(&mut input_layer.len().to_be_bytes().to_vec());
    ret.append(&mut conv_0_weights.len().to_be_bytes().to_vec());
    ret.append(&mut conv_1_weights.len().to_be_bytes().to_vec());
    ret.append(&mut linear_weights.len().to_be_bytes().to_vec());
    for i in 0..input_layer.len() {
        ret.append(&mut f64::to_be_bytes(input_layer[i]).to_vec());
    }
    for i in 0..conv_0_weights.len() {
        ret.append(&mut f64::to_be_bytes(conv_0_weights[i]).to_vec());
    }
    for i in 0..conv_1_weights.len() {
        ret.append(&mut f64::to_be_bytes(conv_1_weights[i]).to_vec());
    }
    for i in 0..linear_weights.len() {
        ret.append(&mut f64::to_be_bytes(linear_weights[i]).to_vec());
    }
    ret.append(&mut f64::to_be_bytes(target).to_vec());
    ret
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::{full_pass, work_unit_factory, write_payload};

    #[test]
    fn test_full_pass() {
        let id = uuid::Uuid::max();
        let mut wu = work_unit_factory(false, &id).unwrap();
        let input_layer = vec![
            1.0, 0.5, 1.2, 0.8, 1.5, 0.9, 1.3, 0.7, 1.1, 0.6, 1.4, 0.8, 1.7, 1.0, 1.6, 0.9, 1.2,
            0.5, 1.3, 0.7, 1.8, 1.1, 1.4, 0.6, 1.9, 1.0, 1.5, 0.9, 1.2, 0.8, 1.6, 1.3, 1.1, 0.7,
            1.4, 0.9, 1.1, 0.8, 1.5, 1.0, 1.7, 1.2, 1.4, 0.6, 1.3, 0.7, 1.3, 0.9, 1.4, 1.1, 1.8,
            1.0, 1.5, 0.8, 1.2, 0.6, 1.4, 0.9, 1.3, 1.0, 1.6, 1.1, 1.7, 0.8, 1.1, 0.7, 1.5, 1.2,
            1.4, 0.9, 1.3, 1.0, 1.0, 1.5, 0.8, 1.2, 0.9, 1.3, 1.1, 0.7, 1.4,
        ];

        let conv_0_weights = vec![
            0.1, 0.2, 0.3, 0.2, 0.1, 0.2, 0.4, 0.6, 0.4, 0.2, 0.3, 0.6, 0.9, 0.6, 0.3, 0.2, 0.4,
            0.6, 0.4, 0.2, 0.1, 0.2, 0.3, 0.2, 0.1,
        ];

        let conv_1_weights = vec![1.0, 0.5, 0.2, 0.5, 1.0, 0.5, 0.2, 0.5, 1.0];

        let linear_weights = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];

        wu.payload = Some(Arc::new(write_payload(
            &input_layer,
            &conv_0_weights,
            &conv_1_weights,
            &linear_weights,
            2.0,
        )));
        wu = full_pass(wu);
    }
}
