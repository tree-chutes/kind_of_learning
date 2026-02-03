use csv::{ReaderBuilder, StringRecord};
use futures::TryFutureExt;
use rustls::{AlertDescription, ClientConfig, ClientConnection, Error, RootCertStore};
use rustls_pki_types::ServerName;
use std::io::{self, BufReader, Cursor, Read, Write};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use std::time::Duration;
use tokio::io::Interest;
use tokio::net::tcp::{OwnedReadHalf, OwnedWriteHalf};
use tokio::net::TcpStream;
use tokio::task::JoinHandle;
use rand::prelude::*;
use tokio::io::AsyncReadExt;
use tokio::io::AsyncWriteExt;
use uuid::Uuid;

#[derive(PartialEq)]
#[repr(u8)]
pub enum TcpSteps {
    ACK = 0,
    AUTH = 1,
    QUERY = 2,
    REQUEST = 3,
}

impl TcpSteps {
    pub fn from_u8(value: u8) -> Self {
        match value {
            0 => TcpSteps::ACK,
            1 => TcpSteps::AUTH,
            2 => TcpSteps::QUERY,
            3 => TcpSteps::REQUEST,
            _ => todo!(),
        }
    }

    pub fn to_u8(value: TcpSteps) -> u8 {
        match value {
            TcpSteps::ACK => 0 as u8,
            TcpSteps::AUTH => 1 as u8,
            TcpSteps::QUERY => 2 as u8,
            TcpSteps::REQUEST => 3 as u8,
        }
    }
}

const HEADER_OFFSET: usize = 1 + 8; // 1 Op + 8 len of full request
const AUTH_OFFSET: usize = HEADER_OFFSET + 8; //8 for usize the default len in rust
const MAX_TLS_RECORD_SIZE: usize = 1024 * 16;
const TLS_RECORD_OFFSETS: [usize; 2] = [3, 4];
const TLS_RECORD_HEADER_LEN: usize = 5;
const PAGE_SIZE: usize = 4096;
const MAX_SEGMENT_SIZE: usize = 1460;
const TLS_OVERHEAD: usize = TLS_RECORD_HEADER_LEN + 17; //This is only for TLS 1.3
const OP_LEN: usize = 1;
const REQUEST_LEN: usize = 8;
const AUTH_LEN: usize = 16;
const ID_LEN: usize = 16;
const TARGET_LEN: usize = 1;

#[tokio::main(flavor = "multi_thread", worker_threads = 2)]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // env_logger::init();
    // 1. Setup Root Certificates (using Mozilla's trust store)
    let mut root_cert_store = RootCertStore::empty();

    let ca_cert = "-----BEGIN CERTIFICATE-----\nMIIBYTCCAQigAwIBAgIUeYKetGqTHO1HemaWdwxquRwnZfAwCgYIKoZIzj0EAwIw\nFjEUMBIGA1UEAwwLQ081IFJ1c3QgQ0EwIBcNNzUwMTAxMDAwMDAwWhgPNDA5NjAx\nMDEwMDAwMDBaMBYxFDASBgNVBAMMC0NPNSBSdXN0IENBMFkwEwYHKoZIzj0CAQYI\nKoZIzj0DAQcDQgAE9NgtbFrarZbGZ3dYHuYej9ahCmiu3dxATc9d2ZjEK9YGaGUj\nL0RzLd1SG8zCXy/Qmb4cISvsx7V85/WO+bxXW6MyMDAwHQYDVR0OBBYEFFmGMQIT\nwDQ5xusEc34yAGka9W+eMA8GA1UdEwEB/wQFMAMBAf8wCgYIKoZIzj0EAwIDRwAw\nRAIgUssr3KWuywCrm7ozjntjXR0NIH5CeasdOEkmO9xlFicCIB9Clk5ogjFBxr+l\ntFbe/aWzQaSSUGsawbdeJ7ZeN8Un\n-----END CERTIFICATE-----\n";
    let cursor = Cursor::new(ca_cert.as_bytes());
    let mut reader = BufReader::new(cursor);
    for cert in rustls_pemfile::certs(&mut reader) {
        root_cert_store.add(cert?)?; // Add each certificate to the store
    }

    // 2. Build Client Configuration (using aws-lc-rs as the default provider)

    let mut outbounds: Vec<JoinHandle<()>> = Vec::new();
    let mut inbounds: Vec<JoinHandle<()>> = Vec::new();
    let addr = "localhost:8443";
    let auth_ticket = Arc::new(RwLock::new(Vec::<u8>::new()));

    for i in 0..1 {
        // 1. Setup Root Certificates (using Mozilla's trust store)
        let mut root_cert_store = RootCertStore::empty();

        let ca_cert = "-----BEGIN CERTIFICATE-----\nMIIBYTCCAQigAwIBAgIUeYKetGqTHO1HemaWdwxquRwnZfAwCgYIKoZIzj0EAwIw\nFjEUMBIGA1UEAwwLQ081IFJ1c3QgQ0EwIBcNNzUwMTAxMDAwMDAwWhgPNDA5NjAx\nMDEwMDAwMDBaMBYxFDASBgNVBAMMC0NPNSBSdXN0IENBMFkwEwYHKoZIzj0CAQYI\nKoZIzj0DAQcDQgAE9NgtbFrarZbGZ3dYHuYej9ahCmiu3dxATc9d2ZjEK9YGaGUj\nL0RzLd1SG8zCXy/Qmb4cISvsx7V85/WO+bxXW6MyMDAwHQYDVR0OBBYEFFmGMQIT\nwDQ5xusEc34yAGka9W+eMA8GA1UdEwEB/wQFMAMBAf8wCgYIKoZIzj0EAwIDRwAw\nRAIgUssr3KWuywCrm7ozjntjXR0NIH5CeasdOEkmO9xlFicCIB9Clk5ogjFBxr+l\ntFbe/aWzQaSSUGsawbdeJ7ZeN8Un\n-----END CERTIFICATE-----\n";
        let cursor = Cursor::new(ca_cert.as_bytes());
        let mut reader = BufReader::new(cursor);
        for cert in rustls_pemfile::certs(&mut reader) {
            root_cert_store.add(cert?)?; // Add each certificate to the store
        }
        let mut config = ClientConfig::builder()
            .with_root_certificates(root_cert_store)
            .with_no_client_auth();

        let domain = ServerName::try_from("localhost")?;
        let mut tcp_stream = TcpStream::connect(addr).await?;
        tcp_stream.set_nodelay(true)?;
        let mut client_connection = ClientConnection::new(Arc::new(config), domain)
            .map_err(|e| tokio::io::Error::new(tokio::io::ErrorKind::Other, e))?;
        let _ = finish_handshake(&mut tcp_stream, &mut client_connection).await;
        let client_connection: Arc<Mutex<ClientConnection>> =
            Arc::new(Mutex::new(client_connection));
        let (socket_in, socket_out) = tcp_stream.into_split();
        let s_o = Arc::new(Mutex::new(socket_out));
        let received = Arc::new(AtomicBool::new(false));
        let done = Arc::new(AtomicBool::new(false));
        let mut a_t = auth_ticket.clone();
        let mut r = received.clone();
        let d = done.clone();
        let load = Arc::new(AtomicUsize::new(0));
        let load_1 = load.clone();
        let mut rdr = ReaderBuilder::new().from_path("mnist_train.csv").unwrap();
        let records = rdr
            .records()
            .collect::<Result<Vec<StringRecord>, csv::Error>>()
            .unwrap();
        let mut mnist_train: Vec<Vec<f32>> = vec![];
        records.iter().for_each(|r| {
            let mut record: Vec<f32> = vec![];
            r.iter().for_each(|item| {
                record.push(item.parse::<f32>().unwrap());
            });
            mnist_train.push(record);
        });
        let mut count = mnist_train.len();
        let mut s_o_0 = s_o.clone();
        let c_c = client_connection.clone();
        outbounds.push(tokio::spawn(async move {
            thread::sleep(Duration::from_micros(500));
            let auth_ticket = a_t;
            let flag = d;
            let received = r;
            let load = load_1;
            let mut req = vec![1u8; 1];
            let socket_out = s_o_0;
            let client_connection = c_c;
            req.extend_from_slice(&mut (1 + 8 + 8 + "valid".len()).to_be_bytes());
            req.extend_from_slice(&"valid".len().to_be_bytes());
            req.append(&mut "valid".as_bytes().to_vec());

            write_to_socket(&req, &socket_out, &client_connection);
            while !received.load(Ordering::SeqCst) {
                thread::sleep(Duration::from_micros(500));
            }

            let conv0_weights = random_weights(12 * 12);
            let conv1_weights = random_weights(12 * 12);
            let linear_weights = random_weights(36);
            flag.store(false, Ordering::SeqCst);
            while !flag.load(Ordering::SeqCst) {
                if load.load(Ordering::SeqCst) == 0{
                    received.store(false, Ordering::SeqCst);
                    req.clear();
                    req.push(TcpSteps::to_u8(TcpSteps::QUERY));
                    match auth_ticket.clone().read() {
                        Ok(a_t) => {
                            req.extend_from_slice(&mut usize::to_be_bytes(
                                (1 + size_of::<usize>() + a_t.len()) as usize,
                            ));
                            req.extend_from_slice(&a_t);
                        }
                        Err(_) => todo!(),
                    }
                    write_to_socket(&req, &socket_out, &client_connection);
                    while !received.load(Ordering::SeqCst){
                        thread::sleep(Duration::from_micros(500));
                    }
                } else {
                    received.store(false, Ordering::SeqCst);
                    let mut input = mnist_train.pop().expect("done");
                    input.remove(0);
                    let mut payload = Vec::<u8>::new();
                    payload.push(TcpSteps::to_u8(TcpSteps::REQUEST));
                    payload.extend_from_slice(
                        &(OP_LEN
                            + REQUEST_LEN
                            + AUTH_LEN
                            + ID_LEN
                            + size_of::<usize>() * 4
                            + size_of::<f32>()
                                * (TARGET_LEN
                                    + input.len()
                                    + conv0_weights.len()
                                    + conv1_weights.len()
                                    + linear_weights.len()))
                        .to_be_bytes(),
                    );
                    payload.extend_from_slice(&auth_ticket.read().expect("TODO"));
                    payload.extend_from_slice(Uuid::new_v4().as_bytes());
                    payload.extend_from_slice(&input.len().to_be_bytes());
                    payload.extend_from_slice(&conv0_weights.len().to_be_bytes());
                    payload.extend_from_slice(&conv1_weights.len().to_be_bytes());
                    payload.extend_from_slice(&linear_weights.len().to_be_bytes());
                    input
                        .iter()
                        .for_each(|f| payload.extend_from_slice(&(*f as f32).to_be_bytes()));
                    conv0_weights
                        .iter()
                        .for_each(|f| payload.extend_from_slice(&(*f).to_be_bytes()));
                    conv1_weights
                        .iter()
                        .for_each(|f| payload.extend_from_slice(&(*f).to_be_bytes()));
                    linear_weights
                        .iter()
                        .for_each(|f| payload.extend_from_slice(&(*f).to_be_bytes()));
                    payload.extend_from_slice(&(0.5 as f32).to_be_bytes());
                    write_to_socket(&payload, &socket_out, &client_connection);
                    while !received.load(Ordering::SeqCst){
                        thread::sleep(Duration::from_micros(500));
                    }
                }
                thread::sleep(Duration::from_micros(500));
            }
            println!("complete");
        }));
        a_t = auth_ticket.clone();
        
        inbounds.push(tokio::spawn(async move {
            thread::sleep(Duration::from_micros(500));
            let load = load;
            let auth_ticket = a_t;
            let received = received;
            let done = done;
            let mut counter: usize = count;
            let mut socket_in = socket_in;
            let client_connection = client_connection;
            let mut message = vec![];
            let mut encrypted_buffer = vec![];
            let mut current_len = 0usize;
            loop {
                match read_from_socket(
                    &mut socket_in,
                    &client_connection,
                    &mut encrypted_buffer,
                    &mut message,
                )
                .await
                {
                    Ok(mut complete) => {
                        while complete > 0{
                            current_len =  usize::from_be_bytes(message[1..9].try_into().expect("TODO"));
                            match TcpSteps::from_u8(message[0]) {
                                TcpSteps::AUTH => {
                                    assert!(message.len() > 24, "Expected 25 bytes");
                                    match auth_ticket.write() {
                                        Ok(mut r) => {
                                            _ = r.extend_from_slice(&message[9..]);
                                            received.store(true, Ordering::SeqCst);
                                        }
                                        Err(_) => todo!(),
                                    }
                                }
                                TcpSteps::QUERY => {
                                    assert!(message.len() > 16, "Expected 17 bytes");                                                                                
                                    load.store(usize::from_be_bytes(
                                        message[9..17].try_into().expect("TODO"),
                                    ), Ordering::SeqCst);
                                    received.store(true, Ordering::SeqCst);
                                }
                                TcpSteps::REQUEST => {
                                    counter -= 1;
                                    if counter == 0{
                                        done.store(true, Ordering::SeqCst);
                                    }
                                    
                                }
                                TcpSteps::ACK => {
                                    assert!(message.len() > 16, "Expected 17 bytes");                                                                                
                                    load.fetch_sub(1, Ordering::SeqCst);
                                    received.store(true, Ordering::SeqCst);
                                }
                            }
                            message.drain(..current_len);
                            complete -= 1;
                        }
                        thread::sleep(Duration::from_micros(100));
                    }
                    Err(e) => {
                        panic!("{}", e)
                    }
                }
            }
        }));
    }
    for jh in outbounds {
        while !jh.is_finished() {
            thread::sleep(Duration::from_micros(500));
        }
    }
    for jh in inbounds {
        while !jh.is_finished() {
            thread::sleep(Duration::from_micros(500));
        }
    }
    Ok(())
}

async fn finish_handshake(
    socket: &mut TcpStream,
    client_connection: &mut ClientConnection,
) -> Result<(), Error> {
    let mut buffer = vec![];
    while client_connection.is_handshaking() {
        while client_connection.wants_write() {
            socket.ready(Interest::WRITABLE).await.expect("todo");
            _ = client_connection.write_tls(&mut buffer);
            socket.write_all(&buffer).await.expect("todo");
            buffer.clear();
        }
        buffer.resize(4096, 0u8);
        while client_connection.wants_read() {
            socket.ready(Interest::READABLE).await.expect("todo");
            match socket.try_read(&mut buffer) {
                Ok(0) => return Err(Error::HandshakeNotComplete),
                Ok(n) => {
                    client_connection.read_tls(&mut &buffer[..n]);
                    client_connection.process_new_packets();
                }
                Err(e) => return Err(Error::General(format!("{}", e))),
            }
        }
    }
    Ok(())
}

fn write_to_socket(
    buffer: &[u8],
    socket_out: &Arc<Mutex<OwnedWriteHalf>>,
    client_connection: &Arc<Mutex<ClientConnection>>,
) -> bool {
    let mut send_buffer = Vec::new();
    let t = buffer.len();
    for chunk in buffer.chunks(MAX_SEGMENT_SIZE) {
        match client_connection.lock() {
            Ok(mut s_c) => match s_c.writer().write_all(chunk) {
                Ok(_) => match s_c.write_tls(&mut send_buffer) {
                    Ok(count) => match socket_out.lock() {
                        Ok(s_o) => match s_o.try_write(&send_buffer) {
                            Ok(c) => {
                                if c != count {
                                    todo!("Sent less than was written?");
                                }
                                send_buffer.clear();
                            }
                            Err(_) => todo!(),
                        },
                        Err(_) => todo!(),
                    },
                    Err(_) => todo!(),
                },
                Err(_) => todo!(),
            },
            Err(_) => todo!(),
        }
        thread::sleep(Duration::from_micros(50));
    }
    return true;
}

pub async fn read_from_socket(
    socket_in: &mut OwnedReadHalf,
    client_connection: &Arc<Mutex<ClientConnection>>,
    encrypted_buffer: &mut Vec<u8>,
    decrypted_buffer: &mut Vec<u8>,
) -> Result<u8, Error> {
    let mut buffer = [0u8; MAX_TLS_RECORD_SIZE];
    let mut record_len: usize = 0;
    let mut request_len: usize = 0;
    let mut ret = 0u8;
    let mut current_len = 0usize;

    _ = socket_in.ready(Interest::READABLE).await;

    loop {
        _ = match socket_in.try_read(&mut buffer) {
            Ok(0) => return Err(Error::AlertReceived(AlertDescription::CloseNotify)),
            Ok(n) => {
                encrypted_buffer.extend_from_slice(&buffer[..n]);
            }
            Err(ref e) if e.kind() == io::ErrorKind::WouldBlock => {
                break;
            }
            Err(e) => return Err(rustls::Error::General(format!("{}", e))),
        };
        thread::sleep(Duration::from_micros(10));
    }
    while encrypted_buffer.len() > 0 {
        record_len = ((encrypted_buffer[TLS_RECORD_OFFSETS[0]] as u16) << u8::BITS
            | encrypted_buffer[TLS_RECORD_OFFSETS[1]] as u16) as usize;

        match client_connection.lock() {
            Ok(mut c_c) => {
                record_len = c_c
                    .read_tls(&mut &encrypted_buffer[..TLS_RECORD_HEADER_LEN + record_len])
                    .expect("TODO");
                match c_c.process_new_packets() {
                    Ok(state) => {
                        if state.plaintext_bytes_to_read() != 0 {
                            match c_c.reader().read(&mut buffer[current_len..]) {
                                Ok(count) => {
                                    if request_len == 0 {                                        
                                        if decrypted_buffer.len() == 0 || ret > 0{
                                            request_len = usize::from_be_bytes(buffer[1..9].try_into().expect("TODO"));
                                        }
                                        else{
                                            request_len = usize::from_be_bytes(decrypted_buffer[1..9].try_into().expect("TODO"));
                                            current_len = decrypted_buffer.len();
                                        }                                            
                                    }
                                    decrypted_buffer.extend_from_slice(&buffer[current_len..current_len + count]);                                    
                                    current_len += count;

                                    if current_len == request_len {
                                        ret += 1;
                                        request_len = 0;
                                        current_len = 0;
                                    }
                                },
                                Err(e) => return Err(rustls::Error::General(format!("{}", e)))
                            }
                        }
                        encrypted_buffer.drain(0..record_len);
                    }
                    Err(e) => {
                        println!("{}", e);
                    }
                }
            }
            Err(e) => return Err(rustls::Error::General(format!("{}", e))),
        }
    }
    return Ok(ret);
}

fn random_weights(len: u32) -> Vec<f32> {
    let mut rng = rand::rng();
    let mut nums: Vec<u32> = (0..len).collect();
    nums.shuffle(&mut rng);
    let nt: Vec<f32> = nums.iter().map(|t| rng.random_range(0.0..1.0)).collect();
    return nt;
}

