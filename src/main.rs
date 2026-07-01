use std::path::Path;

use rand::RngExt;

#[derive(Debug)]
struct Matrix {
    rows: u32,
    cols: u32,
    data: Vec<f64>,
}

impl<'a> Matrix {
    pub fn full(rows: u32, cols: u32, v: f64) -> Self {
        let mut data = Vec::with_capacity((rows * cols) as usize);
        for _ in 0..=rows * cols {
            data.push(v);
        }
        Self {
            rows: rows,
            cols: cols,
            data,
        }
    }
    pub fn rand(rows: u32, cols: u32) -> Self {
        let mut data = Vec::with_capacity((rows * cols) as usize);
        let mut rng = rand::rng();
        for _ in 0..=rows * cols {
            data.push(rng.random());
        }
        Self {
            rows: rows,
            cols: cols,
            data,
        }
    }
    pub fn new(rows: u32, cols: u32) -> Self {
        Self::rand(rows, cols)
    }
    pub fn index(&'a self, row: u32, col: u32) -> usize {
        ((self.cols * row) + col) as usize
    }
    pub fn col(&'a self, col: u32) -> &'a [f64] {
        &self.data[self.index(0, col)..self.index(self.rows, col)]
    }
    pub fn op(&mut self, func: fn(f64) -> f64) {
        for i in self.data.iter_mut() {
            *i = func(*i);
        }
    }
    pub fn opa(&mut self, func: fn(f64, f64) -> f64, x: f64) {
        for i in self.data.iter_mut() {
            *i = func(*i, x);
        }
    }
    pub fn falten(&mut self) {
        self.cols = self.rows * self.cols;
        self.rows = 0;
    }
}

impl std::ops::Add<&Matrix> for Matrix {
    type Output = Self;

    fn add(self, rhs: &Self) -> Self::Output {
        assert!(self.cols == rhs.cols && self.rows == rhs.rows);
        let mut d = Vec::<f64>::new();
        for i in 0..self.data.len() {
            d.push(self.data[i] + rhs.data[i]);
        }
        Matrix {
            cols: self.cols,
            rows: self.rows,
            data: d,
        }
    }
}

impl std::ops::Mul<&Matrix> for Matrix {
    type Output = Self;

    fn mul(self, rhs: &Self) -> Self::Output {
        assert!(self.cols == rhs.rows);
        let mut d = Matrix {
            cols: rhs.cols,
            rows: self.rows,
            data: Vec::with_capacity((rhs.cols * self.rows) as usize),
        };
        for r in 0..self.rows {
            for c in 0..rhs.cols {
                d.data.push(0.0);
                for j in 0..rhs.rows {
                    let idx = d.index(r, c);
                    d.data[idx] += self.data[self.index(r, j)] * rhs.data[rhs.index(j, c)];
                }
            }
        }
        d
    }
}

struct Images {
    data: Vec<Matrix>,
}

impl From<(Vec<u8>, Vec<u32>)> for Images {
    fn from(value: (Vec<u8>, Vec<u32>)) -> Self {
        assert!(value.1.len() >= 2);
        let mut data = Vec::<Matrix>::new();
        let whole = (value.1[1] * value.1[2]) as usize;
        for i in 0..(value.1[0] as usize) {
            let mut mdata = Vec::<f64>::new();
            for j in 0..whole {
                mdata.push((value.0[(i * whole) + j] as f64) / 255.0);
            }
            data.push(Matrix {
                data: mdata,
                cols: value.0[2].try_into().unwrap(),
                rows: value.0[1].try_into().unwrap(),
            });
        }
        Self { data }
    }
}

struct Labels {
    data: Vec<u8>,
}

impl From<(Vec<u8>, Vec<u32>)> for Labels {
    fn from(value: (Vec<u8>, Vec<u32>)) -> Self {
        Self {
            data: value.0[0..value.1[0] as usize].to_vec(),
        }
    }
}

struct Network {
    weights: Vec<Matrix>,
    biases: Vec<Matrix>,
    eps: f64,
    rate: f64,
    i: usize,
}

struct Shape {
    input: u32,
    output: u32,
}

mod ops {
    pub fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }
    pub fn tanh(x: f64) -> f64 {
        x.tanh()
    }
    pub fn relu(x: f64) -> f64 {
        x.min(0.0)
    }
    pub fn power(x: f64, y: f64) -> f64 {
        x.powf(y)
    }
}

impl Network {
    pub fn new(layers: Vec<Shape>, eps: f64, rate: f64) -> Self {
        let mut w = Vec::<Matrix>::new();
        let mut b = Vec::<Matrix>::new();
        for l in layers.iter() {
            w.push(Matrix::new(l.input, l.output));
            b.push(Matrix::new(1, l.output));
        }
        Self {
            biases: b,
            weights: w,
            eps,
            rate,
            i: 0,
        }
    }
    pub fn feed_forward(&mut self, inp: Matrix) -> Matrix {
        assert!(self.i < self.weights.len());
        assert!(inp.rows == 1);
        let l = (inp * &self.weights[self.i]) + &self.biases[self.i];
        self.i += 1;
        l
    }
}

struct IdxFile<T: From<(Vec<u8>, Vec<u32>)>> {
    dims: Vec<u32>,
    magic: u32,
    bytes: Vec<u8>,
    items: T,
}

impl<T: From<(Vec<u8>, Vec<u32>)>> IdxFile<T> {
    pub fn read<J: AsRef<Path>>(path: J, dimsc: usize) -> std::io::Result<Self> {
        let source = std::fs::read(path)?;
        let magic = u32::from_be_bytes(source[0..4].try_into().unwrap());
        let mut dims = Vec::<u32>::new();
        for i in 0..dimsc {
            dims.push(u32::from_be_bytes(
                source[(4 * i)+4..(4 * i) + 4*2].try_into().unwrap(),
            ));
        }
        let bytes = source[4 + (4 * dims.len())..].to_vec();
        Ok(Self {
            dims: dims.clone(),
            magic,
            bytes: bytes.clone(),
            items: T::from((bytes, dims)),
        })
    }
}

fn main() {
    let mut n = Network::new(
        vec![
            Shape {
                input: 784,
                output: 784,
            },
            Shape {
                input: 784,
                output: 16,
            },
            Shape {
                input: 16,
                output: 16,
            },
            Shape {
                input: 16,
                output: 10,
            },
        ],
        1e-3,
        1e-3,
    );
    let mut inp = Matrix::new(1, 784);
    while n.i < n.weights.len() {
        inp = n.feed_forward(inp);
    }
    //println!("{inp:?}");
    let train_imgs =
        IdxFile::<Images>::read("data/trainset-imgs/train-images-idx3-ubyte", 3).unwrap();
    let train_txt =
        IdxFile::<Labels>::read("data/trainset-text/train-labels-idx1-ubyte", 1).unwrap();
    println!(
        "train_imgs: {}, train_txt: {}",
        train_imgs.items.data.len(),
        train_txt.items.data.len()
    );
}
