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
                    d.data[idx] += self.data[self.index(r, j)]*rhs.data[rhs.index(j, c)];
                }
            }
        }
        d
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

fn main() {
    let mut n = Network::new(vec![
        Shape{ input: 2, output: 8 },
        Shape{ input: 8, output: 8 },
        Shape{ input: 8, output: 8 },
        Shape{ input: 8, output: 4 },
    ], 1e-3, 1e-3);
    let mut inp = Matrix::full(1, 2, 2.0);
    while n.i < n.weights.len() {
        inp = n.feed_forward(inp);
    }
    println!("{inp:?}");
}
