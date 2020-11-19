const MNIST_SIDE: usize = 28;
const MNIST_AREA: usize = MNIST_SIDE * MNIST_SIDE;

pub trait MnistNeuron {
    fn load_val(&mut self, x: usize, y: usize, val: f32);
}

pub trait MnistNetwork {
    fn get_neurons(&self) -> Vec<Box<dyn MnistNeuron>>;

    /// Takes the vector of all images, and the index of the image
    /// you want to load
    fn load_img(&self, img_vec: Vec<f32>, img_i: usize) {
        for j in 0..MNIST_SIDE {
            for i in 0..MNIST_SIDE {
                let val_i =
                    // First get img offset
                    ((img_i + 5800) * MNIST_AREA)
                        // Then get y offset
                        + (j * MNIST_SIDE)
                        // Then get x offset
                        + i;

                let &val = img_vec.get(val_i).unwrap();

                for mut neuron in self.get_neurons() {
                    neuron.load_val(i, j, val);
                }
            }
        }
    }

    fn perform_adjustment(&self);

    fn take_metric(&self, train_img_vec: Vec<f32>);
}