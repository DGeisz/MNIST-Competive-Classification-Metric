use std::rc::Rc;

pub const MNIST_SIDE: usize = 28;
pub const MNIST_AREA: usize = MNIST_SIDE * MNIST_SIDE;

pub trait MnistNeuron {
    fn get_name(&self) -> String;

    /// This is the value the neuron outputs in response
    /// to a particular input
    fn compute_em(&self) -> f32;
}

pub trait MnistNetwork {
    fn get_neurons(&self) -> Vec<Rc<dyn MnistNeuron>>;
    fn load_val(&self, x: usize, y: usize, val: f32);

    /// Takes the vector of all images, and the index of the image
    /// you want to load
    fn load_img(&mut self, img_vec: &Vec<f32>, img_i: usize) {
        for j in 0..MNIST_SIDE {
            for i in 0..MNIST_SIDE {
                let val_i =
                    // First get img offset
                    (img_i * MNIST_AREA)
                        // Then get y offset
                        + (j * MNIST_SIDE)
                        // Then get x offset
                        + i;

                let &val = img_vec.get(val_i).unwrap();

                self.load_val(i, j, val);
            }
        }
    }

    fn perform_adjustment(&mut self);

    /// Returns the accuracy of current network in classifying
    /// mnist using competitive prototypes
    fn take_metric(
        &mut self,
        train_img_vec: Vec<f32>,
        train_lbl_vec: Vec<u8>,
        train_epochs: usize,
        test_img_vec: Vec<f32>,
        test_lbl_vec: Vec<u8>,
        logger_on: bool
    ) -> f32 {
        // Train the network
        let train_len = train_lbl_vec.len();

        for _ in 0..train_epochs {
            for img_i in 0..train_len {
                self.load_img(&train_img_vec, img_i);
                self.perform_adjustment();

                // Log the boi
                if logger_on {
                    if codexc_log::run(2) {
                        if img_i % 100 == 0 {
                            println!("Trained on img: {}", img_i);
                        }
                    } else if codexc_log::run(1) {
                        if img_i % 1000 == 0 {
                            println!("Trained on img: {}", img_i);
                        }
                    } else if codexc_log::run(0) {
                        if img_i % 10000 == 0 {
                            println!("Trained on img: {}", img_i);
                        }
                    }
                }
            }
        }

        // Create classifier
        let mut neuron_class_wins: Vec<[f32; 10]> =
            (0..self.get_neurons().len()).map(|_| [0.0; 10]).collect();
        let mut class_total_wins = [0.0_f32; 10];

        for (img_i, &lbl) in train_lbl_vec.iter().enumerate() {
            self.load_img(&train_img_vec, img_i);

            let em_vec = self
                .get_neurons()
                .iter()
                .map(|neuron| neuron.compute_em())
                .collect::<Vec<f32>>();

            let mut max_em = 0.0;
            let mut max_em_i = 0;

            // Find the max index
            for (i, &em) in em_vec.iter().enumerate() {
                if em > max_em {
                    max_em = em;
                    max_em_i = i;
                }
            }

            // Uptick win for max neuron in proper classification type
            let max_class = neuron_class_wins.get_mut(max_em_i).unwrap();
            max_class[lbl as usize] += 1.0;

            // Uptick total wins
            class_total_wins[lbl as usize] += 1.0;

            // Log the boi
            if logger_on {
                if codexc_log::run(2) {
                    if img_i % 100 == 0 {
                        println!("Created classification for img: {}", img_i);
                    }
                } else if codexc_log::run(1) {
                    if img_i % 1000 == 0 {
                        println!("Created classification for img: {}", img_i);
                    }
                } else if codexc_log::run(0) {
                    if img_i % 10000 == 0 {
                        println!("Created classification for img: {}", img_i);
                    }
                }
            }
        }

        // Test model on test data
        let mut total_correct = 0;

        for (img_i, &lbl) in test_lbl_vec.iter().enumerate() {
            self.load_img(&test_img_vec, img_i);

            let mut weighted_classification_vec = [0.0_f32; 10];

            for (neuron, neuron_wins) in self.get_neurons().iter().zip(neuron_class_wins.iter_mut())
            {
                let em = neuron.compute_em();

                for i in 0..10 {
                    weighted_classification_vec[i] += neuron_wins[i] * em;
                }
            }

            let mut classification_vec = [0.0_f32; 10];

            for i in 0..10 {
                classification_vec[i] = weighted_classification_vec[i] / class_total_wins[i];
            }

            let mut max_val = 0.0;

            //After the for loop, max_i will be the classification
            let mut max_i = 0;

            for i in 0..10 {
                if classification_vec[i] > max_val {
                    max_i = i;
                    max_val = classification_vec[i];
                }
            }

            // Check classification
            if max_i == lbl as usize {
                total_correct += 1;
            }

            // Log the boi
            if logger_on {
                if codexc_log::run(2) {
                    if img_i % 100 == 0 {
                        println!("Classified img: {}", img_i);
                    }
                } else if codexc_log::run(1) {
                    if img_i % 1000 == 0 {
                        println!("Classified img: {}", img_i);
                    }
                } else if codexc_log::run(0) {
                    if img_i % 10000 == 0 {
                        println!("Classified img: {}", img_i);
                    }
                }
            }
        }

        return total_correct as f32 / test_lbl_vec.len() as f32;
    }
}
