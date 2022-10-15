# quantum_classifier_low_rank_obs
Course project on quantum machine learning to generate a random dataset using a parametric quantum circuit by measuring a high-rank observable, and classifying the data with a quantum SVM by measuring a low-rank observable. The code is based on `Cirq`.

### Objectives and Tasks
- Generate artificial data by fixing a parameterized quantum circuit model, feeding it random input parameters, and measuring a high-rank observable to obtain the labels.
- Train the same parameterized quantum circuit but with a low-rank observable to classify the above generated artificial dataset.
- Benchmark the performance of the resulting low-rank model using cross-validation.

### Introduction
In the present implementation, the structure of the classifier is based on explicit model of quantum SVM. There are three parts of the complete circuit. The first is an encoding circuit to map input features non-linearly in the Hilbert space. This circuit has parametric gates where parameters are angles of rotations of gates, which are supplied with feature values (These parameters are not trainable). The second part is a variational circuit, which has trainable parameters so that the quantum classifier can find boundary of separation between encoded data points. The third part is the observable measured at the end of the circuit to determine how well the circuit is performing as a classifier.

### Generation of random data
In this implementation, a quantum circuit consisting of the encoding circuit and measuring a high rank observable is used to generate the random data. Specifically, the architecture of the encoding circuit chosen is as follows: Each qubit is acted upon by a sequence RY-RZ-RY. The gates have the same angle of rotation for the same qubit. The angle is different if the qubit index is different. This way, the input features have the same dimension as the number of qubits in the circuit. 

The way data is generated is as follows. For each data sample, a random vector of features is generated and used as angles for the circuit in Fig. 3. Let us call one instance of features as $x_i$. These values are generated uniformly randomly between −π and π. The output statevector of the encoding circuit is then found by simulation (let’s call it $|ψ(x_i)⟩$). Then, the expectation value of a high rank observable in the state $|ψ(x_i)⟩$ is calculated. Let the observable be $O$, then the expectation value is $⟨ψ(x_i)|O |ψ(x_i)⟩$.

The high rank observable used here is $O = Z ⊗ Z ⊗ Z . . . ⊗ Z$, i.e. the Pauli $Z$ on every qubit. In particular, this observable is chosen because it is full-rank, the eigenvalues are easy to calculate (they are -1 and +1), so it can be seen no eigenvalue is 0. Also, this is same as measuring all qubits in computational basis, so no extra basis rotations would be needed in principle if the measurement is actually implemented. After calculating this expectation value, the ground-truth labels are assigned as $y_i = +1$ if $⟨ψ(x_i)|O |ψ(x_i)⟩ > 0$, otherwise the label is -1.

### Explanation of the code
#### General
The quantum circuits are simulated using `Cirq`. For all circuits, the number of qubits is kept fixed at 4. The dataset generated has a total of 300 data points. For training and evaluation of metrics, functions from `numpy`, `scipy`, and `sklearn` have been used. For plotting, `matplotlib` has been used.

#### Encoding circuit and data generation
The logic for data generation and encoding circuit is the same that was explained before. The function `qml_encoding_circuit` takes as input the feature vector and returns a generator for encoding circuit (which is RY-RZ-RY on every qubit). Since features are reused, the dimension of input features is same
as number of qubits. The function `qml_data_generation` circuit returns the data encoding (data generation) quantum circuit using the generator returned by `qml_encoding_circuit`. The function `dataset_generation` is a helper function to repeatedly call `qml_data_generation` circuit to generate a dataset of required size. The logic is based on expectation value of the high rank observable $Z⊗Z · · ·⊗Z$ on all qubits, as described before. Later on in the code, I use the generated dataset of 300 datapoints in two ways - once as 80:20 split and another time as input to a 3 fold cross-validation training strategy.

#### Variational circuit and architectures studied
The function `qml_variational_circuit` returns a generator for the variational circuit which has trainable parameters, given by `params` argument. Unlike the encoding circuit, the variational circuit does not reuse parameters. The function has another argument, `architecture_type`, which is an integer to select one out of three implemented variational circuit architectures. For all three architectures, the variational circuit consists of multiple possible layers of rotation and entangling gates. The first architecture considered has alternating RX rotation gates and linearly connected CZ. The second architecture considered has alternating RX rotation gates, RZ rotation gates, and linearly connected CZ. The third architecture considered has one elementary unit consisting of RX, RZ, then followed by two-qubit CX gates with all possible combinations of control and target qubits, followed by RX and RZ gates. The intention behind studying multiple types of variational circuits is to investigate whether there is an impact of using a circuit which has more complexity on the ability of the circuit to classify the data.

#### Low rank observables used
The low-rank observables used here are the projection operators. Projectors of different ranks have been constructed as follows: rank-1 projector is taken as $|0000⟩ ⟨0000|$, rank-2 projector is taken as $|0000⟩ ⟨0000| + |0001⟩ ⟨0001|$, rank-3 projector is taken as $|0000⟩ ⟨0000| + |0001⟩ ⟨0001| + |0010⟩ ⟨0010|$, etc. Each progressively projects onto a subspace of increasing dimension. My motivation for choosing these observables is that considering all these gives an easy way to train the classifiers with observables of different ranks. Note that the low-rank observables are chosen are such that the projector of rank $r$ will have $r$ eigenvalues equal to 1 and $16 − r$ eigenvalues equal to zero.

#### Classifier circuit and prediction
The function `qml_classifier_circuit` takes the generators of encoding circuit and variational circuit to construct a Cirq quantum circuit to be used as a classifier. It is important to clarify how the classifier makes a prediction, given the input features ( $x_i$ ), a given set of values of parameters for variational circuit $θ$, and a given low-rank observable, say $P$. Let the output state of the circuit be $\ket{\widetilde{ψ}(x_i, θ)}$ 
(available due to simulation). First, the expectation value of the low-rank observable is calculated as $\bra{\widetilde{ψ}(x_i, θ)} P \ket{\widetilde{ψ}(x_i, θ)}$. Note that, since the eigenvalues of the low-rank observables (projectors) are either 0 or 1, this expectation value will lie between 0 and 1. To use this to predict a label, this is first maped to the range -1 to 1 (i.e. same range as target labels). Then the predicted label is calculated as:
$y_i = (= +1, \text{if} 2 ∗ \bra{\widetilde{ψ}(x_i, θ)} P \ket{\widetilde{ψ}(x_i, θ)} − 1 > 0, \text{and} =−1, \text{otherwise})$.

The `classifier_predictions` function takes as input the frame of feature vectors and constructs the specified type of classifier circuit. The parameters and the low-cost observable are also specified as arguments. This function simulates the required circuits, and returns the predicted labels for the input feature vectors.

#### Cost function and optimization

The loss per training example is inspired from SVMs and is taken as (1 - (predicted label)*(ground truth label)) (hinge loss). The total cost function to be minimized is the sum of these values divided by the number of training examples as a normalization. This is done by the `training_loss` function, which
calculates the cost on both the training set and the test set for analysis (normalized by number of data points). This function only returns the cost on training set, so that this function can be used as input to the optimizer to minimize the training cost. (Note that although this function takes the test data as input, that is only done to keep track of cost on test data on every iteration. So, there is no leakage of information from test data set to training data set).

For optimization, `scipy.optimize.minimize` is used and the minimization algorithm used is the gradient-free `COBYLA`. Maximum iterations is kept at 250. The initial value of parameters (of variational circuit, over which we are optimizing) is drawn randomly from uniform distribution between −2π and 2π.

#### Metrics of performance

In this implementation, the performance after optimization is quantified using accuracy on training data, accuracy on test data, and the difference between these two values. Further, the cost values on training set and test set are also tracked for every iteration. The motivation for these quantities is to try to infer if the model is generalizing or over-fitting. If the model over-fits, accuracy on training set should be substantially higher than accuracy on test/validation set. I also use the difference in the values of these quantities to try to determine how well the model has generalized. Similarly, cost function value for test/validation set will be much higher than that on training set if the model has not generalized well. If the optimization gets stuck in a local minima, this can also be concluded by looking at cost function values on training and test set for every optimization iteration (the curves should become flat in that case).

#### Training with 80:20 split
As a first investigation, the generated data set is divided into a training and test set in the ratio 80:20 (in a
stratified manner). This is done to be able to train classifiers relatively quickly to see the performance. For
every architecture type, classifiers are trained while the rank of observable is varied from 2 to 14 in steps of 2.
For first two architectures, 2 and 4 layers are taken, while only 2 layers are taken for architecture 3 (because a
4 layer circuit will have too many parameters - making it very slow to optimize). So the dataset of 300 entries
leads to 240 entries in training set and 60 entries in test set.
After the optimization steps, the accuracy values on training set, test set and their differences are plotted.

#### Training with cross-validation

One observation I make is that the result of the classifiers trained in the previous step will depend on the
particular subset of data chosen for training, as well as the choice of random initial parameters (and hence the
success of the optimization step). To average over these two effects, I train classifiers belonging to all three
architecture types using 3-fold cross-validation method (the folds are taken on original data set). So the dataset
of 300 entries gives 200 entries in training set and 100 entries in validation set in every iteration. The average
accuracy values resulting from this are plotted.

### Results and Discussion
Results are presented in the jupyter notebook.

#### Discussion-1

First, I make some general observations. Referring to the figures on variation of training cost with optimization
iteration, I observe:
- In most cases, value of total cost on both training set and test set seems to be decreasing with training
iterations. This seems to suggest that the classifier is able to learn trends from training data and able

to generalize it to test data, at least to some limited extent. In most classifiers considered here, the
cost on test set remains higher than cost on training set at the end of optimization. This could suggest
slight overfitting due to model/observable or training data size being insufficient. 

- In architecture 1, there are some instances where cost on test set is actually increasing
with iterations (while the cost on training set is still decreasing). This could indicate that the particular
optimum parameters achieved for this model may have led to a model that has overfit and failed to
generalize.
1 In some cases, the costs on training sets don’t vary much with iteration and the curves seem to be flat.
It is possible that the optimizer gets stuck in local minima (which seems to last for a substantial range of
parameter values, otherwise perhaps the optimizer could have found some parameter combination after a
larger number of iterations which leads to decrease in cost).
Now, I turn my attention to variation of accuracy with rank, layers and architecture. First, I will formulate some hypotheses about these variations and then try to interpret
from the data if these seem to be correct (to some extent) or not.

Hypotheses:

- Variation with rank: The low-rank observables chosen for the classifier are projectors such that a projector
of rank r has r eigenvalues equal to 1 and 16 − r eigenvalues equal to 0. If r is very small, then most
of the outputs this observable measures are the 0 eigenvalues, hence the expectation value is very close
to 0. In this case, the classifier has the tendency to classify most of the inputs as belonging to −1 class.
Similarly, if r is very large, then most of the outputs this observable measures are the +1 eigenvalues,
hence the expectation value is very close to 1. In this case, the classifier has the tendency to classify most
of the inputs as belonging to +1 class. In both cases of very low or very high observable rank, it should be
difficult for the model to generalize, and good accuracy and generalization should be achieved for ranks
closer to half of maximum possible rank of 16. (This is because the input data set also has roughly even
distribution of both classes).
- Variation with layers: In general, increasing the number of layers should lead to an increase in expressivity
of the circuit, so the accuracy should increase (at least until it reaches the point of overfitting, after that
I expect there should be a decrease in test set accuracy).
- Variation with architecture: Architecture 1 has lowest complexity because it has only RX rotation gates.
Architecture 3 is most complex with all possible entangling gates, along with RX and RZ gates. The
complexity of architecture 2 is in between these two. So I expect architecture 1 to show lowest accuracy
values as it does not have sufficient diversity of rotation gates.
