# quantum_classifier_low_rank_obs
Course project on quantum machine learning to generate a random dataset using a parametric quantum circuit by measuring a high-rank observable, and classifying the data with a quantum SVM by measuring a low-rank observable. The code is based on `Cirq`. (Full report available on request).

### Objectives and Tasks
- Generate artificial data by fixing a parameterized quantum circuit model, feeding it random input parameters, and measuring a high-rank observable to obtain the labels.
- Train the same parameterized quantum circuit but with a low-rank observable to classify the above generated artificial dataset.
- Benchmark the performance of the resulting low-rank model using cross-validation.

### References
- aQa lecture slides
- https://arxiv.org/abs/1905.10876 (Expressibility and entangling capability of parameterized quantum circuits
for hybrid quantum-classical algorithms)
- https://arxiv.org/pdf/1804.11326.pdf (Supervised learning with quantum enhanced feature spaces
)
- https://arxiv.org/pdf/1802.06002.pdf (Classification with Quantum Neural Networks
on Near Term Processors)
- https://arxiv.org/pdf/1804.00633.pdf (Circuit-centric quantum classifiers
)
- https://www.tensorflow.org/quantum/tutorials/mnist
- https://pennylane.ai/qml/demos/tutorial_variational_classifier.html

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
of rank $r$ has $r$ eigenvalues equal to 1 and $16 − r$ eigenvalues equal to 0. If $r$ is very small, then most
of the outputs this observable measures are the 0 eigenvalues, hence the expectation value is very close
to 0. In this case, the classifier has the tendency to classify most of the inputs as belonging to −1 class.
Similarly, if $r$ is very large, then most of the outputs this observable measures are the +1 eigenvalues,
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

Now I present my interpretation of the data.

- Variation with rank: In general, all architecture seem to show an increase in test set accuracy as rank is
increase, followed by either a decrease or attaining a not so increasing value (let’s call this case saturation
of accuracy with rank). For architecture 1, for 2 layer case, better test set accuracy (among different
ranks) is seen at ranks 6 and 8, after that there is a decrease followed by an increase again at rank 14. This seems to agree with better performance expected at intermediate values of rank. (However, for 4
layer case, test set accuracy generally keeps increasing with rank - so that trend is not so clear). Looking
at difference of accuracy on training and test set, for two layer case, the difference is low at intermediate
ranks of 4 and 8, implying good generalization at these ranks. For 4 layer case, low difference and hence
good generalization is observed at ranks 10, 12 and to some extent at rank 14. (Again, with 4 layer case,
the trend of variation with accuracy with rank is not so clear). For architecture 2, 2 layer case is showing higher test accuracy at ranks 6,8,10, and 4 layer
case at ranks 4, 10, 12. Difference in training accuracy and test accuracy is lowest at ranks 4,6,10 for 2
layers and at ranks 10,12 for 4 layers (implying good generalization). This also seems to agree with my
hypothesis of good performance at intermediate ranks. For architecture 3, The behaviour is interesting as test accuracy shows peaks around two
different ranks (6 and 12), and difference in accuracy suggests relatively good generalization at ranks 6,
and then again at 10 and 12. My take is that it again suggests good accuracy at intermediate ranks,
although increase in model complexity could be having an additional effect on expressivity behaviour.
- Variation with layers: For architecture 1, circuit with 4 layers generally shows higher test
accuracy than 2 layers for different ranks. However, that does not seem to be true for architecture 2. A possible reason could be that architecture 1 contains only RX gates, so it is less expressive by itself, and increasing the layers could be increasing the expressivity, while architecture 2 already has better
expressivity so perhaps increasing the number of layers is promoting overfitting in that case. (Although,
what we see is a combined effect of rank and number of layers, so perhaps it requires more investigation
to draw a conclusion).
- Variation with architecture: Comparing the 3 architectures for same number of layers (2).
For most values of rank, architectures 2 and 3 give higher test set accuracy than architecture 1 (more often
it is architecture 3). Architectures 2 and 3 show lower difference in training and test accuracy (implying
better generalization) than architecture 1 for different ranks. Hence, I conclude that increased complexity
in architectures 2 and 3 is leading to an increase in expressivity of the model. This is in agreement with
my hypothesis.

In addition, there are some general observations. The accuracy values achieved are are not great. I propose
possible reasons for that at the end. Also, the above results were based on a specific instance of training data
set chosen and choice of random initial parameters. To average over these effects, 3 fold cross validation was
done and results are presented in the next section.

#### 3 fold cross-validation (Discussion-2)

For 3-fold validation, only 2-layer circuits are considered.

I observe the same trends in average test accuracy that it is higher for intermediate ranks than
for low and high ranks of the observable. From difference of accuracy on training and test set, generalization
seems to be better on intermediate ranks (although the value at rank 14 does differ from this trend). Although
largely I would say 3 fold CV results agree with earlier conclusions, the effect (for e.g. increase in accuracy of
architectures 2 and 3 as compared to 1) is less pronounced here.

### Conclusion and future scope

Intuitively, I would describe expressivity as the (inherent) capability of a model to learn and generalize the
variation and complexity of a given learning problem. While the exact quantum circuit chosen for the classifier
itself would specify a maximum amount of expressivity obtainable through the model, the observable itself
being measured at the end of the circuit would dictate its own restrictions on the expressivity of the whole
model. This is why it is important to investigate this expressivity of the observable - since it is this observable
which is the window to see from the classical world into the decision boundary obtained in the Hilbert space.

Broadly, based on these results, I would stand by my earlier conclusion that if the dataset is balanced, then very
low rank or high rank observable measured at the end of the circuit will cause most of the inputs to be predicted
to belong to one class or another (because most of the eigenvalues of very low rank observables considered here
are 0, leading to most inputs labelled as −1, and most of the eigenvalues of high rank observables considered
here are 1, leading to most inputs labelled as +1). Note that this is based on the fact that observables taken here
are projectors with eigenvalues 0 and 1 only. So, the best results for binary classification should be obtained
for observable rank around half of maximum possible rank (which is dimension of Hilbert space).

One needs to consider that these conclusions are based on this specific choice of observable, i.e. on the specific
eigenvalues of these projectors (0 and 1). If different eigenvalues (or different distribution of the same eigenvalues,
such as what is done here by changing the rank) are chosen, it is possible that the behaviour will change
because the expectation value of the observable will change. This is because the prediction is based on this
expectation value. It could be interesting to investigate expressivity of observable with different eigenvalues
having the same rank.

Another aspect is that the conclusions here that rank should be somewhere intermediate are based on the
dataset being nearly balanced, i.e. roughly equal division in both classes. Based on this, there is a possibility
that very low or very high rank projectors could do well on datasets which are highly imbalanced (i.e. skewed,
meaning points belonging to one class are much larger than the other). This is from the same logic that low rank
projectors have most eigenvalues 0 (hence expectation value is closer to 0) and with increasing rank, expectation
value moves closer to 1 for high ranks). It could be interesting as a further study to investigate this.

An alternative explanation for the observed trends is also possible: perhaps the expressivity is indeed increasing
monotonically with the rank, and maybe the dataset proves to be insufficient (i.e. has low number
of data points) such that the model is not able to learn. To investigate this further, this behaviour should be
studied with larger data sets (this was not possible here because that takes longer to train).

This also leads to the low accuracy which is observed. It has to be acknowledged that the accuracy is not
good enough (although one can devise a further experiment to see how well classical ML models perform on
quantum data like this). There could be many possible reasons which could be investigated further. One is
that perhaps the circuits considered here as classifiers are too simple and fail to capture the complexity of the
data properly. So, more complex quantum classifiers can be considered (i.e. increasing the variance). Second,
perhaps the circuits are expressive enough but the data set is simply too small, so that the model is not able
to learn sufficiently, so the experiments could be carried out with larger data set. Third, the optimization technique
used is not gradient based, it is possible that even in the existing combination of circuits and data that
an even better optimum can be found with a gradient-based approach. Finally, it is also important to consider
that the observable used to generate the data is quite different from the one used for prediction - the former
has eigenvalues +1 and -1, whereas the latter uses projectors with eigenvalues 1 and 0. So another possible
direction of further study could be choosing the low rank observable in a way that the non-zero eigenvalues of
it are a subset of eigenvalues of high-rank observable used to generate the data.

It is also possible to apply more classically-inspired ML techniques to such quantum classifiers. For example,
we can investigate whether multiple classifiers with low rank observables perform well enough, if their
outputs are combined, for example with an ensemble method.
