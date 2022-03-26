import pickle
import numpy as np
from time import time
from sympy import default_sort_key
import matplotlib.pyplot as plt  
from sklearn.svm import SVC
from noisyopt import minimizeSPSA

import warnings
warnings.filterwarnings('ignore')  # Ignore warnings

from discopy import rigid
from discopy.quantum import qubit, Circuit, Id, Measure, Bra

from lambeq.ccg2discocat import DepCCGParser
from lambeq.circuit import IQPAnsatz
from lambeq.core.types import AtomicType
from lambeq.ansatz import Symbol

from pytket import Circuit as tketCircuit
from pytket.extensions.qiskit import tk_to_qiskit, AerBackend, IBMQBackend, IBMQEmulatorBackend

plt.rcParams.update({'font.size': 18})

# Global Vars
SEED = 123
np.random.seed(SEED)  # Fix the seed
rng = np.random.default_rng(SEED)

N = AtomicType.NOUN
S = AtomicType.SENTENCE

DEFAULT_PARSER = DepCCGParser(possible_root_cats=['S[dcl]'])
DEFAULT_ANSATZ = IQPAnsatz({N: 1, S: 1}, n_layers=1)

WEIGHT_DIR = 'weights/'
DATA_DIR = 'data/'

def timestamp(te, ts, msg=''):
    print(f'Took {round(te - ts, 3)} s {msg}')

def randint(rng, low=-1 << 63, high=1 << 63-1):
    return rng.integers(low, high)

def normalise(predictions):
    # apply smoothing to predictions
    predictions = np.abs(predictions) + 1e-9
    return predictions / predictions.sum()

def make_pred_fn(circuits, parameters, backend_config):
    measured_circuits = [c >> Id().tensor(*[Measure()] * len(c.cod)) for c in circuits]
    circuit_fns = [c.lambdify(*parameters) for c in measured_circuits]

    def predict(params):
        outputs = Circuit.eval(*(c_fn(*params) for c_fn in circuit_fns),
                               **backend_config, seed=randint(rng))
        return np.array([normalise(output.array) for output in outputs])
    return predict

def make_cost_fn(pred_fn, labels):
    def cost_fn(params, **kwargs):          
        predictions = pred_fn(params)

        cost = -np.sum(labels * np.log(predictions)) / len(labels)  # binary cross-entropy loss
        costs.append(cost)

        acc = np.sum(np.round(predictions) == labels) / len(labels) / 2  # half due to double-counting
        accuracies.append(acc)

        return cost

    costs, accuracies = [], []
    return cost_fn, costs, accuracies

def make_callback_fn(vocab, fname):
    summary = {'vocab': vocab, 'count': 0}
    def callback_fn(params, **kwargs):
        summary['vals'] = params
        summary['count'] += 1
        save_weights(fname, **summary)
        
        # Display progess update (there is minimizeSPSA bug for disp kwarg) 
        if summary['count'] % 10 == 0:
            print(f'Completed iteration: {summary["count"]}')
            
    return callback_fn

def plot_training_result(costs, accs):
    fig = plt.figure(figsize=(8,6))
    
    ax1 = fig.add_axes([.2, .6, .6, .3])
    ax1.plot(range(len(costs)), costs)
    ax1.set_ylabel('Cost')
    
    ax2 = fig.add_axes([.2, .2, .6, .3])
    ax2.plot(range(len(accs)), accs)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Accuracy')
    
    plt.show()

def train_embedding(f_training, f_testing, f_save_weights='trained_embeddings.pkl', f_load_weights='', parser=DEFAULT_PARSER, ansatz=DEFAULT_ANSATZ, backend_config={}):
    """Automates word embedding training.
    
    Parameters
    ----------
    f_training : str
        File name of training data. Should be located in the `data/` folder.
    f_testing : str
        File name of testing data. Should be located in the `data/` folder.
    f_save_weights : str
        File to save vocabulary parameters to. Will be saved in the `WEIGHT_DIR` folder.
    f_load_weights : str
        File used to load existing weights from. Defualt is `''` so randomly initialized
        weights will be used.
    parser : lambeq.ccg2discocat.CCGParser
        Parser used to convert sentence data to DisCoCat string diagrams. Default is
        `DepCCGParser`.
    ansatz : lambeq.circuit.CircuitAnsatz
        Ansatz used for the parameterized quantum cicuit representation of DisCoCat
        diagrams.
    backend_config : dict
        Configuration setting for quantum backend. Will be passed to circuit evaluation
        function.
    """
    
    # Load data
    train_labels, train_data = read_data(f_training)
    test_labels, test_data = read_data(f_testing)
    
    # Build circuits
    train_circs = build_circuits(train_data, parser, ansatz)
    test_circs = build_circuits(test_data, parser, ansatz)
    
    # Get circuit parameters (vocabulary) and initial weights
    if f_load_weights:
        embedding = load_weights(f_load_weights)
        vocab = embedding['vocab']
        x0 = embedding['vals']
    else:
        all_circuits = train_circs + test_circs
        vocab = sorted(
            {s for circ in all_circuits for s in circ.free_symbols},
            key=default_sort_key
        )
        x0 = np.random.rand(len(vocab))
    
    train_pred_fn = make_pred_fn(train_circs, vocab, backend_config)
    test_pred_fn = make_pred_fn(test_circs, vocab, backend_config)

    train_cost_fn, train_costs, train_accs = make_cost_fn(train_pred_fn, train_labels)
    callback = make_callback_fn(vocab, f_save_weights)
    
    result = minimizeSPSA(train_cost_fn, x0=x0, a=0.2, c=0.06, niter=111, callback=callback)
    
    save_weights(f_save_weights, vocab=vocab, vals=list(result.x))
    
    plot_training_result(train_costs, train_accs)
    
    test_cost_fn, _, test_accs = make_cost_fn(test_pred_fn, test_labels)
    test_cost_fn(result.x)
    print('Test accuracy:', test_accs[0])
    return vocab, list(result.x)

def read_data(fname, dir=DATA_DIR):
    if fname == '':
        print('No filename passed, returning empty lists.')
        return [], []
    labels, sentences = [], []
    with open(dir + fname) as f:
        for line in f:
            labels.append([1, 0] if line[0] == '1' else [0, 1])
            sentences.append(line[1:].strip())
    return np.array(labels), sentences

def load_weights(fname='trained_embeddings.pkl', dir=WEIGHT_DIR):
    with open(WEIGHT_DIR + fname, 'rb') as f:
        loaded_dict = pickle.load(f)
        # loaded_dict['vocab'] = [Symbol(v) for v in loaded_dict['vocab']]
    return loaded_dict

def save_weights(fname, dir=WEIGHT_DIR, **kwargs):
    with open(dir + fname, 'wb') as f:
        pickle.dump(kwargs, f)

def remove_cups(diagram):
    # Remove cups to reduce post-selection in the circuit, for faster execution
    diags = []
    for box, offset in zip(diagram.boxes, diagram.offsets):
        if not box.dom:  # word box
            diags.insert(offset, box)
        else:  # cup (the only other type of box in these diagrams)
            i = 0
            off = offset
            while off != len(diags[i].cod) - 1:
                assert off > 0
                off -= len(diags[i].cod)
                i += 1
            left, right = diags[i:i+2]
            
            if len(left.cod) == 1:
                new_diag = right >> (left.r.dagger() @ rigid.Id(right.cod[1:]))
            else:
                assert len(right.cod) == 1
                new_diag = left >> (rigid.Id(left.cod[:-1]) @ right.l.dagger())

            diags[i:i+2] = [new_diag]

    assert len(diags) == 1
    return diags[0]

def build_circuit(x:str, parser=DEFAULT_PARSER, ansatz=DEFAULT_ANSATZ):
    if type(x) is not str:
        raise Exception('ERROR: `x` must be a string for build_circuit')
    raw_diagram = parser.sentence2diagram(x)
    opt_diagram = remove_cups(raw_diagram)  # optimize circuit by reduce amount of post-selection
    circ = ansatz(opt_diagram)
    return circ

def build_circuits(x:list, parser=DEFAULT_PARSER, ansatz=DEFAULT_ANSATZ):
    if type(x) is str:
        return [build_circuit(x, parser, ansatz)]
    raw_diagrams = parser.sentences2diagrams(x)
    opt_diagrams = [remove_cups(diagram) for diagram in raw_diagrams]  # optimize circuit by reduce amount of post-selection
    circs = [ansatz(diagram) for diagram in opt_diagrams]
    return circs

def sort_data(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    
    is_class_0 = y[:, 0] == 0
    sorted_x = np.concatenate([x[is_class_0], x[~is_class_0]])
    sorted_y = np.concatenate([y[is_class_0], y[~is_class_0]])
    return sorted_x, sorted_y

def get_transition_amp_sim_fn_deprc(backend_config={}, seed=SEED):
    print('WARNING: depreciated function - should use `transition_amp_sim_fn`')

    def fn(x_circ, y_circ):
        sim_circ = x_circ >> y_circ.dagger()
        return Circuit.eval(sim_circ, **backend_config, seed=seed).array[0]
    return fn

def transition_amp_sim_fn(x_circ, y_circ, backend_config={}, seed=SEED):
    """Implements a transition amplitude of two DisCoCat sentences to estimate the two
    state's fidelity. NOTE: this is buggy and only works for 1 qubit sentences.
    
    This is done by inverting one circuit, stripping the post-selction from the last set
    of measurements (typically the three qubits that encode a transitive verb), and
    replacing these with post-selection on the edge qubits (assuming that this is the
    sentence pregroup qubit). This allows us then to measure only the transition ampplitude
    on the sentence qubit by executing the kernel circuit and finding probability of measuring
    0. This is all very sketchy and has not been tested in a range of scenarios.

    Parameters
    ----------
    x_circ : discopy.quantum.Circuit
        Circuit representation of the first datapoint provided.
    y_circ : discopy.quantum.Circuit
        Circuit representation of the second datapoint provided.
    backend_config : dict
        Configuration parameters for executing the circuit when calling `get_counts()`.
        Must include a quantum backend (e.g. `AerBackend`).
    seed : int
        Random number seed.
    """

    # Find adjoint of second datapoint circuit
    y_dagger = y_circ.dagger()

    # Extract list of DiCoPy operations and their offsets - these will be adjusted
    operations = y_dagger.boxes
    offsets = y_dagger.offsets
    # New values to build circuit from
    new_ops = []
    new_offs = []
    # Iterate over definition of circuit and remove final post-selection
    for i in range(len(operations)):
        if not isinstance(operations[i], Bra):
            # Immediately add operations that are not post-selecting Bras
            new_ops.append(operations[i])
            new_offs.append(offsets[i])
        else:
            # Add single Bras since sentence qubits are encoded with multiple neighbouring
            # qubits (e.g. transitive verb has the tensor: $N \otimes S \otimes N$)
            if len(operations[i].dom) == 1:
                new_ops.append(operations[i])
                new_offs.append(offsets[i])

    dom = qubit  # domain of circuit
    cod = qubit @ qubit @ qubit  # new codomain of circuit without final post-selection
    yd_new = Circuit(dom, cod, new_ops, new_offs)
    yd_new = yd_new >> Bra(0) @ Id(1) @ Bra(0)  # post-select edge noun qubits

    # Build and execute kernel circuit
    kernel = x_circ >> yd_new
    kernel = kernel.to_tk()
    counts, = kernel.get_counts(**backend_config, seed=seed, measure_all=True, post_select=True)

    # Normalize probabilites
    total_p = sum(counts.values())
    res = {k : p/total_p for k, p in counts.items()}
    # Similarity is probability of measuring 0s
    sim = res.get((0,), 0)
    return sim

def swap_test_fn(c1, c2, backend_config={}, seed=SEED):
    ts = time()
    tk_circ1 = c1.to_tk()
    tk_circ2 = c2.to_tk()

    # Create index mappers to stack c2 and c1
    qubit_mapper = [n + tk_circ1.n_qubits for n in range(tk_circ2.n_qubits)]
    bit_mapper = [n + tk_circ1.n_bits for n in range(tk_circ2.n_bits)]
    tk_circ2_new_post_selection = {
        q_idx + tk_circ1.n_bits : val for q_idx, val in tk_circ2.post_selection.items()
    }
    # Combine circuits in parallel
    qc = tk_circ1.add_circuit(circuit=tk_circ2, qubits=qubit_mapper, bits=bit_mapper)
    qc.post_select(tk_circ2_new_post_selection)

    # Add 1 qubit/cbit circuit for ancillary bit
    anc = tketCircuit(1, 1)
    anc_qubit = qc.n_qubits
    anc_cbit = qc.n_bits
    qc.add_circuit(circuit=anc, qubits=[anc_qubit], bits=[anc_cbit])

    # Find control and target qubits for swap test. Note that they are the unmeasured qubits
    control_qubit = qc.qubits[-1]
    target_qubits = []
    for q in qc.qubits[:-1]:
        if q not in qc.qubit_to_bit_map:  # unmeasured
            target_qubits.append(q)

    # Complete SWAT test
    qc.H(anc_qubit)
    qc.CSWAP(control_qubit, *target_qubits)
    qc.H(anc_qubit)
    qc.Measure(anc_qubit, anc_cbit)

    t_setup = time() - ts
    ts = time()

    # Execute circuit
    res = qc.get_counts(**backend_config, seed=seed, normalize=True)
    # Normalize results
    total_p = sum(res[0].values())
    res = {k : p/total_p for k, p in res[0].items()}
    p0 = res.get((0,), 0.5)  # probability of measuring zero. If empty assume states are orthogonal
    sim = 2 * p0 - 1  # estimate inner product (similarity) using a SWAP test

    t_execute = time() - ts
    print('Execution took {:.3} %'.format(t_execute / (t_execute+t_setup)))
    return sim

def make_kernel_callback_fn(train_circs, test_circs, train_labels, test_labels, vocab, sim_fn=swap_test_fn, backend_config={}, seed=SEED):
    train_circs = [c.lambdify(*vocab) for c in train_circs]
    test_circs = [c.lambdify(*vocab) for c in test_circs]

    n_train = len(train_circs)
    n_test = len(test_circs)
    gram_train = np.zeros((n_train, n_train))
    gram_test = np.zeros((n_test, n_train))

    accuracies = []
    def callback_fn(params, **kwargs):
        print('Testing SVM')
        bound_train_circs = [c(*params) for c in train_circs]
        bound_test_circs = [c(*params) for c in test_circs]

        for i in range(n_train):
            for j in range(i+1):
                sim = sim_fn(
                    bound_train_circs[i], bound_train_circs[j], 
                    backend_config=backend_config, seed=seed
                )
                gram_train[i, j] = sim
                gram_train[j, i] = sim

        for r in range(gram_test.shape[0]):
            for c in range(gram_test.shape[1]):
                gram_test[r, c] = sim_fn(
                    bound_train_circs[c], bound_test_circs[r], 
                    backend_config=backend_config, seed=seed
                )

        svc = SVC(kernel="precomputed")
        svc.fit(gram_train, train_labels[:, 0])
        score = svc.score(gram_test, test_labels[:, 0])
        accuracies.append(score)
        return score

    return callback_fn, accuracies
    
def make_kernel_fn(params, param_vals, sim_fn=swap_test_fn, parser=DEFAULT_PARSER, ansatz=DEFAULT_ANSATZ, backend_config={}, seed=SEED):    
    def kernel_fn(x, y):
        if type(x) is str:
            x = [x]
        if type(y) is str:
            y = [y]

        # build circuits
        ts = time()
        x_circs = build_circuits(x, parser=parser, ansatz=ansatz)
        y_circs = build_circuits(y, parser=parser, ansatz=ansatz)

        x_circs = [c.lambdify(*params)(*param_vals) for c in x_circs]
        y_circs = [c.lambdify(*params)(*param_vals) for c in y_circs]
        timestamp(time(), ts, 'to prep circs')

        ts = time()
        kernel_mat = np.zeros((len(x), len(y)))
        for r in range(len(x)):
            for c in range(len(y)):
                kernel_mat[r, c] = sim_fn(
                    x_circs[r], y_circs[c], 
                    backend_config=backend_config, seed=seed
                )

        timestamp(time(), ts, 'to build matrix')
        return kernel_mat

    return kernel_fn

def make_circ_kernel_fn(sim_fn=swap_test_fn, backend_config={}, seed=SEED):
    def kernel_fn(x_circs, y_circs):      
        nx = len(x_circs)
        ny = len(y_circs)
        kernel_mat = np.zeros((nx, ny))
        for r in range(nx):
            for c in range(ny):
                kernel_mat[r, c] = sim_fn(
                    x_circs[r], y_circs[c], 
                    backend_config=backend_config, seed=seed
                )
        return kernel_mat
    return kernel_fn

def build_gram_matrices_from_circ(train_circs, test_circs, sim_fn=swap_test_fn, backend_config={}, seed=SEED):
    ts = time()
    print('Building training Gram matrix...')
    n_train = len(train_circs)
    gram_train = np.zeros((n_train, n_train))
    for i in range(n_train):
        for j in range(i+1):
            sim = sim_fn(
                train_circs[i], train_circs[j],
                backend_config=backend_config, seed=seed
            )
            gram_train[i, j] = sim
            gram_train[j, i] = sim
    timestamp(time(), ts)

    ts = time()
    print('Building testing Gram matrix...')
    n_test = len(test_circs)
    gram_test = np.zeros((n_test, n_train))
    for r in range(gram_test.shape[0]):
        for c in range(gram_test.shape[1]):
            gram_test[r, c] = sim_fn(
                train_circs[c], test_circs[r],
                backend_config=backend_config, seed=seed
            )
    timestamp(time(), ts)

    return gram_train, gram_test

def build_gram_matrices(x_train, x_test, params, param_vals, sim_fn=swap_test_fn, parser=DEFAULT_PARSER, ansatz=DEFAULT_ANSATZ, backend_config={}, seed=SEED):
    ts = time()
    print('Building circuits...')
    train_circs = build_circuits(x_train, parser=parser, ansatz=ansatz)
    test_circs = build_circuits(x_test, parser=parser, ansatz=ansatz)

    train_circs = [c.lambdify(*params)(*param_vals) for c in train_circs]
    test_circs = [c.lambdify(*params)(*param_vals) for c in test_circs]
    timestamp(time(), ts)

    return build_gram_matrices_from_circ(
        train_circs, test_circs, sim_fn=sim_fn, 
        backend_config=backend_config, seed=seed
    )

def display_grams(gram_train, gram_test, save_name=''):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    axs[0].imshow(np.asmatrix(gram_train), interpolation="nearest", origin="upper", cmap="Blues")
    axs[0].set_title("Training kernel matrix")
    axs[1].imshow(np.asmatrix(gram_test), interpolation="nearest", origin="upper", cmap="Reds")
    axs[1].set_title("Testing kernel matrix")

    if save_name:
        plt.savefig(f'{save_name}.pdf', dpi=1200, bbox_inches="tight")
    plt.show()

