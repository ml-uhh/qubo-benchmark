#from azure.quantum.qiskit import AzureQuantumProvider
from qiskit_ionq import IonQProvider
import qiskit
from qiskit.circuit import Parameter
#from qiskit.utils import QuantumInstance
#from qiskit.opflow import PauliSumOp
from qiskit_algorithms import QAOA
from qiskit_optimization.algorithms import MinimumEigenOptimizer, RecursiveMinimumEigenOptimizer
from qiskit_optimization.problems import QuadraticProgram
from qiskit_algorithms.optimizers import ADAM, COBYLA, NELDER_MEAD, SLSQP
from qiskit.quantum_info import Pauli, SparsePauliOp
from qiskit.circuit.library import QAOAAnsatz

import numpy as np
import datetime
import json
import itertools

from qiskit.primitives import BackendSampler as Sampler
from qiskit_ibm_runtime import QiskitRuntimeService
token = 'f4e4046404e2f70c1a76f9b6b6f5d497ae2540d4d8852b29d9d246e18d74ca63827159f5781c8e7dbbd00f2cdc76e2a408ddf4da7446b42d48ca622923fb3820'

class MyJsonEncoder(json.JSONEncoder):
    """The class is used to transform NumPy objects and datetime objects into a format that can be fed to json"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, datetime.timedelta):
            return str(obj)
        return super(MyJsonEncoder, self).default(obj)


class DwaveTrace:
    """Trace the execution of the quantum annealing"""

    def __init__(self, path, print=False) -> None:
        """Constructor.
        :param path: relative or absolute path to save the files, including the first part of the file name. 
            Passing './dwave' results in the creation of two files dwave_<timestamp>.json and dwave_<timestamp>.txt 
            in the directory of the current execution.
        :param print: true if the tool is allowed to print on stdout.
        """
        self.info = {}
        self.info_text = ""
        self.print = print
        self.path = path

    def add(self, section, subsection, key, value):
        """Add new information
        :param section: primary tag of the information (str)
        :param subsection: secondary tag of the information, optional (str or None)
        :param key: name of the information (str)
        :param value: content of the information (obj)
        :return None
        """
        if type(value) == np.ndarray:
            value = value.tolist()

        # save for json
        if section not in self.info:
            self.info[section] = {}
        if subsection is not None:
            if subsection not in self.info[section]:
                self.info[section][subsection] = {}
            self.info[section][subsection][key] = value
        else:
            self.info[section][key] = value
        
        # save for text
        if subsection is not None:
            this_text = f"{section:40s} :: {subsection:30s} :: {key:40s} = {value}\n"
        else:
            this_text = f"{section:40s} :: {'':30s} :: {key:40s} = {value}\n"
        self.info_text += this_text
        if self.print:
            print(this_text, end='', flush=True)

    def save(self):
        """Save to json and txt"""
        json.dump(self.info, open(f"{self.path}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.json", "w"), cls=MyJsonEncoder)
        open(f"{self.path}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.txt", "w").writelines(self.info_text)


def create_pauli_term(i, j, N):
    """
    Create a Pauli ZZ term at the given indices i, j of size N
    :param i:
    :param j:
    :param N:
    :return:
    """
    return ''.join(['Z' if k in [i, j] else 'I' for k in range(N)])


def convert_ising_to_operator(h, J):
    """

    :param h:
    :param J:
    :return:
    """
    N = len(h)
    pauli_list = []
    for i, h_coeff in enumerate(h):
        pauli_term = (create_pauli_term(i, i, N), h_coeff)
        pauli_list.append(pauli_term)
    for ((start, end), j_coeff) in J.items():
        pauli_term = (create_pauli_term(start, end, N), j_coeff)
        pauli_list.append(pauli_term)
    pauli_operator = SparsePauliOp.from_list(pauli_list)#PauliSumOp.from_list(pauli_list)
    return pauli_operator, pauli_list


def get_ising_cost(h, J, vars):
    """

    :param h:
    :param J:
    :param vars:
    :return:
    """
    cost = 0
    for x, hcoeff in zip(vars, h):
        cost += x * hcoeff
    for ((s,e), jcoeff) in J.items():
        cost += vars[s] * vars[e] * jcoeff
    return cost


def run_qaoa(vendor, azure_subscription_id, azure_location, azure_backend, h, J,
             optimizer, maxiter, reps, n_shots, is_recursive_qaoa, save_path):
    """
    Run the quantum annealer
    :param vendor: Hardware vendor, one between 'none', 'azure'
    :param azure_subscription_id: Azure Quantum subscription id
    :param azure_location: Azure Quantum Zone
    :param azure_backend: QPU name, your subscription must have access to such hardware otherwise an error occurs
    :param h: Ising formulation h coefficients (np.nparray[float] of shape (N,))
    :param j: Ising formulation J coefficients (dict[tuple(int,int) -> float])
    :param optimizer: one between 'ADAM', 'COBYLA', 'NELDER_MEAD' or 'SLSQP'
    :param maxiter: Number of iteration of the optimization algorithm
    :param reps: Number of repetitions of the QAOA ansatz
    :param n_shots: Number of sample estimating the expectation value
    :param is_recursive_qaoa: 0 if you want to use plain QAOA, != 0 if you want to use Recursive-QAOA
    :param save_path: Path where to save the file including the first part of the file name
    :return solution to the qubo problem (list[int] of size N) or [] in case of errors.
    """

    trace = DwaveTrace(print=True, path=save_path)

    trace.add('input', None, 'vendor', vendor)
    trace.add('input', 'azure', 'azure_subscription_id', 'hidden')
    trace.add('input', 'azure', 'azure_location', azure_location)
    trace.add('input', 'azure', 'azure_backend', azure_backend)
    trace.add('input', None, 'optimizer', optimizer)
    trace.add('input', None, 'maxiter', maxiter)
    trace.add('input', None, 'reps', reps)
    trace.add('input', None, 'n_shots', n_shots)
    trace.add('input', None, 'is_recursive_qaoa', is_recursive_qaoa)

    # get QPU or simulator device
    if vendor == 'none':
        trace.add('configuration', None, 'device', 'qasm_simulator')
        backend = qiskit.Aer.get_backend('qasm_simulator')
    elif vendor == 'azure':
        provider = QiskitRuntimeService(channel='ibm_quantum', instance='ibm-q/open/main', token=token)#AzureQuantumProvider(resource_id=azure_subscription_id, location=azure_location)
        trace.add('input', 'azure', 'azure_available_backends', [backend for backend in provider.backends()])
        trace.add('configuration', None, 'device', azure_backend)
        backend = provider.backend(azure_backend)

    # convert Ising formulation to Pauli operators
    pauli_operator, pauli_list = convert_ising_to_operator(h, J)
    trace.add('conversion', None, 'pauli_list', pauli_list)
    problem = QuadraticProgram('EPISTASIS_QAOA')
    problem.from_ising(pauli_operator)

    # calculate how many calls will be to the quantum circuit for the QAOA call
    qc_calls_per_qaoa = (1 + 2 * reps) * maxiter + 1  # for each iter 1 global + 1 per parameter; finally a last call
    trace.add('stats', None, 'qc_call_per_qaoa', qc_calls_per_qaoa)
    qc_call_per_rec_qaoa = qc_calls_per_qaoa * (len(h) - 1)  # repeated calls for the given number of variables
    trace.add('stats', None, 'qc_call_per_rec_qaoa', qc_call_per_rec_qaoa)

    # choose optimizer
    optimizer_dict = {'ADAM': ADAM(maxiter=maxiter), 'COBYLA': COBYLA(maxiter=maxiter), 'NELDER_MEAD': NELDER_MEAD(maxiter=maxiter), 'SLSQP': SLSQP(maxiter=maxiter)}
    the_optimizer = optimizer_dict[optimizer]

    # create QAOA callback
    def qaoa_callback(evaluation_count: int, parameters: np.ndarray, eval_mean: float, eval_std: float):
        trace.add('qaoa_trace', f"solver_call_rec_{qaoa_callback.qaoa_solver_calls // qaoa_callback.qc_calls_per_qaoa}_iter_{qaoa_callback.qaoa_solver_calls % qaoa_callback.qc_calls_per_qaoa}", 'eval_count', evaluation_count)
        trace.add('qaoa_trace', f"solver_call_rec_{qaoa_callback.qaoa_solver_calls // qaoa_callback.qc_calls_per_qaoa}_iter_{qaoa_callback.qaoa_solver_calls % qaoa_callback.qc_calls_per_qaoa}", 'eval_parameters', parameters)
        trace.add('qaoa_trace', f"solver_call_rec_{qaoa_callback.qaoa_solver_calls // qaoa_callback.qc_calls_per_qaoa}_iter_{qaoa_callback.qaoa_solver_calls % qaoa_callback.qc_calls_per_qaoa}", 'eval_mean', eval_mean)
        trace.add('qaoa_trace', f"solver_call_rec_{qaoa_callback.qaoa_solver_calls // qaoa_callback.qc_calls_per_qaoa}_iter_{qaoa_callback.qaoa_solver_calls % qaoa_callback.qc_calls_per_qaoa}", 'eval_std', eval_std)
        qaoa_callback.qaoa_solver_calls += 1

    qaoa_callback.qaoa_solver_calls = 0
    qaoa_callback.qc_calls_per_qaoa = qc_calls_per_qaoa

    # create QAOA instance
    #quantum_instance = QuantumInstance(backend, shots=n_shots)
    sampler = Sampler(backend=backend, options={'shots':3})
    qaoa = QAOA(optimizer=the_optimizer, reps=reps, sampler=sampler, callback=qaoa_callback)
    if is_recursive_qaoa == 0:
        internal_optimizer = None
        optimizer = MinimumEigenOptimizer(qaoa)
    else:
        internal_optimizer = MinimumEigenOptimizer(qaoa)
        optimizer = RecursiveMinimumEigenOptimizer(internal_optimizer)

    # save stats about QAOA circuit length size ecc. on current platform
    '''params = [Parameter(f'x{i}') for i in range(2*reps)]
    qaoa_qc = qaoa.construct_circuit(params, pauli_operator)[0]
    qaoa_qc_current = qiskit.transpile(qaoa_qc, backend=backend)
    trace.add('stats', 'current_device', 'size', qaoa_qc_current.size())
    trace.add('stats', 'current_device', 'depth', qaoa_qc_current.depth())
    trace.add('stats', 'current_device', 'num_1qubits_op', np.sum([instr.num_qubits == 1 for instr, _, _ in qaoa_qc_current._data]))
    trace.add('stats', 'current_device', 'num_2qubits_op', np.sum([instr.num_qubits == 2 for instr, _, _ in qaoa_qc_current._data]))
    trace.add('stats', 'current_device', 'circuit_qasm', qaoa_qc_current.assign_parameters({params[i]: i for i in range(len(params))}).qasm())
    '''
    # run QAOA
    result = optimizer.solve(problem)
    trace.add('output', 'raw', 'result_fval', result.fval)
    trace.add('output', 'raw', 'result_x', result.x)

    # rewrite results in the correct format
    result_variables = [-1 if x == 1 else 1 for x in result.x[::-1]]
    trace.add('output', 'postprocessed', 'result_variables', result_variables)
    trace.add('output', 'postprocessed', 'result_cost', get_ising_cost(h, J, result_variables))

    trace.save()
    return result_variables


def test_run_quantum_universal_computer():
    """Run a basic Ising model to check if the library works correctly"""
    # h = [0, 0]
    # J = {(0, 0): 0, (0, 1): 1, (1, 0): 0, (1, 1): 0.5}
    N = 3
    h = np.array([-12.0, 0.5, 0.3])
    J = {(0, 1): 0.2, (0, 2): 0.5, (1, 2): 1.2}
    for vars in itertools.product([-1, 1], repeat=N):
        print(f"vars={str(vars):20s} cost={get_ising_cost(h, J, vars):3.3f}")
    result = run_qaoa('azure', None, None, None, h, J, 'ADAM', 100, 4, 1024, 1, 'azurelog')
    return result

if __name__ == '__main__':
    test_run_quantum_universal_computer()
