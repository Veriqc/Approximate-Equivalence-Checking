#To calculate the fidelity of a noisy circuit with its ideal circuit

import random
import numpy as np
from cir_input.qasm import CreateCircuitFromQASM
from cir_input.circuit_DG import CreateDGfromQASMfile
import time
import datetime
import copy
import pandas as pd
from qiskit.quantum_info import Operator, process_fidelity
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors import kraus_error,mixed_unitary_error,QuantumError
from qiskit import *
from cir_input.circuit_process import get_real_qubit_num,get_gates_number
from func_timeout import func_set_timeout

def dag_2_circuit(dag_cir,noisy_position=[]):
    """transfer a dag circuit to a quantum circuit used by qiskit
    """
    num_qubit = get_real_qubit_num(dag_cir)
    cir = QuantumCircuit(num_qubit)
    for k in dag_cir.nodes():
        if k in noisy_position:
            continue
        operation = dag_cir.nodes[k]['operation']
        if operation.name == 'CX':
            cir.cx(operation.control_qubit,operation.target_qubit)
        else:
            cir.unitary(Operator(operation.u_matrix), operation.involve_qubits_list)
    return cir

def get_error_channel(dag_cir,noisy_position,p):
    """to get the error channel of the noisy circuit
    """
    t_start=time.time()
    num_qubit = get_real_qubit_num(dag_cir)
    noise_ops=[([{"name": 'unitary', "qubits": [num_qubit-1], 'params': [np.eye(2)]}], 1)]
    error=QuantumError(noise_ops,number_of_qubits=num_qubit).to_quantumchannel()
    for k in dag_cir.nodes:
        operation = dag_cir.nodes[k]['operation']
        nam=operation.name
        q=operation.involve_qubits_list
        if nam!='CX':
            U=operation.u_matrix
        else:
            if q[0]<q[1]:
                U=np.array([[1,0,0,0],[0,0,0,1],[0,0,1,0],[0,1,0,0]])
            else:
                U=np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]])
        noise_ops=[([{"name": 'unitary', "qubits": q, 'params': [U]}], 1)]
        temp_error=QuantumError(noise_ops,number_of_qubits=num_qubit).to_quantumchannel()
        error=error.compose(temp_error)
        if k in noisy_position:
            q=[q[0]]
            for q0 in q:
                noise_ops=[([{"name": 'id', "qubits": [q0]}], 1-p),([{"name": 'x', "qubits": [q0]}], p/3),([{"name": 'y', "qubits": [q0]}], p/3),([{"name": 'z', "qubits": [q0]}], p/3)]
                temp_error=QuantumError(noise_ops,number_of_qubits=num_qubit).to_quantumchannel()
                error=error.compose(temp_error)
    return error

@func_set_timeout(3600)
def get_fidelity_with_qiskit(dag_cir,noisy_position):
    cir=dag_2_circuit(dag_cir)
    U=Operator(cir)
    channel=get_error_channel(dag_cir,noisy_position,0.001)
    fide=process_fidelity(channel,U)
    return fide

def Simulation_with_time_out(dag_cir,noisy_position):
    try:
        return get_fidelity_with_qiskit(dag_cir,noisy_position)
    except:
        print('Time out!')
        return 0

def get_error_position(n,k):
    """n is the gates number, and k in the noisy number"""
    pos=[]
    while len(pos)< k:
        temp_pos=np.random.randint(0,n)
        if not temp_pos in pos:
            pos.append(temp_pos)
    return pos

if __name__=='__main__':
    path = 'Benchmarks/'
    file_name = 'bv_n3.qasm'

    cir1, res1 = CreateDGfromQASMfile(file_name, path, flag_single=True, flag_interaction=True)

    dag_cir1 = res1[0]

    print('circuit:', file_name)
    num_qubit = get_real_qubit_num(dag_cir1)
    print('qubits:', num_qubit)
    gate_num = get_gates_number(dag_cir1)
    print('gates number:', gate_num)
    time_now = datetime.datetime.now()
    print(time_now.strftime('%m.%d-%H:%M:%S'))

    original_gate_num = get_gates_number(dag_cir1)

    noisy_gate_num = np.random.randint(1, min(15, original_gate_num))
    print('noisy_num:', noisy_gate_num)

    noisy_position = get_error_position(original_gate_num, noisy_gate_num)

    print(noisy_position)

    t_start = time.time()

    fide = Simulation_with_time_out(dag_cir1, noisy_position)

    run_time = time.time() - t_start
    print('run_time:', run_time)
    print(fide)
