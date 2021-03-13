import random
import numpy as np
from TDD.TDD import Index,Ini_TDD,TDD,Find_Or_Add_Unique_table,contraction
from TDD.TDD_show import TDD_show
from cir_input.qasm import CreateCircuitFromQASM
from cir_input.circuit_DG import CreateDGfromQASMfile
from cir_input.gate_operation import OperationCNOT, OperationSingle,OperationU,OperationTwo
from cir_input.circuit_process import get_real_qubit_num,get_gates_number,get_tensor_index,get_tree_decomposition,find_contraction_index,\
    contract_an_index

import time
import datetime
import copy
import pandas as pd
from itertools import permutations
from func_timeout import func_set_timeout
import func_timeout
import networkx as nx

@func_set_timeout(3600)
def Simulation_with_tree_decomposition(cir, num_qubit, scheme=1, connect_in_and_out=False, ini_tdd=True):
    """Simulate a circuit with TDD;
    cir is a dag circuit;
    num_qubit is the real qubit number of the circuit;
    scheme = 1
    when you want to calculate the trace of a circuit connect_in_and_out=False,
    else it simulate the circuit itself
    return the result tdd and other corresponding information
    """
    if ini_tdd:
        var_order = []
        gate_num = len(cir.nodes)
        for k in range(num_qubit - 1, -1, -1):
            var_order.append('x' + str(k))
            for j in range(gate_num):
                var_order.append('x' + str(k) + str(0) + str(j))
            var_order.append('y' + str(k))

        Ini_TDD(var_order)

    max_node_num = 0
    node = Find_Or_Add_Unique_table(1, 0, 0, None, None)
    tdd = TDD(node)

    node_2_index, index_2_node, index_set = get_tensor_index(cir, num_qubit, connect_in_and_out)

    if scheme == 1:
        decom_tree, tree_width = get_tree_decomposition(cir, num_qubit, index_set, index_2_node, node_2_index,
                                                        connect_in_and_out)
        #         print('tree_width',tree_width)
        cont_index = find_contraction_index(decom_tree)
        computed_tdd_list = []
        while cont_index:
            computed_tdd_list, node_num1 = contract_an_index(cir, num_qubit, cont_index, index_2_node, node_2_index,
                                                             computed_tdd_list)
            max_node_num = max(max_node_num, node_num1)
            #             print(cont_index)
            cont_index = find_contraction_index(decom_tree)

        for temp_tdd in computed_tdd_list:
            tdd = contraction(tdd, temp_tdd)
            max_node_num=max(max_node_num,tdd.node_number())
        return tdd, max_node_num, tree_width

def Simulation_with_time_out(cir, num_qubit,scheme=1,connect_in_and_out=False,ini_tdd=True):
    try:
        return Simulation_with_tree_decomposition(cir, num_qubit,scheme,connect_in_and_out,ini_tdd)
    except:
        print('Time out!')
        node = Find_Or_Add_Unique_table(1,0,0,None,None)
        tdd = TDD(node)
        return tdd,0,0

def noisy_model(p,model=-1):
#   define the noisy model
    if model==-1:
        model = random.randint(1,3)
    if model == 1:#bit flip
        E0=np.sqrt(1-p)*np.eye(2)
        E1=np.sqrt(p)*np.array([[0,1],[1,0]])
        M=np.kron(E0,E0)+np.kron(E1,E1)
        return model,[E0,E1],M
    if model == 2:#phase flip
        E0=np.sqrt(1-p)*np.eye(2)
        E1=np.sqrt(p)*np.array([[1,0],[0,-1]])
        M=np.kron(E0,E0)+np.kron(E1,E1)
        return model,[E0,E1],M
    if model == 3:#phase flip
        E0 = np.sqrt(1-p)*np.eye(2)
        E1 = np.sqrt(p/3)*np.array([[0,1],[1,0]])
        E2 = np.sqrt(p/3)*np.array([[0,-1j],[1j,0]])
        E3 = np.sqrt(p/3)*np.array([[1,0],[0,-1]])
        M=np.kron(E0,E0)+np.kron(E1,E1)+np.kron(E2,E2.conjugate())+np.kron(E3,E3)
        return model,[E0,E1,E2,E3],M

def assign_noisy(cir,the_noisy_position,noise_case,p):
    """
    add noises into the circuit
    """
    noisy_cir=nx.DiGraph()
    gate_num=0
    noisy_gate_num = 0
    num_qubit = get_real_qubit_num(cir)
    the_noisy_model = dict()
    for k in cir.nodes:
        operation = cir.nodes[k]['operation']
        nam=operation.name
        q=operation.involve_qubits_list
        noisy_cir.add_node(gate_num)
        noisy_cir.nodes[gate_num]['operation'] = operation
        gate_num+=1
        if k in the_noisy_position:
            model,E,M = noisy_model(p,model=3)
            q=[q[0]]
            for noisy_qubit in q:
                new_operation = OperationU([noisy_qubit],'noisy',E[noise_case[noisy_gate_num]])
                noisy_cir.add_node(gate_num)
                noisy_cir.nodes[gate_num]['operation'] = new_operation
                gate_num += 1
                noisy_gate_num +=1
                the_noisy_model[k]=model
    return noisy_cir,noisy_gate_num

def get_miter(cir,noisy_cir):
    temp_cir = noisy_cir.copy()
    gate_num = len(noisy_cir.nodes)
    for k in range(len(cir.nodes)-1,-1,-1):
        operation = cir.nodes[k]['operation']
        temp_cir.add_node(gate_num)
        temp_cir.nodes[gate_num]['operation'] = copy.copy(operation)
        if temp_cir.nodes[gate_num]['operation'].name !='CX':
            temp_cir.nodes[gate_num]['operation'].u_matrix=temp_cir.nodes[gate_num]['operation'].u_matrix.conjugate().T
        gate_num+=1

    return temp_cir

def get_base_change(n,x):
    a=[0,1,2,3,4,5,6,7,8,9,'A','b','C','D','E','F']
    b=[]
    while True:
        s=n//x
        y=n%x
        b=b+[y]
        if s==0:
            break
        n=s
    b.reverse()
    return b

def get_error_position(n,k):
    """n is the gates number, and k in the noisy number"""
    pos=[]
    while len(pos)< k:
        temp_pos=np.random.randint(0,n)
        if not temp_pos in pos:
            pos.append(temp_pos)
    return pos

if __name__=="__main__":
    path = 'Benchmarks/'
    file_name = 'bv_n6.qasm'

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

    noisy_gate_num = np.random.randint(1, min(7, original_gate_num))
    print('noisy_num:', noisy_gate_num)
    noisy_position = get_error_position(original_gate_num, noisy_gate_num)
    print(noisy_position)
    print('-------------')

    t_start = time.time()
    var_order = []
    for k in range(num_qubit - 1, -1, -1):
        var_order.append('x' + str(k))
        for j in range(3 * gate_num):
            var_order.append('x' + str(k) + str(0) + str(j))
        var_order.append('y' + str(k))
    Ini_TDD(var_order)

    the_sum_of_trace_square = 0
    max_node_num_total = 0
    for cir_num in range(4 ** noisy_gate_num):
        nois = get_base_change(cir_num, 4)
        if len(nois) < noisy_gate_num:
            nois = [0] * (noisy_gate_num - len(nois)) + nois
        #     print(nois)
        noisy_cir, _ = assign_noisy(dag_cir1, noisy_position, nois, 0.001)
        miter_cir = get_miter(dag_cir1, noisy_cir)
        t_start2 = time.time()
        tdd, max_node_num, tree_width = Simulation_with_tree_decomposition(miter_cir, num_qubit, scheme=1, connect_in_and_out=True,
                                                                ini_tdd=False)
        run_time2 = time.time() - t_start2
        the_trace = tdd.weight
        the_sum_of_trace_square += abs(the_trace) ** 2
        max_node_num_total = max(max_node_num_total, max_node_num)
        if time.time() - t_start > 3600:
            print('time out')
            break
    print('-------------')
    run_time = time.time() - t_start
    print('total_run_time:', run_time)
    print('tree_width:', tree_width)
    print('max_node_num:', max_node_num_total)
    d = 2 ** num_qubit
    print('The fidelity is:', the_sum_of_trace_square / d ** 2)
