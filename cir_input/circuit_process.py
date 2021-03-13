import numpy as np
from TDD.TDD import Index,Find_Or_Add_Unique_table,get_int_key,set_index_order,diag_matrix_2_TDD2
from TDD.TDD import Ini_TDD,TDD,Single_qubit_gate_2TDD,diag_matrix_2_TDD,cnot_2_TDD,Slicing,contraction,TDD_2_matrix,Two_qubit_gate_2TDD
from TDD.TDD_show import TDD_show
import networkx as nx
import time
from networkx.algorithms.approximation.treewidth import treewidth_min_degree,treewidth_min_fill_in

def get_real_qubit_num(cir):
    qubit_num=0
    for k in cir.nodes():
        temp=max(cir.nodes[k]['operation'].involve_qubits_list)
        qubit_num=max(qubit_num,temp)
    qubit_num+=1
    return qubit_num

def get_gates_number(cir):
    return len(cir.nodes)


def is_diagonal(U):
    i, j = np.nonzero(U)
    return np.all(i == j)

# do not use hyper edge for noisy gate
def get_tensor_index(cir, num_qubit, connect_in_and_out=False):
    """return a dict that link every quantum gate to the corresponding index"""
    qubits_index = [0] * num_qubit
    node_2_index = dict()
    index_2_node = dict()
    index_set = set()
    hyper_index = dict()
    for k in cir.nodes():
        operation = cir.nodes[k]['operation']
        nam = operation.name
        gate_qubits = len(operation.involve_qubits_list)
        if nam == 'CX':
            q = [operation.control_qubit, operation.target_qubit]
            var_con = 'x' + str(q[0]) + str(0) + str(qubits_index[q[0]])
            var_tar_in = 'x' + str(q[1]) + str(0) + str(qubits_index[q[1]])
            var_tar_out = 'x' + str(q[1]) + str(0) + str(qubits_index[q[1]] + 1)
            if not var_con in hyper_index:
                hyper_index[var_con] = 0
            if not var_tar_in in hyper_index:
                hyper_index[var_tar_in] = 0
            if not var_tar_out in hyper_index:
                hyper_index[var_tar_out] = 0
            node_2_index[k] = [Index(var_con, hyper_index[var_con]), Index(var_con, hyper_index[var_con] + 1),
                               Index(var_con, hyper_index[var_con] + 2), Index(var_tar_in, hyper_index[var_tar_in]),
                               Index(var_tar_out, hyper_index[var_tar_out])]
            hyper_index[var_con] += 2
            qubits_index[q[1]] += 1
            index_set.add(var_con)
            index_set.add(var_tar_in)
            index_set.add(var_tar_out)
            if not var_con in index_2_node:
                index_2_node[var_con] = [k]
            else:
                index_2_node[var_con].append(k)
            if not var_tar_in in index_2_node:
                index_2_node[var_tar_in] = [k]
            else:
                index_2_node[var_tar_in].append(k)
            if not var_tar_out in index_2_node:
                index_2_node[var_tar_out] = [k]
            else:
                index_2_node[var_tar_out].append(k)
            continue

        if gate_qubits == 1:
            q = operation.involve_qubits[0]
            var_in = 'x' + str(q) + str(0) + str(qubits_index[q])
            var_out = 'x' + str(q) + str(0) + str(qubits_index[q] + 1)
            if not var_in in hyper_index:
                hyper_index[var_in] = 0
            if not var_out in hyper_index:
                hyper_index[var_out] = 0
            if is_diagonal(operation.u_matrix) and nam != 'noisy':
                node_2_index[k] = [Index(var_in, hyper_index[var_in]), Index(var_in, hyper_index[var_in] + 1)]
                hyper_index[var_in] += 1
                index_set.add(var_in)
                if not var_in in index_2_node:
                    index_2_node[var_in] = [k]
                else:
                    index_2_node[var_in].append(k)
                continue
            else:
                node_2_index[k] = [Index(var_in, hyper_index[var_in]), Index(var_out, hyper_index[var_out])]
                qubits_index[q] += 1
                index_set.add(var_in)
                index_set.add(var_out)
                if not var_in in index_2_node:
                    index_2_node[var_in] = [k]
                else:
                    index_2_node[var_in].append(k)
                if not var_out in index_2_node:
                    index_2_node[var_out] = [k]
                else:
                    index_2_node[var_out].append(k)
                continue
        if gate_qubits == 2:
            q = operation.involve_qubits_list
            var_con_in = 'x' + str(q[0]) + str(0) + str(qubits_index[q[0]])
            var_con_out = 'x' + str(q[0]) + str(0) + str(qubits_index[q[0]] + 1)
            var_tar_in = 'x' + str(q[1]) + str(0) + str(qubits_index[q[1]])
            var_tar_out = 'x' + str(q[1]) + str(0) + str(qubits_index[q[1]] + 1)
            if not var_con_in in hyper_index:
                hyper_index[var_con_in] = 0
            if not var_con_out in hyper_index:
                hyper_index[var_con_out] = 0
            if not var_tar_in in hyper_index:
                hyper_index[var_tar_in] = 0
            if not var_tar_out in hyper_index:
                hyper_index[var_tar_out] = 0

            if is_diagonal(operation.u_matrix):
                node_2_index[k] = [Index(var_con_in, hyper_index[var_con_in]),
                                   Index(var_con_in, hyper_index[var_con_in] + 1),
                                   Index(var_tar_in, hyper_index[var_tar_in]),
                                   Index(var_tar_in, hyper_index[var_tar_in] + 1)]
                hyper_index[var_con_in] += 1
                hyper_index[var_tar_in] += 1
                index_set.add(var_tar_in)
                index_set.add(var_con_in)
                if not var_tar_in in index_2_node:
                    index_2_node[var_tar_in] = [k]
                else:
                    index_2_node[var_tar_in].append(k)
                if not var_con_in in index_2_node:
                    index_2_node[var_con_in] = [k]
                else:
                    index_2_node[var_con_in].append(k)
                continue
            else:
                node_2_index[k] = [Index(var_con_in, hyper_index[var_con_in]),
                                   Index(var_con_out, hyper_index[var_con_out]),
                                   Index(var_tar_in, hyper_index[var_tar_in]),
                                   Index(var_tar_out, hyper_index[var_tar_out])]
                qubits_index[q[0]] += 1
                qubits_index[q[1]] += 1
                index_set.add(var_tar_in)
                index_set.add(var_tar_out)
                index_set.add(var_con_in)
                index_set.add(var_con_out)
                if not var_tar_in in index_2_node:
                    index_2_node[var_tar_in] = [k]
                else:
                    index_2_node[var_tar_in].append(k)
                if not var_tar_out in index_2_node:
                    index_2_node[var_tar_out] = [k]
                else:
                    index_2_node[var_tar_out].append(k)
                if not var_con_in in index_2_node:
                    index_2_node[var_con_in] = [k]
                else:
                    index_2_node[var_con_in].append(k)
                if not var_con_out in index_2_node:
                    index_2_node[var_con_out] = [k]
                else:
                    index_2_node[var_con_out].append(k)
                continue

    for k in range(num_qubit):
        last1 = 'x' + str(k) + str(0) + str(qubits_index[k])
        new1 = 'y' + str(k)
        last2 = 'x' + str(k) + str(0) + str(0)
        new2 = 'x' + str(k)
        if qubits_index[k] != 0:
            index_set.remove(last1)
            index_set.add(new1)
            index_set.remove(last2)
            index_set.add(new2)
            index_2_node[new1] = index_2_node[last1]
            index_2_node[new2] = index_2_node[last2]
            index_2_node.pop(last1)
            index_2_node.pop(last2)
        elif last1 in index_set:
            index_set.remove(last1)
            index_set.add(new1)
            index_2_node[new1] = index_2_node[last1]
            index_2_node.pop(last1)
        for m in node_2_index:
            node_2_index[m] = [Index(new1, item.idx) if item.key == last1 else item for item in node_2_index[m]]
            node_2_index[m] = [Index(new2, item.idx) if item.key == last2 else item for item in node_2_index[m]]

    if connect_in_and_out:
        for k in range(num_qubit):
            if qubits_index[k] != 0:
                node_2_index['q' + str(k)] = [Index('x' + str(k), 0), Index('y' + str(k), hyper_index[
                    'x' + str(k) + str(0) + str(qubits_index[k])])]
                if not 'x' + str(k) in index_2_node:
                    index_2_node['x' + str(k)] = ['q' + str(k)]
                else:
                    index_2_node['x' + str(k)].append('q' + str(k))
                if not 'y' + str(k) in index_2_node:
                    index_2_node['y' + str(k)] = ['q' + str(k)]
                else:
                    index_2_node['y' + str(k)].append('q' + str(k))
            elif 'y' + str(k) in index_set:
                node_2_index['q' + str(k)] = [Index('y' + str(k), 0), Index('y' + str(k), hyper_index[
                    'x' + str(k) + str(0) + str(qubits_index[k])])]
                if not 'y' + str(k) in index_2_node:
                    index_2_node['y' + str(k)] = ['q' + str(k)]
                else:
                    index_2_node['y' + str(k)].append('q' + str(k))
            else:
                node_2_index['q' + str(k)] = []
                index_set.add('y' + str(k))
                index_2_node['y' + str(k)] = ['q' + str(k)]

    return node_2_index, index_2_node, index_set


def get_tree_decomposition(cir, num_qubit, index_set, index_2_node, node_2_index, connect_in_and_out=False):
    lin_graph = nx.Graph()
    lin_graph.add_nodes_from(index_set)
    for k in cir.nodes():
        operation = cir.nodes[k]['operation']
        nam = operation.name
        gate_qubits = len(operation.involve_qubits_list)
        if nam == 'CX':
            lin_graph.add_edge(node_2_index[k][0].key, node_2_index[k][3].key)
            lin_graph.add_edge(node_2_index[k][0].key, node_2_index[k][4].key)
            lin_graph.add_edge(node_2_index[k][3].key, node_2_index[k][4].key)
            continue
        if gate_qubits == 2:
            if is_diagonal(operation.u_matrix):
                lin_graph.add_edge(node_2_index[k][0].key, node_2_index[k][2].key)
            else:
                lin_graph.add_edge(node_2_index[k][0].key, node_2_index[k][1].key)
                lin_graph.add_edge(node_2_index[k][0].key, node_2_index[k][2].key)
                lin_graph.add_edge(node_2_index[k][0].key, node_2_index[k][3].key)
                lin_graph.add_edge(node_2_index[k][1].key, node_2_index[k][2].key)
                lin_graph.add_edge(node_2_index[k][1].key, node_2_index[k][3].key)
                lin_graph.add_edge(node_2_index[k][2].key, node_2_index[k][3].key)
            continue
        if node_2_index[k][0].key != node_2_index[k][1].key:
            lin_graph.add_edge(node_2_index[k][0].key, node_2_index[k][1].key)

    if connect_in_and_out:
        for k in range(num_qubit):
            if len(node_2_index['q' + str(k)]) == 0:
                continue
            if node_2_index['q' + str(k)][0].key != node_2_index['q' + str(k)][1].key:
                lin_graph.add_edge(node_2_index['q' + str(k)][0].key, node_2_index['q' + str(k)][1].key)

    tree_width, de_graph = treewidth_min_fill_in(lin_graph)
    #     print('The treewidth is',tree_width)
    return de_graph, tree_width


def find_contraction_index(tree_decomposition):
    idx = None
    if len(tree_decomposition.nodes) == 1:
        nod = [k for k in tree_decomposition.nodes][0]
        if len(nod) != 0:
            idx = [idx for idx in nod][0]
            nod_temp = set(nod)
            nod_temp.remove(idx)
            tree_decomposition.add_node(frozenset(nod_temp))
            tree_decomposition.remove_node(nod)
        return idx
    nod = 0
    for k in tree_decomposition.nodes:
        if nx.degree(tree_decomposition)[k] == 1:
            nod = k
            break

    neib = [k for k in tree_decomposition.neighbors(nod)][0]
    for k in nod:
        if not k in neib:
            idx = k
            break
    if idx:
        nod_temp = set(nod)
        nod_temp.remove(idx)
        tree_decomposition.remove_node(nod)
        if frozenset(nod_temp) != neib:
            tree_decomposition.add_node(frozenset(nod_temp))
            tree_decomposition.add_edge(frozenset(nod_temp), neib)
        return idx
    else:
        tree_decomposition.remove_node(nod)
        return find_contraction_index(tree_decomposition)


def contract_an_index(cir, num_qubit, cont_index, index_2_node, node_2_index, computed_tdd_list):
    temp_tdd, max_node_num = get_tdd_of_a_part_circuit(index_2_node[cont_index], list(range(num_qubit)), cir,
                                                       node_2_index)

    for node in index_2_node[cont_index]:
        for idx in node_2_index[node]:
            if idx.key != cont_index:
                if node in index_2_node[idx.key]:
                    index_2_node[idx.key].remove(node)

    index_2_node.pop(cont_index)

    temp_computed_tdd_list = []
    for tdd in computed_tdd_list:
        tdd_idx_out = [k.key for k in tdd.index_set]

        if cont_index in tdd_idx_out:
            temp_tdd = contraction(tdd, temp_tdd)
            max_node_num = max(max_node_num,temp_tdd.node_number())
        else:
            temp_computed_tdd_list.append(tdd)

    temp_computed_tdd_list.append(temp_tdd)
    computed_tdd_list = temp_computed_tdd_list

    return computed_tdd_list, max_node_num

def get_tdd(operation,var_list,involve_qubits):
    """get the TDD of the correct part of quantum gate"""
    nam=operation.name
    gate_qubits = len(operation.involve_qubits_list)
    if nam =='CX':
        if operation.control_qubit in involve_qubits and operation.target_qubit in involve_qubits:
            return cnot_2_TDD(var_list,case=1)
        if operation.control_qubit in involve_qubits and not operation.target_qubit in involve_qubits:
            return cnot_2_TDD(var_list,case=2)
        else:
            return cnot_2_TDD(var_list,case=3)
    if gate_qubits ==1:
        if is_diagonal(operation.u_matrix) and nam != 'noisy':
            return diag_matrix_2_TDD(operation.u_matrix,var_list)
        else:
            return Single_qubit_gate_2TDD(operation.u_matrix,var_list)
    if gate_qubits == 2:
        if is_diagonal(operation.u_matrix):
            return diag_matrix_2_TDD2(operation.u_matrix,var_list)
        else:
            return Two_qubit_gate_2TDD(operation.u_matrix,var_list)
        
def get_tdd_of_a_part_circuit(involve_nodes,involve_qubits,cir,node_2_index):
    """get the TDD of a part of circuit"""
#     print('involve_nodes',involve_nodes)
    compute_time = time.time()
    node=Find_Or_Add_Unique_table(1,0,0,None,None)
    tdd=TDD(node)
    max_node_num = 0
    for k in involve_nodes:
        if isinstance(k,str) and k[0]=='q':
            if len(node_2_index[k])==0:
                temp_tdd=TDD(node)
                temp_tdd.weight=2
            elif node_2_index[k][0].key!=node_2_index[k][1].key:
                temp_tdd=Single_qubit_gate_2TDD(np.eye(2),node_2_index[k])
            else:
                temp_tdd=diag_matrix_2_TDD(np.eye(2),node_2_index[k])
        else:
            temp_tdd=get_tdd(cir.nodes[k]['operation'],node_2_index[k],involve_qubits)
        tdd=contraction(tdd,temp_tdd)
    return tdd,max_node_num
