"""
Utility function used over all the schedulers
"""


def assemble_timesteps_from_gates(number_of_nodes, gates):
    """
    Compute the depth given a circuit

    :param number_of_nodes: int, number of qubits
    :param gates: [(q1, q2)], the list of gates
    :return: int, number of timesteps, i.e. circuit depth
    """

    d = [0] * number_of_nodes
    timesteps = []

    for (gate_type, n1, n2) in gates:
        d_max = max(d[n1], d[n2])

        new_depth = d_max + 1

        d[n1] = new_depth
        d[n2] = new_depth

        if new_depth > len(timesteps):
            timesteps.append([(gate_type, n1, n2)])
        else:
            timesteps[new_depth-1].append((gate_type, n1, n2))

    return timesteps
