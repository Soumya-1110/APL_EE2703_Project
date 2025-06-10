import os
import numpy as np

def evalSpice(filename):
    if not filename or not os.path.isfile(filename): #check for valid file 
        raise FileNotFoundError("Please give the name of a valid SPICE file as input")

    components = []
    nodes = set()
    check_components={}
    within_circuit = False  # set flags

    with open(filename, "r") as file:  #read the file 
        for line in file:
            line = line.strip()

            if line.startswith(".circuit"):  #set the flags if data is within the .circuit block
                within_circuit = True
                continue
            elif line.startswith(".end"):  #break if you hit .end
                break

            if within_circuit and line: #execute code only for circuit elements within the circuit block
                parts = line.split()
                # Extract component details based on the number of parts
                if len(parts) >= 4:
                    name = parts[0]
                    node1 = parts[1]
                    node2 = parts[2]
                    value = parts[3]

                    # Determine the type of the component (V for voltage source, R for resistor, etc.)
                    component_type = name[0].upper()
                    if component_type == "V" or component_type == "I":
                        if parts[3] != "dc" and parts[3] != "ac":
                            raise ValueError("Malformed circuit file")  #raise error if type of source not defined
                        value = parts[4]
                    if (
                        component_type not in ('V', 'I', 'R')
                    ):
                        raise ValueError("Only V, I, R elements are permitted") #raise error if other components detected

                nodes.add(node1)  #make a set of all the nodes
                nodes.add(node2)
                #create the dictionary
                component = {
                    "type": component_type,
                    "name": name,
                    "nodes": tuple(([node1, node2])),
                    "value": value,
                }
                
                components.append(component)
                components = sorted(components, key=lambda component: (component["type"] != "V", component["type"]))

                node_mapping = {"GND": 0}  # GND is always assigned to 0
                node_number = 1

                for node in sorted(nodes):   #map all the nodes to specific numbers
                    if node != "GND":  # Skip GND since it's already assigned
                        node_mapping[node] = node_number
                        node_number += 1
        else:
            if not within_circuit:
                raise ValueError("Malformed circuit file")

        # Count the number of voltage source components
        v_components = sum(1 for component in components if component["type"] == "V")

        # Determine the number of nodes
        num_nodes = len(node_mapping)

        # Create the matrix A
        A_dim = num_nodes - 1 + v_components
        A = np.zeros((A_dim, A_dim), dtype=float)  
        B = np.zeros(A_dim, dtype=float)  # Create a null vector B with dimensions A_dim
        # Fill B with the values of the voltage sources
        if v_components != 0:
            v_index = num_nodes - 1  # Start index for voltage sources in the B vector

            for component in components:
                if component["type"] == "V":
                    B[v_index] = float(component["value"])  # Add the value of the voltage source to B
                    v_index += 1  # Move to the next index for the next voltage source

                if component["type"] == "R":  #Fill A with the values of conductance
                    node1, node2 = component["nodes"]
                    resistance = float(component["value"])
                    conductance = 1 / resistance

                    i = node_mapping[node1]
                    j = node_mapping[node2]

                    # Update diagonal elements (self-node conductance sums)
                    if i != 0:
                        A[i - 1, i - 1] += conductance  # Node i conductance
                    if j != 0:
                        A[j - 1, j - 1] += conductance  # Node j conductance

                    # Update off-diagonal elements (between nodes i and j)
                    if i != 0 and j != 0:
                        A[i - 1, j - 1] -= conductance
                        A[j - 1, i - 1] -= conductance

                if component["type"] == "V":
                    v_index = num_nodes - 1  # Start index for voltage sources in A
                    node_pos, node_neg = component["nodes"]

                    i = node_mapping[node_pos]
                    j = node_mapping[node_neg]

                    # Set up the rows and columns corresponding to the voltage source
                    if i != 0:
                        A[v_index, i - 1] = 1  # Positive terminal
                        A[i - 1, v_index] = 1  # Positive terminal
                    if j != 0:
                        A[v_index, j - 1] = -1  # Negative terminal
                        A[j - 1, v_index] = -1  # Negative terminal

                    v_index += 1

                if component["type"] == "I":
                    node_pos, node_neg = component["nodes"]

                    i = node_mapping[node_pos]
                    j = node_mapping[node_neg]

                    # Set up the rows and columns corresponding to the voltage source
                    if i != 0:
                        B[i - 1] = -float(component["value"])  # Positive terminal
                    if j != 0:
                        B[j - 1] = float(component["value"])  # Negative terminal

        # Solve the system of equations
        if np.linalg.det(A)==0:
            raise ValueError('Circuit error: no solution')  #error if there are two different current/voltage sources connected across two same nodes

        solution = np.linalg.solve(A, B)
        V = {node: solution[i - 1] for node, i in node_mapping.items() if node != "GND"}  #create dictionary to fit the format of final answer
        V["GND"] = 0.0
        I = {
            component["name"]: solution[num_nodes - 1 + i]
            for i, component in enumerate(components)
            if component["type"] == "V"
        }

    return (V, I)