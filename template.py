import json
import heapq

########################################################################

# Do not install any external packages. You can only use Python's default libraries such as:
# json, math, itertools, collections, functools, random, heapq, etc.

########################################################################

# Node class to represent the nodes in the graph
class Node:
    neighbours = []
    def __init__(self, index):

        self.index = index  # Instance attribute 

class DisjointSet:
    def __init__(self, n):
        self.parent = [i for i in range(n)]
        self.rank = [0 for i in range(n)]
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        x_root = self.find(x)
        y_root = self.find(y)

        if x_root == y_root:
            return
        
        if self.rank[x_root] < self.rank[y_root]:
            self.parent[x_root] = y_root
        elif self.rank[x_root] > self.rank[y_root]:
            self.parent[y_root] = x_root
        else:
            self.parent[y_root] = x_root
            self.rank[x_root] += 1

    def is_same_set(self, x, y):
        return self.find(x) == self.find(y)

class Inference:
    def __init__(self, data):
        """
        Initialize the Inference class with the input data.
        
        Parameters:
        -----------
        data : dict
            The input data containing the graphical model details, such as variables, cliques, potentials, and k value.
        
        What to do here:
        ----------------
        - Parse the input data and store necessary attributes (e.g., variables, cliques, potentials, k value).
        - Initialize any data structures required for triangulation, junction tree creation, and message passing.
        
        Refer to the sample test case for the structure of the input data.
        """

        self.input = data
        self.VariablesCount = self.input["VariablesCount"]

        self.graph = [[] for i in range(self.VariablesCount)]
        self.nodes = [Node(i) for i in range(self.VariablesCount)]
        self.potentials = {}
        self.k_value = self.input["k value (in top k)"]

        # Iterate thorugh the cliques and potentials and create the graph
        for candp in self.input["Cliques and Potentials"]:
            for i in range(candp["clique_size"]):
                for j in range(i+1, candp["clique_size"]):
                    # Add unique edges to the graph
                    if candp["cliques"][j] not in self.graph[candp["cliques"][i]]:
                        self.graph[candp["cliques"][i]].append(candp["cliques"][j])
                    
                    if candp["cliques"][i] not in self.graph[candp["cliques"][j]]:
                        self.graph[candp["cliques"][j]].append(candp["cliques"][i])

            # Add the potentials to the potentials dictionary
            if self.potentials.get(tuple(candp["cliques"])) is None:
                self.potentials[tuple(candp["cliques"])] = {}
            if(len(candp["cliques"]) == 1):
                 
                 index_list = ['#']*self.VariablesCount 
                 index_list[candp["cliques"][0]] = '1'
                 index = ''.join(index_list)

                 if index in self.potentials.get(tuple(candp["cliques"])):
                    self.potentials[tuple(candp["cliques"])][index] *= candp["potentials"][1]
                 else :
                    self.potentials[tuple(candp["cliques"])][index] = candp["potentials"][1]

                 index_list = ['#']*self.VariablesCount 
                 index_list[candp["cliques"][0]]= '0'
                 index = ''.join(index_list) 
                 if index in self.potentials.get(tuple(candp["cliques"])):
                    self.potentials[tuple(candp["cliques"])][index] *= candp["potentials"][0]
                 else :
                    self.potentials[tuple(candp["cliques"])][index] = candp["potentials"][0]
            else:
                count = 0
                for i in range(2):
                    for j in range(2):
                        index_list = ['#']*self.VariablesCount 
                        index_list[candp["cliques"][0]] = str(i)
                        index_list[candp["cliques"][1]] = str(j)
                        index = ''.join(index_list)
                        if index in self.potentials.get(tuple(candp["cliques"])):
                            self.potentials[tuple(candp["cliques"])][index] *= candp["potentials"][count]
                        else :
                            self.potentials[tuple(candp["cliques"])][index] = candp["potentials"][count]
                        count += 1
        print('graph', self.graph) 
        print("_"*100)
        print(f'potentials {self.potentials}')
        pass

    def isSimplicial(self, node, graph):
        # Check if the node is simplicial or not
        for i in range(len(graph[node])):
            for j in range(i+1, len(graph[node])):
                if graph[node][j] not in graph[graph[node][i]] and graph[node][i] not in graph[graph[node][j]]:
                    return False
        return True

    def make_node_simplicial(self, graph, node):
        # Creat a clique between all the neighbours of the node
        for i in range(len(graph[node])):
            for j in range(i+1, len(graph[node])):
                if graph[node][j] not in graph[graph[node][i]]:
                    graph[graph[node][i]].append(graph[node][j])
                    graph[graph[node][j]].append(graph[node][i])
 
                    self.graph[graph[node][i]].append(self.graph[node][j])
                    self.graph[graph[node][j]].append(self.graph[node][i])
        pass
    
    def forms_clique(self, ordering):
        for i in range(len(ordering)):
            for j in range(i+1, len(ordering)):
                if ordering[j] not in self.graph[ordering[i]] and ordering[i] not in self.graph[ordering[j]]:
                    return False
        return True

    def triangulate_and_get_cliques(self):  
        """
        Triangulate the undirected graph and extract the maximal cliques.
        
        What to do here:
        ----------------
        - Implement the triangulation algorithm to make the graph chordal.
        - Extract the maximal cliques from the triangulated graph.
        - Store the cliques for later use in junction tree creation.

        Refer to the problem statement for details on triangulation and clique extraction.
        """
        simplicial_vertices = []
        # First find the initial simplicial vertices
        for i in range(self.VariablesCount):
            if self.isSimplicial(i, self.graph):
                simplicial_vertices.append(i) 
        # Copy the graph and simplicial vertices 
        copy_graph = [self.graph[i].copy() for i in range(self.VariablesCount)]
        simplicial_vertices_set = set(simplicial_vertices)
        ordering = [] # Variable elimination ordering
        count = 0
        while(len(simplicial_vertices) != 0):
            if(count == 10):
                break
            count += 1 
            top = simplicial_vertices.pop(0)
            ordering.append(top) 
            # Remove the top from the graph and all the edges associated with it
            for i in range(self.VariablesCount):
                if top in copy_graph[i]:
                    copy_graph[i].remove(top)

            copy_graph[top].clear

            new_simplicial_vertices = []
            for i in range(self.VariablesCount):
                if i not in simplicial_vertices_set and self.isSimplicial(i, copy_graph):
                    new_simplicial_vertices.append(i)
                    simplicial_vertices_set.add(i) 

            # If no new simplicial vertices are found, then we need to add a new simplicial vertex
            if(len(new_simplicial_vertices) == 0) and (len(simplicial_vertices_set) != self.VariablesCount):
                # Goind with first heuristic: Choose vertex with smallest degree and connect all its neighbours
                min_degree = 100000000
                min_degree_vertex = -1
                for i in range(self.VariablesCount):
                    if len(copy_graph[i]) < min_degree:
                        min_degree = len(copy_graph[i])
                        min_degree_vertex = i 
                # Connect all the neighbours of the min_degree_vertex
                self.make_node_simplicial(copy_graph, min_degree_vertex)

                # Add the min_degree_vertex to the simplicial vertices
                new_simplicial_vertices.append(min_degree_vertex)
                simplicial_vertices_set.add(min_degree_vertex)
            # print(f'new_simplicial_vertices {new_simplicial_vertices} {simplicial_vertices_set} \n copy_graph {copy_graph}')
            simplicial_vertices.extend(new_simplicial_vertices)

        self.clique_list = []
        visited = set()
        print("#"*100)
        print(f'ordering {ordering}')
        for i in range(len(ordering)):
            new_clique = []
            new_clique.append(ordering[i])
            new_clique += [j for j in self.graph[ordering[i]] if j not in visited]
            self.clique_list.append(new_clique)
            visited.add(ordering[i])
            
            if(self.forms_clique(ordering[i:])):
                break
        for i in range(len(self.clique_list)):
            self.clique_list[i] = sorted(self.clique_list[i])
        
        print(f'cliques {self.clique_list}') 


        pass
    
    def get_junction_tree(self):
        """
        Construct the junction tree from the maximal cliques.
        
        What to do here:
        ----------------
        - Create a junction tree using the maximal cliques obtained from the triangulated graph.
        - Ensure the junction tree satisfies the running intersection property.
        - Store the junction tree for later use in message passing.

        Refer to the problem statement for details on junction tree construction.
        """

        self.junction_tree = [[] for i in range(len(self.clique_list))]
        edges_list = []
        for i in range(len(self.clique_list)):
            for j in range(i+1, len(self.clique_list)):
                intersection_set = set(self.clique_list[i]).intersection(set(self.clique_list[j])) 
                edges_list.append([i, j, len(intersection_set)])

        edges_list = sorted(edges_list, key = lambda x: x[2], reverse = True)
        print(f'edges_list {edges_list}')
        ds = DisjointSet(len(self.clique_list))
        for i in range(len(edges_list)):
            if ds.find(edges_list[i][0]) != ds.find(edges_list[i][1]):
                self.junction_tree[edges_list[i][0]].append(edges_list[i][1])
                self.junction_tree[edges_list[i][1]].append(edges_list[i][0])
                ds.union(edges_list[i][0], edges_list[i][1])

        
        print("#"*100) 
        print(f'junction tree {self.junction_tree}')
        pass

    def assign_potentials_to_cliques(self):
        """
        Assign potentials to the cliques in the junction tree.
        
        What to do here:
        ----------------
        - Map the given potentials (from the input data) to the corresponding cliques in the junction tree.
        - Ensure the potentials are correctly associated with the cliques for message passing.
        
        Refer to the sample test case for how potentials are associated with cliques.
        """

        self.junction_tree_potentials = [{} for i in range(len(self.junction_tree))]
        visited_cliques = set()
        for idx, node in enumerate(self.clique_list):   
            for cliq in self.potentials:
                if set(cliq).issubset(set(node)) and cliq not in visited_cliques: 
                    visited_cliques.add(cliq)
                    for mask in range(2**len(node)):
                        index_list = ['#']*self.VariablesCount
                        for i in range(len(node)):
                            index_list[node[i]] = str((mask>>i)&1)
                        index = ''.join(index_list)
                        index_list2 = ['#']*self.VariablesCount
                        for i in range(len(cliq)):
                            index_list2[cliq[i]] = index_list[cliq[i]]
                        index2 = ''.join(index_list2)
                        if index2 in self.potentials[cliq]:
                            if index in self.junction_tree_potentials[idx]:
                                self.junction_tree_potentials[idx][index] *= self.potentials[cliq][index2]
                            else:
                                self.junction_tree_potentials[idx][index] = self.potentials[cliq][index2]

        # print("#"*100)
        # for p in self.junction_tree_potentials:
        #     print(p)

        # print("\n\n\n")
        pass
    

    def get_z_value(self):
        """
        Compute the partition function (Z value) of the graphical model.
        
        What to do here:
        ----------------
        - Implement the message passing algorithm to compute the partition function (Z value).
        - The Z value is the normalization constant for the probability distribution.
        
        Refer to the problem statement for details on computing the partition function.
        """
        final_beliefs = {}
        def multiply_potentials(potential1, potential2): 
            result = {}
            
            for key1, value1 in potential1.items():
                for key2, value2 in potential2.items():
                    # merged_key = ''.join([k1 if k1 != '#' else k2 for k1, k2 in zip(key1, key2)])
                    # result[merged_key] = result.get(merged_key, 0) + value1 * value2
                    set1 = set([i for i in range(len(key1)) if key1[i] != '#'])
                    set2 = set([i for i in range(len(key2)) if key2[i] != '#'])
                    # print(f'set1 {set1} set2 {set2}')
                    equals = True
                    for i in set2:
                        if i in set1 and key1[i] != key2[i]:
                            equals = False
                            break
                    if not equals:
                        continue
                    if set2.difference(set1) == set(): # If set2 is a subset of set1
                        result[key1] = value1 * value2
            #             print(f'key1 {key1} key2 {key2} value1 {value1} value2 {value2} {result[key1]}')
            # print(f'potential1 {potential1} \npotential2 {potential2} \nresult {result}')
            return result

        def sum_out_variable(potential, variable_index):
            """Sum out a variable from a potential table."""
            new_potential = {}
            for key in potential:
                new_key = key[:variable_index] + '#' + key[variable_index + 1:]
                new_potential[new_key] = new_potential.get(new_key, 0) + potential[key]
            return new_potential

        # Step 1: Select a root clique
        root = 2  # Choosing the first clique as the root

        # Step 2: Perform upward (collect) message passing
        visited = set()
        messages = {}

        def collect_messages(node, parent):
            """Recursively collect messages from child to parent."""
            visited.add(node)
            incoming_potential = self.junction_tree_potentials[node]
            for neighbor in self.junction_tree[node]:
                if neighbor != parent and neighbor not in visited:
                    collect_messages(neighbor, node) 
                    # print(f'node {node} neighbor {neighbor}')
                    incoming_potential = multiply_potentials(incoming_potential, messages[neighbor])
                    # print('_'*100)
                    # print(f'node {node} parent {parent}')
                    # print(f'incoming_potential {incoming_potential} \n messages[neighbor] {messages[neighbor]}')
            # Sum out variables not in the separator set (between parent and child)
            final_beliefs[node] = incoming_potential
            if parent is not None:
                separator = set(self.clique_list[node]) & set(self.clique_list[parent])
                # print(f'separator {separator}')
                for var in self.clique_list[node]:
                    if var not in separator:
                        incoming_potential = sum_out_variable(incoming_potential, var)

            messages[node] = incoming_potential
            # print(f'node {node} incoming_potential {incoming_potential}')

        collect_messages(root, None)
 
        self.final_beliefs = final_beliefs
        self.messages = messages
        for msg in messages:
            print(f'{msg} {messages[msg]}')
        z_value = sum(messages[root].values())
        self.z_value = z_value
        print(z_value)
        print('-'*100) 
        pass

    def compute_marginals(self):
        """
        Compute the marginal probabilities for all variables in the graphical model.
        
        What to do here:
        ----------------
        - Use the message passing algorithm to compute the marginal probabilities for each variable.
        - Return the marginals as a list of lists, where each inner list contains the probabilities for a variable.
        
        Refer to the sample test case for the expected format of the marginals.
        """
        visited = set()
        visited_variable = set()
        marginals = {}
        root = 2
        def multiply_potentials(potential1, potential2): 
            """Multiply two potential tables."""
            
            result = {}
            for key1, value1 in potential1.items():
                for key2, value2 in potential2.items():
                    set1 = {i for i in range(len(key1)) if key1[i] != '#'}
                    set2 = {i for i in range(len(key2)) if key2[i] != '#'}
                    
                    if any(i in set1 and key1[i] != key2[i] for i in set2):
                        continue  # Skip mismatched keys
                    
                    if set2.issubset(set1):  # If set2 is a subset of set1
                        result[key1] = result.get(key1, 0) + value1 * value2
            return result
        def divide_potentials(potential1, potential2): 
            result = {}
            for key1, value1 in potential1.items():
                for key2, value2 in potential2.items():
                    set1 = {i for i in range(len(key1)) if key1[i] != '#'}
                    set2 = {i for i in range(len(key2)) if key2[i] != '#'}
                    
                    if any(i in set1 and key1[i] != key2[i] for i in set2):
                        continue   
                    
                    if set2.issubset(set1):
                        result[key1] = result.get(key1, 0) + value1 * (1/value2)
            return result
        def sum_out_variable(potential, variable_index):
            """Sum out a variable from a potential table.""" 
            new_potential = {}
            for key, value in potential.items():
                new_key = key[:variable_index] + '#' + key[variable_index + 1:]
                new_potential[new_key] = new_potential.get(new_key, 0) + value
            return new_potential
        
        def get_marginals(potential, variable_index):

            new_potential = {}
            for key, value in potential.items():
                new_key = key[:variable_index] + '#' + key[variable_index + 1:]
                curr_key = 1 if key[variable_index] == '1' else 0
                new_potential[curr_key] = new_potential.get(curr_key, 0) + value
            return new_potential
        def distribute_messages(node, parent, parent_potential=None):
            """Recursively distribute messages from parent to child."""  
            visited.add(node)
            if parent != None:
                parent_potential[node] = multiply_potentials(parent_potential[node], parent_potential[parent])
            

            for var in self.clique_list[node]: 
                if var in visited_variable:
                    continue
                visited_variable.add(var)
                current_marginals = get_marginals(parent_potential[node], var)
                marginals[var] = current_marginals
                 
            # Pass messages to children
            for neighbor in self.junction_tree[node]:
                if neighbor not in visited:
                    separator = set(self.clique_list[node]) & set(self.clique_list[neighbor])
                    current_marginals = {}
                    for var in self.clique_list[neighbor]:
                        if var not in separator:
                            current_marginals = sum_out_variable(parent_potential[neighbor], var)
                    outgoing_messages = dict(parent_potential)
                    outgoing_messages[node] = divide_potentials(outgoing_messages[node], current_marginals)
                    
                    current_marginals = {}
                    for var in self.clique_list[node]:
                        if var not in separator:
                            outgoing_messages[node] = sum_out_variable(outgoing_messages[node], var)
                    distribute_messages(neighbor, node, outgoing_messages)



        # Run downward pass
        distribute_messages(root, None, self.final_beliefs)
        for mrgs in marginals:
            print(f"variable_index {mrgs}")
            for val in marginals[mrgs]:
                print(f'{val} {marginals[mrgs][val]/self.z_value}')
        print("_"*100) 

        print(marginals)
        

    def compute_top_k(self):
    
        """
        Compute the top-k most probable assignments in the graphical model.
        
        What to do here:
        ----------------
        - Use the message passing algorithm to find the top-k assignments with the highest probabilities.
        - Return the assignments along with their probabilities in the specified format.
        
        Refer to the sample test case for the expected format of the top-k assignments.
        """ 
        final_beliefs = {}
        k = self.k_value
        def multiply_potentials(potential1, potential2):
            """Multiply two potential tables, keeping top-k assignments."""
            result = []
            for (key1, value1) in potential1:
                for (key2, value2) in potential2:
                    set1 = {i for i in range(len(key1)) if key1[i] != '#'}
                    set2 = {i for i in range(len(key2)) if key2[i] != '#'}

                    if any(i in set1 and key1[i] != key2[i] for i in set2):
                        continue  # Skip mismatched keys

                    merged_key = ''.join([key1[i] if key1[i] != '#' else key2[i] for i in range(len(key1))])
                    result.append((merged_key, value1 * value2))

            # Keep only top-k assignments
            return heapq.nlargest(k, result, key=lambda x: x[1])

        def sum_out_variable(potential, variable_index):
            """Sum out a variable while keeping track of top-k assignments."""
            grouped_potentials = {}

            for assignment, prob in potential:
                new_key = assignment[:variable_index] + '#' + assignment[variable_index + 1:]
                if new_key not in grouped_potentials:
                    grouped_potentials[new_key] = []
                heapq.heappush(grouped_potentials[new_key], (-prob, assignment))  # Use min heap for top-k tracking

            # Retain only top-k assignments per group
            new_potential = []
            for new_key, heap in grouped_potentials.items():
                top_k_values = heapq.nsmallest(k, heap)  # Extract top-k largest
                new_potential.extend([(assignment, -prob) for prob, assignment in top_k_values])

            return new_potential

        # Step 1: Select a root clique
        root = 2  # Choose a root clique (can be changed as needed)

        # Step 2: Perform upward (collect) message passing
        visited = set()
        messages = {}

        def collect_messages(node, parent):
            """Recursively collect messages from child to parent, keeping top-k assignments."""
            visited.add(node)
            incoming_potential = [(key, prob) for key, prob in self.junction_tree_potentials[node].items()]

            for neighbor in self.junction_tree[node]:
                if neighbor != parent and neighbor not in visited:
                    collect_messages(neighbor, node)
                    incoming_potential = multiply_potentials(incoming_potential, messages[neighbor])

            final_beliefs[node] = incoming_potential

            if parent is not None:
                separator = set(self.clique_list[node]) & set(self.clique_list[parent])
                for var in self.clique_list[node]:
                    if var not in separator:
                        incoming_potential = sum_out_variable(incoming_potential, var)

            messages[node] = incoming_potential

        collect_messages(root, None)

        # Step 3: Extract top-k assignments from root message
        top_k_assignments = heapq.nlargest(k, messages[root], key=lambda x: x[1])
        print(top_k_assignments)
        # Step 4: Format output
        formatted_assignments = [(assignment.replace('#', '0'), prob/self.z_value) for assignment, prob in top_k_assignments]
        print('_'*100)
        
        final_top_k_processed = [{"assignment":[int(digit) for digit in x[0]], "probability":x[1]} for x in  formatted_assignments]
        print('top k assignments')
        print(final_top_k_processed)




########################################################################

# Do not change anything below this line

########################################################################

class Get_Input_and_Check_Output:
    def __init__(self, file_name):
        with open(file_name, 'r') as file:
            self.data = json.load(file)
    
    def get_output(self):
        n = len(self.data)
        output = []
        for i in range(n):
            inference = Inference(self.data[i]['Input'])
            inference.triangulate_and_get_cliques()
            inference.get_junction_tree()
            inference.assign_potentials_to_cliques()
            z_value = inference.get_z_value()
            marginals = inference.compute_marginals()
            top_k_assignments = inference.compute_top_k()
            output.append({
                'Marginals': marginals,
                'Top_k_assignments': top_k_assignments,
                'Z_value' : z_value
            })
        self.output = output

    def write_output(self, file_name):
        with open(file_name, 'w') as file:
            json.dump(self.output, file, indent=4)


if __name__ == '__main__':
    evaluator = Get_Input_and_Check_Output('testCase1.json')
    evaluator.get_output()
    evaluator.write_output('Sample_Testcase_Output.json')