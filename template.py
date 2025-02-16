import json



########################################################################

# Do not install any external packages. You can only use Python's default libraries such as:
# json, math, itertools, collections, functools, random, heapq, etc.

########################################################################

# Node class to represent the nodes in the graph
class Node:
    neighbours = []
    def __init__(self, index):

        self.index = index  # Instance attribute
        # self. = age  # Instance attribute

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
        self.potentials = []
        self.k_value = self.input["k value (in top k)"]

        # Iterate thorugh the cliques and potentials and create the graph
        for candp in self.input["Cliques and Potentials"]:
            for i in range(candp["clique_size"]):
                for j in range(i+1, candp["clique_size"]):
                    # Add unique edges to the graph
                    self.graph[candp["cliques"][i]].append(candp["cliques"][j])
                    self.graph[candp["cliques"][j]].append(candp["cliques"][i])

                    # Add neighbours to the nodes
                    self.nodes[candp["cliques"][i]].neighbours.append(candp["cliques"][j])
                    self.nodes[candp["cliques"][j]].neighbours.append(candp["cliques"][i])


            self.potentials.append(candp["potentials"])
                
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
            simplicial_vertices.extend(new_simplicial_vertices)

        self.clique_list = []
        visited = set()
        print(ordering)
        for i in range(len(ordering)):
            new_clique = []
            new_clique.append(ordering[i])
            new_clique += [j for j in self.graph[ordering[i]] if j not in visited]
            self.clique_list.append(new_clique)
            visited.add(ordering[i])
            
            if(self.forms_clique(ordering[i:])):
                break

        print(self.clique_list)
        # print("#"*100)
        # print(f'ordering{ordering}')
        # print("#"*100)
        # print(f'graph {self.graph}')


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
        for i in range(len(self.clique_list)):
            for j in range(i+1, len(self.clique_list)):
                if len(set(self.clique_list[i]).intersection(set(self.clique_list[j]))) != 0:
                    self.junction_tree[i].append(j)
                    self.junction_tree[j].append(i)

        print(self.junction_tree)
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
        # junction_tree_potentials = []
        # for i in range(0,len(self.junction_tree)):
        #     if(len)
        # pass

    def get_z_value(self):
        """
        Compute the partition function (Z value) of the graphical model.
        
        What to do here:
        ----------------
        - Implement the message passing algorithm to compute the partition function (Z value).
        - The Z value is the normalization constant for the probability distribution.
        
        Refer to the problem statement for details on computing the partition function.
        """
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
        pass

    def compute_top_k(self):
        """
        Compute the top-k most probable assignments in the graphical model.
        
        What to do here:
        ----------------
        - Use the message passing algorithm to find the top-k assignments with the highest probabilities.
        - Return the assignments along with their probabilities in the specified format.
        
        Refer to the sample test case for the expected format of the top-k assignments.
        """
        pass



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
    evaluator = Get_Input_and_Check_Output('Sample_Testcase.json')
    evaluator.get_output()
    evaluator.write_output('Sample_Testcase_Output.json')