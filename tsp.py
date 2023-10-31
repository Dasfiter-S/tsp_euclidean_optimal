import random
import multiprocessing as mp
from scipy.spatial import Delaunay
import networkx as nx
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def generate_cities(num_cities):
    # Generate random city coordinates
    cities = [(random.uniform(0, 10), random.uniform(0, 10)) for _ in range(num_cities)]
    return cities

def calculate_distances(points):
    # Calculate pairwise Euclidean distances between points
    num_points = len(points)
    distances = []
    for i in range(num_points):
        row = []
        for j in range(num_points):
            row.append(((points[i][0] - points[j][0]) ** 2 + (points[i][1] - points[j][1]) ** 2) ** 0.5)
        distances.append(row)
    return distances

def tsp_euclidean_optimal(cities):
    # Step 1: Construct Delaunay triangulation
    points = list(cities)
    distances = calculate_distances(points)
    triangulation = Delaunay(points)
    
    # Step 2: Build Minimum Spanning Tree (MST)
    mst_edges = []
    mst_graph = nx.Graph()
    
    for simplex in triangulation.simplices:
        city1, city2, city3 = simplex
        dist1, dist2, dist3 = distances[city1][city2], distances[city2][city3], distances[city3][city1]
        
        # Add edges to the MST graph
        mst_graph.add_edge(city1, city2, weight=dist1)
        mst_graph.add_edge(city2, city3, weight=dist2)
        mst_graph.add_edge(city3, city1, weight=dist3)
        
        mst_edges.extend([(city1, city2, dist1), (city2, city3, dist2), (city3, city1, dist3)])
    
    mst = nx.minimum_spanning_tree(mst_graph)
    mst_nodes = list(mst.nodes)
    hamiltonian_cycle = nx.approximation.traveling_salesman_problem(mst, cycle=True)
    
    # Step 4: Convert the Hamiltonian cycle to a path
    hamiltonian_path = []
    visited = set()
    
    for node in hamiltonian_cycle:
        if node not in visited:
            hamiltonian_path.append(node)
            visited.add(node)
    
    # Step 5: Calculate the total distance (solution)
    total_distance = sum(distances[hamiltonian_path[i]][hamiltonian_path[i+1]]
                         for i in range(len(hamiltonian_path)-1))
    
    return hamiltonian_path, total_distance

def tsp_agent(agent_id):
    tour, distance = tsp_euclidean_optimal(cities)
    return tour, distance

if __name__ == "__main__":
    num_cities = 20
    num_agents = 8
    cities = generate_cities(num_cities)

    scaler = StandardScaler()
    cities = scaler.fit_transform(cities)
    pca = PCA(n_components=2)
    cities = pca.fit_transform(cities)

    # Initialize the multiprocessing Pool with initargs
    with mp.Pool(processes=num_agents) as pool:
        tours_and_distances = pool.map(tsp_agent, range(num_agents))

    best_tour, best_distance = min(tours_and_distances, key=lambda x: x[1])

    print("Best TSP Tour:", best_tour)
    print("Total Distance:", best_distance)
