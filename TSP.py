import numpy as np
import itertools
import matplotlib.pyplot as plt

def generate_cities(N):
    # Generate N cities randomly in a 1x1 square box
    cities_coordinates = np.random.rand(N, 2)  # Each city is represented by its (x, y) coordinates
    return cities_coordinates

def calculate_distances(cities_coordinates, N):
    distances = np.zeros((N, N))  # Initialize an N x N matrix for distances
    
    for i in range(N):
        for j in range(N):
            distances[i][j] = np.sqrt(np.sum((cities_coordinates[i] - cities_coordinates[j]) ** 2))  # Calculate Euclidean distance
    
    return distances

def generate_initial_route(distances, N):
    route = [0]  # Start from city 0
    visited = set([0])  # Keep track of visited cities
    
    while len(route) < N:
        # Find the nearest neighbour of the last city in the route that hasn't been visited yet
        min_dist = float('inf')  # Initialize to a large value
        next_city = -1  # Index of the next city to add to the route
        for j in range(N):
            if j not in visited:  # Ignore already visited cities
                dist = distances[route[-1]][j]  # Distance from the last city in the route to city j
                if dist < min_dist:  # Update the minimum distance and next city
                    min_dist = dist
                    next_city = j
        route.append(next_city)  # Add the next city to the route
        visited.add(next_city)  # Mark the city as visited
    
    return route  # The initial route as a sequence of city indices

def swap_route_segment(route, N, i, j):
    # Check if the swap is within bounds and makes sense
    if i < j and j - i < N - 2:  # Condition for valid swap
        # Perform the swap by reversing the sequence from route[i] to route[j]
        route[i:j+1] = reversed(route[i:j+1])  # Swap the segment
        return route  # Return the swapped route
    else:
        return route

def should_swap(distances, route, N, i, j, T):
    if i < j and j - i < N - 2:  # Condition for valid swap
        # Calculate the energy difference between the swapped and unswapped route
        E = distances[route[(i - 1) % N]][route[j]] + distances[route[i]][route[(j + 1) % N]] - distances[route[(i - 1) % N]][route[i]] - distances[route[j]][route[(j + 1) % N]]
        
        # If E is negative, always accept the swap
        if E < 0:
            return True
        
        # If E is positive, accept the swap with probability e^{-E/T}
        if np.exp(- E / T) > np.random.rand():  # Using NumPy's random number generator for speed and efficiency
            return True
        else:
            return False
    else:
        return False

def calculate_initial_temperature(distances, N):
    # Calculate the mean value of squared distances
    squared_distances = np.square(distances)
    sum_squared_distances = np.sum(squared_distances)
    mean_squared_distances = sum_squared_distances / (N * (N - 1))
    
    # Calculate the mean value of distances
    sum_distances = np.sum(distances)
    mean_distances = sum_distances / (N * (N - 1))
    
    # Calculate the standard deviation of all city routes
    std_dev = np.sqrt(mean_squared_distances - mean_distances ** 2)
    
    # Triple the standard deviation as the initial temperature
    initial_temperature = 2 * std_dev
    
    return initial_temperature

def current_temperature(initial_temperature, step, N):
    # Calculate the number of possible swaps (N*(N-1)/2)
    num_swaps = N * (N - 1) // 2  # Using integer division
    
    # Determine the current step's temperature decrement
    temperature_decrement = 0.1 * initial_temperature  # Decrease by 10% of initial temperature
    
    # Calculate the current temperature based on the step number and decrement
    current_temperature = initial_temperature - (step // num_swaps) * temperature_decrement
    
    # Ensure the temperature doesn't fall below a minimum threshold (e.g., a small positive value)
    current_temperature = max(current_temperature, 0.01 * temperature_decrement)  # Adjust this minimum threshold as needed
    
    return current_temperature

def calculate_route_length(distances, route, N):
    route_length = 0  # Initialize the length of the route to zero
    for i in range(N):
        # Calculate the distance between the current city and the next one in the route
        city_i = route[i]
        city_j = route[(i + 1) % N]  # Use modulo to loop back to the start if necessary
        route_length += distances[city_i][city_j]  # Add the distance to the total route length
    return route_length

def search_for_shortest_route(distances, N, initial_temperature, route, step):
    process = False # Flag to indicate if any swaps were processed during this iteration
    # Check all possible swaps
    for i in range(N):
        for j in range(i + 1, N):  # Only check swaps with a different city index to avoid duplicates
            T = current_temperature(initial_temperature, step, N) # Calculate the temperature
            if should_swap(distances, route, N, i, j, T):  # Call your custom function to determine if a swap should be made
                route = swap_route_segment(route, N, i, j) # Make the swap
                route = route[i:] + route[:i] # Do the rotation
                step = step + 1 # Record the step
                process = True # Swap processed
                return route, step, process
            else:
                step = step + 1 # Record the step
    return route, step, process

def find_shortest_route(distances, N):
    route = generate_initial_route(distances, N) # Generate an initial guess of the route
    initial_temperature = calculate_initial_temperature(distances, N) # Calculate the initial temperature
    
    # Initialization of temporary variables
    step = 0
    process = True

    # Start the loop
    while process:
        route, step, process = search_for_shortest_route(distances, N, initial_temperature, route, step)

    # Find the starting point
    index = route.index(0)
    route = route[index:] + route[:index]

    # Calculate the length
    route_length = calculate_route_length(distances, route, N)

    return route, route_length, step

def find_shortest_route_by_enumeration(distances, N):
    # Set the starting city
    start_city = 0
    
    # Generate all permutations of the remaining cities (1 to N-1)
    remaining_cities = list(range(1, N))
    shortest_route = None
    shortest_length = float('inf')
    
    # Iterate through all possible routes starting from city 0
    for perm in itertools.permutations(remaining_cities):
        route = [start_city] + list(perm)  # Route starts at city 0 and follows the permutation
        
        # Calculate the total distance of the current route
        route_length = calculate_route_length(distances, route, N)
        
        # Check if this is the shortest distance found
        if route_length < shortest_length:
            shortest_length = route_length
            shortest_route = route
    
    return shortest_route, shortest_length

def plot_route(route, cities_coordinates):
    # Extract coordinates for cities in the route
    route_coordinates = np.array([cities_coordinates[city] for city in route])
    
    # Create the plot
    plt.figure(figsize=(6, 6))  # Set the figure size to visualize the square region
    plt.axis("equal")  # Equal aspect ratio
    plt.xlim(0, 1)  # Set the limits of the x-axis to match the square region
    plt.ylim(0, 1)  # Set the limits of the y-axis to match the square region
    plt.plot(route_coordinates[:, 0], route_coordinates[:, 1], linestyle='-', color='black')
    plt.scatter(route_coordinates[1:, 0], route_coordinates[1:, 1], color='black')
    plt.scatter(route_coordinates[0, 0], route_coordinates[0, 1], s=100, color='red')
    
    # Annotate the cities
    plt.annotate(f'City {route[0]}', route_coordinates[0], textcoords="offset points", color='red', xytext=(0,10), ha='center')
    for i, (x, y) in enumerate(route_coordinates[1:]):
        plt.annotate(f'City {route[i + 1]}', (x, y), textcoords="offset points", color='black', xytext=(0,10), ha='center')
    
    # Close the route
    plt.plot([route_coordinates[-1, 0], route_coordinates[0, 0]], 
             [route_coordinates[-1, 1], route_coordinates[0, 1]], color='black', linestyle='-')

    plt.title('Shortest Route Between Cities')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.show()

N = 5
cities_coordinates = generate_cities(N)
distances = calculate_distances(cities_coordinates, N)
route, route_length, step = find_shortest_route(distances, N)
route_by_enumeration, length_by_enumeration = find_shortest_route_by_enumeration(distances, N)
print(route)
print(route_by_enumeration) 
print(route_length / length_by_enumeration)
print(step)
plot_route(route, cities_coordinates)

