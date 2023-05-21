import numpy as np
import random
import math

def objective_function(product_dims, box_dims):
    # Vérifier si la boîte est suffisamment grande pour contenir chaque produit
    for product_dim in product_dims:
        if not all(box_dim >= product_dim for box_dim, product_dim in zip(box_dims, product_dim)):
            print("Box is not large enough for product")
            return float('inf')

    # Calculer le volume de la boîte
    box_volume = box_dims[0] * box_dims[1] * box_dims[2]

    # Calculer le volume total des produits
    total_product_volume = sum([dims[0] * dims[1] * dims[2] for dims in product_dims])

    # Ajouter une pénalité si la boîte n'est pas assez grande pour contenir tous les produits
    if total_product_volume > box_volume:
        print("Box volume is less than total product volume")
        return float('inf')

    # Calculer l'espace vide dans la boîte
    empty_space = box_volume - total_product_volume

    # Calculer le pourcentage d'espace vide dans la boîte
    empty_space_percentage = empty_space / box_volume

    # Calculer la pénalité pour l'espace vide
    if empty_space_percentage < 0.2 or empty_space_percentage > 0.25:
        empty_space_penalty = (empty_space_percentage - 0.225) ** 2  
    else:
        empty_space_penalty = 0

    # Retourner l'espace vide en pourcentage plus la pénalité
    return empty_space_percentage + empty_space_penalty


def random_neighbor(box_dims, product_dims, delta, min_dim=1):
    # Générer un facteur d'échelle aléatoire pour chaque dimension
    scale_factors = [random.uniform(1, 1 + delta) for _ in range(3)]

    # Ajuster chaque dimension de la boîte indépendamment
    new_box_dims = [max(dim * scale_factor, min_dim) for dim, scale_factor in zip(box_dims, scale_factors)]

    # Vérifier si les nouvelles dimensions sont viables
    if all(new_box_dim >= max(product_dims, key=lambda x: x[i])[i] for i, new_box_dim in enumerate(new_box_dims)):
        print(f"New box dims viable: {new_box_dims}")
        return new_box_dims
    else:
        print(f"New box dims not viable: {new_box_dims}")
        return box_dims  # Retourner les dimensions de la boîte actuelle si les nouvelles dimensions ne sont pas viables

def box_volume(box_dims):
    return box_dims[0] * box_dims[1] * box_dims[2]

def can_contain_product(box_dims, product_dims):
    for product_dim in product_dims:
        if not all(box_dim >= product_dim for box_dim, product_dim in zip(box_dims, product_dim)):
            return False
    return True


def simulated_annealing(product_dims, initial_box_dims, initial_temp, cooling_rate, num_iterations):
    dims = initial_box_dims.copy()
    min_volume = box_volume(dims)
    optimal_box_dims = dims.copy()
    temp = initial_temp

    # History of neighbor solutions for debugging
    neighbor_history = []

    for i in range(num_iterations):
        # Generate a neighbor solution
        neighbor_dims = random_neighbor(dims, product_dims, delta=0.05)  # You might want to adjust the value of delta
        neighbor_history.append(neighbor_dims)

        # Calculate volumes of the current solution and the neighbor
        current_volume = box_volume(dims)
        neighbor_volume = box_volume(neighbor_dims)

        # Check if the neighbor solution is better than the current solution
        if neighbor_volume < current_volume:
            # Check if the neighbor solution can contain the product
            if can_contain_product(neighbor_dims, product_dims):
                print(f"Neighbor dims {neighbor_dims} can contain product, replacing current dims {dims}")
                dims = neighbor_dims.copy()
                if neighbor_volume < min_volume:
                    print(f"Neighbor volume {neighbor_volume} is less than minimum volume {min_volume}, updating minimum volume and optimal box dims")
                    min_volume = neighbor_volume
                    optimal_box_dims = neighbor_dims.copy()
            else:
                print(f"Neighbor dims {neighbor_dims} cannot contain product")
        else:
            # If the neighbor solution is worse, accept it with a certain probability
            acceptance_probability = np.exp((current_volume - neighbor_volume) / temp)
            print(f"Neighbor volume {neighbor_volume} is greater than current volume {current_volume}, acceptance probability is {acceptance_probability}")
            if np.random.rand() < acceptance_probability:
                print("Accepting worse neighbor dims")
                dims = neighbor_dims.copy()

        # Decrease the temperature
        temp *= cooling_rate
        print(f"Decreasing temperature to {temp}")

        # Print iteration details
        if i < len(neighbor_history):
            print(f'Iteration {i}: box_dims={dims}, neighbor={neighbor_history[i]}')
        else:
            print(f'Iteration {i}: box_dims={dims}, neighbor=No neighbor')

    return optimal_box_dims
