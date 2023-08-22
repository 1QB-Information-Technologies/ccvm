import pickle

RESULTS_DIR = "./results/dl/"
# Solve the problem
iterations = 15000
for alpha in [0.00001, 0.0001, 0.001, 0.01, 0.1]:
    for beta1 in [0.8, 0.9]:
        for beta2 in [0.999]:
            filename = f"{RESULTS_DIR}adam_alpha{alpha:.05f}_1beta{beta1:.03f}_2beta{beta2:.04f}_iter{iterations:06d}.pkl"
            with open(filename, 'rb') as file:
                data = pickle.load(file)
            print(data)
    # print(f"{data['alpha']}", end=" ")