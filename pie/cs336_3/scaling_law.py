import torch

if __name__ == '__main__':
    # impl of IsoFLOPs method, see https://arxiv.org/pdf/2203.15556 
    # Pratically we have (C, N, D, L) tuples, where C = 6ND. 

    # IsoFLOPs:
    # For a fixed given computing budge C(=6ND), the relationship betwwen model size N and 
    # the final loss L can be qudratic. i.e. L = f_1(c) N^2 + f_2(c) N + f_3(c). Based on this 
    # observation, firstly fit the curve then find the optimal N_i for gievn C_i. Then fit this all thos optimal
    # (N_i, C_i)s using C = N^\{alpha}. Similar process can be performed for traning tokens D.

    # Fit tuples directly:
    # Or, fit tuples using L(N,D) = E + \frac{A}{N^{\alpha}} + \frac{B}{N^{\beta}}, where A, B, E, 
    # \alpha, \beta are fitting parameters. Then for given C = 6ND, we can find optimal N and D based
    # on this constrains, which have closed form. See more in orignal paper. Interestingly, E can be
    # viewed as the entrpy of traning dataset.
    import json
    import math
    data = "./data/isoflops_curves.json"
    with open(data, "r") as f:
        json_str = f.read()
    json_str = json.loads(json_str)

    best_per_compute = {}
    
    for entry in json_str:
        c = entry['compute_budget']
        p = entry['parameters']
        l = entry['final_loss']
        
        if c not in best_per_compute:
            best_per_compute[c] = (p, l)
        else:
            current_p, current_l = best_per_compute[c]
            if l < current_l or (l == current_l and p < current_p):
                best_per_compute[c] = (p, l)
    
    result = []
    x = []
    y = []
    for c, (p, l) in best_per_compute.items():
        result.append({
            "compute_budget": c,
            "parameters": p,
            "final_loss": l
        })
        x.append(math.log(int(c)))
        y.append(math.log(float(p)))
    x = torch.tensor(x)
    y = torch.tensor(y)

    xy = x @ y
    xx = x @ x
    
    alpha = xy / xx
    print(alpha)
