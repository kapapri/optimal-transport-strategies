print("Something plz")
a = 2
b = 3
c = a + b

print(c)

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

