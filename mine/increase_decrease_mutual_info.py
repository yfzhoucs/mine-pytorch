from mine.models.mine import Mine
import numpy as np
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from celluloid import Camera

fig = plt.gcf()
camera = Camera(fig)



statistics_network = nn.Sequential(
    nn.Linear(1 + 1, 100),
    nn.ReLU(),
    nn.Linear(100, 100),
    nn.ReLU(),
    nn.Linear(100, 1)
)

mine = Mine(
    T = statistics_network,
    loss = 'mine', #mine_biased, fdiv
    method = 'concat'
)

joint_samples = np.random.multivariate_normal(mean=np.array([0,0]), cov=np.array([[1, 0.2], [0.2, 1]]), size=5)
print(joint_samples.shape)

X, Y = joint_samples[:, 0:1], joint_samples[:, 1:2]

plt.scatter(X, Y)
camera.snap()
X = torch.tensor(X, dtype=torch.float32, requires_grad=True)
Y = torch.tensor(Y, dtype=torch.float32, requires_grad=True)


optimizer = torch.optim.Adam(mine.parameters())
optimizer_variable = torch.optim.Adam([X, Y])
for i in range(3000):
    optimizer.zero_grad()
    optimizer_variable.zero_grad()
    outp = mine(X, Y)
    loss_T = outp
    loss_T.backward()
    optimizer.step()

    for j in range(1):
        optimizer.zero_grad()
        optimizer_variable.zero_grad()
        outp = mine(X, Y)

        # + means to increase the mutual info (decrease the output)
        loss_V = + outp
        loss_V.backward()
        optimizer_variable.step()

    if (i + 1) % 100 == 0:
        print(i, outp.item())
        X_np = X.detach().numpy()
        Y_np = Y.detach().numpy()
        plt.scatter(X_np, Y_np)
        camera.snap()


animation = camera.animate(blit=False, interval=300)
# animation.save('decrease_mi.mp4')
X = X.detach().numpy()
Y = Y.detach().numpy()
plt.scatter(X, Y)
camera.snap()
plt.show()