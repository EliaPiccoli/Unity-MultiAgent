from matplotlib import pyplot as plt

def data(file):
    y = [int(line) for line in file]
    return [i for i in range(1, len(y)+1)], y

def data_partial(file, lines):
    i = 0
    y = []
    x = []
    for line in file:
        y.append(int(line))
        x.append(i)
        i += 1
        if i > lines:
            break
    return x, y

f = open("results/success_list.txt", "r")
#x, y = data_partial(f, 9859)
x, y = data(f)
_max = max(y)

plt.title("GRainbow - {}%".format(_max))
plt.xlabel("Episodes")
plt.ylabel("Success")
plt.plot(x, y)
plt.savefig("plots/genetic_v11.png")