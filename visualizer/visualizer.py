from matplotlib import pyplot as plt
import json


#
#
#
#
# ax.plot(VecStart_x + VecEnd_x, VecStart_y + VecEnd_y, VecStart_z +VecEnd_z)

def plot_genome(genome: dict):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for drone in genome["drones"]:
        print(drone)
        positions = [[],[],[]]
        for waypoint in drone:
            positions[0].append(waypoint[0])
            positions[1].append(waypoint[1])
            positions[2].append(waypoint[2])

        ax.plot(*positions)


    plt.show()


def plot_json(path: str):
    plot_genome(json.load(open(path)))


plot_json("../resulting_drones/worldly-flower-92_0.json")
