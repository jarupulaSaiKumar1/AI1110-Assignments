import matplotlib
import matplotlib.pyplot as plt


fig = plt.figure()
ax = fig.add_subplot(111)

rect1 = matplotlib.patches.Rectangle((0, 0), 8, 6,ec='black', color ='green',alpha=0.65)


ax.add_patch(rect1)

plt.title('Scaled diagram for the problem')
plt.xlabel('Along the x-axis')
plt.ylabel('Along the y-axis')
plt.xlim([-1, 9])
plt.ylim([-1, 7])

v1 = [8,0]
v2 = [0,6]
plt.plot(v1,v2,color='black')

plt.grid()
plt.show()
