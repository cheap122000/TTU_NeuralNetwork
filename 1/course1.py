import random

min_number = 0
max_number = 32768
#random.seed(10) #反註解這行數字都會一樣
for i in range(0, 10):
	random_number = random.randint(min_number, max_number)
	print(random_number)

segments = [0] * 100
for i in range(0, 100000):
	random_number = random.randint(0, 99)
	segments[random_number] += 1

print(segments)

import matplotlib.pyplot as plt

plt.style.use('ggplot')
fig = plt.figure()
ax1 =fig.add_subplot(1,1,1)
ax1.plot(segments, marker=r'o', color=u'blue', linestyle='-')
ax1.xaxis.set_ticks_position('bottom')
ax1.yaxis.set_ticks_position('left')
ax1.set_title('Random Numbers Distribution')
plt.xlabel('Regions')
plt.ylabel('Times')
plt.legend(loc='best')
plt.savefig('Random_Numbers_Distribution.png', dpi=400, bbox_inches='tight')
plt.show()