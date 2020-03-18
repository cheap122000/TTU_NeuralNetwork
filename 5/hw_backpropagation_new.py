import numpy
from matplotlib.colors import ListedColormap
import math
import time

# hyper tangent 公式
def tanh(x):
	#return (1.0 - numpy.exp(-2*x))/(1.0 + numpy.exp(-2*x))
	return ((numpy.exp(x) - numpy.exp(-x)) / (numpy.exp(x) + numpy.exp(-x)))
	#return (1/(1 + numpy.exp(-1*x)))

# tanh -1 公式
def tanh_derivative(x):
	return (1 + tanh(x)) * (1 - tanh(x))
	#return (tanh(x) * (1 - tanh(x)))

# 宣告一個類神經的類別
class NeuralNetwork:
	# Initial
	def __init__(self, net_arch):
		# 給定隨機亂數種子
		numpy.random.seed(0)

		# 宣告一些自帶變數
		self.activity = tanh
		self.activity_derivative = tanh_derivative
		self.layers = len(net_arch)
		self.steps_per_epoch = 1000
		self.arch = net_arch
		self.firstWeights = []
		self.weights = []

		# 初始化權重採隨機分布
		for layer in range(len(net_arch) - 1):
			w = 2 * numpy.random.rand(net_arch[layer]+1, net_arch[layer+1]) - 1
			self.weights.append(w)
			self.firstWeights.append(w)
	
	# 印出權重
	def finalWeight(self):
		print(self.weights)

	# fitting 的函式，可改變學習率和周期數
	def fit(self, data, labels, learning_rate=0.01, epochs=10, debug=False, X=None, y=None):

		# 將偏差單位加到輸入
		ones = numpy.ones((1, data.shape[0]))
		Z = numpy.concatenate((ones.T, data), axis=1)
		training = epochs * self.steps_per_epoch

		# 把註解拿掉可以看每一次EPOCH 的預測
		for k in range(training):
			if k % self.steps_per_epoch ==0:
				#print('epochs: {}'.format(k/self.steps_per_epoch))
				for s in data:
					#print(s, nn.predict(s))
					pass

			# one hot encoder
			sample = numpy.random.randint(data.shape[0])
			y = [Z[sample]]

			# 加權並輸出
			for i in range(len(self.weights)-1):
				activation = numpy.dot(y[i], self.weights[i])
				activity = self.activity(activation)
				#add the bias for the next layer
				activity = numpy.concatenate((numpy.ones(1), numpy.array(activity)))
				y.append(activity)

			# 輸出層的運算
			activation = numpy.dot(y[-1], self.weights[-1])
			activity = self.activity(activation)
			y.append(activity)

			# 計算偏差
			error = labels[sample] - y[-1]
			delta_vec = [error * self.activity_derivative(y[-1])]
			self.bias = delta_vec
			# 倒傳遞
			for i in range(self.layers-2, 0, -1):
				error = delta_vec[-1].dot(self.weights[i][1:].T)
				error = error * self.activity_derivative(y[i][1:])
				delta_vec.append(error)
			# reverse
			delta_vec.reverse()

			# learning rate
			for i in range(len(self.weights)):
				layer = y[i].reshape(1, nn.arch[i]+1)
				delta = delta_vec[i].reshape(1, nn.arch[i+1])
				self.weights[i] += learning_rate * layer.T.dot(delta)

			if debug == True:
				if (k) % self.steps_per_epoch == self.steps_per_epoch-1:
					print(int((k+1)/self.steps_per_epoch))
					print(self.weights)
					self.plot_decision_regions(numpy.array([[0, 0],[0, 1],[1, 0],[1, 1]]),numpy.array([0, 1, 1, 0]), num=int((k+1)/self.steps_per_epoch))

	# 預測
	def predict(self, x):
		val = numpy.concatenate((numpy.ones(1).T, numpy.array(x)))
		for i in range(0, len(self.weights)):
			val = self.activity(numpy.dot(val, self.weights[i]))
			val = numpy.concatenate((numpy.ones(1).T, numpy.array(val)))
		return val[1]

# 畫圖
	def plot_decision_regions(self, X, y, points=200, num=None):
		import matplotlib.pyplot as plt
		markers = ('o', '^')
		colors = ('red', 'blue')
		cmap = ListedColormap(colors)

		x1_min, x1_max = X[:, 0].min() -1, X[:, 0].max() + 1
		x2_min, x2_max = X[:, 1].min() -1, X[:, 1].max() + 1

		resolution = max(x1_max - x1_min, x2_max - x2_min)/float(200)
		#resolution = 0.01

		xx1, xx2 = numpy.meshgrid(numpy.arange(x1_min, x1_max, resolution), numpy.arange(x2_min, x2_max, resolution))
		input = numpy.array([xx1.ravel(), xx2.ravel()]).T
		Z = numpy.empty(0)
		for i in range(input.shape[0]):
			val = nn.predict(numpy.array(input[i]))
			if val < 0.5 : val = 0
			if val >=0.5 : val = 1
			Z = numpy.append(Z, val)

		Z = Z.reshape(xx1.shape)

		plt.pcolormesh(xx1, xx2, Z, cmap=cmap)
		#plt.xlim(xx1.min(), xx1.max())
		#plt.ylim(xx2.min(), xx2.max())

		plt.xlabel('x-axis')
		plt.ylabel('y-axis')
		#plt.legend(loc='upper left')

		print(self.bias)
		printWeight='Weights\n'
		printBias='Bias\n'
		for w in self.weights:
			printWeight = printWeight + str(w) + '\n\n'

		for b in self.bias:
			printBias = printBias + str(b) + '\n\n'
		node1_weight = 0
		plt.text(-0.8, -0.2, printWeight, color='white')
		plt.text(-0.8, -1.2, printBias, color='white')
		plt.text(-0.8, 1.8, '# {0}'.format(num), color='white')
		#plt.show()
		plt.savefig('output2/{0}.png'.format(("%3d"%num).replace(' ','0')))
		plt.clf()

if __name__ == '__main__':
	# Initial Model 兩個輸出，第一個隱藏層有三個節點，第二個隱藏層有兩個節點，一個輸出
	nn = NeuralNetwork([2,3,2,1])
	# 印出初始權重
	nn.finalWeight()

	# 訓練資料和答案
	X = numpy.array([[0, 0],[0, 1],[1, 0],[1, 1]])
	y = numpy.array([0, 1, 1, 0])

	# fitting
	nn.fit(X, y, epochs=100, debug=True, X=X, y=y, learning_rate=0.01)

	# 畫圖的參數
	plot_x = []
	plot_y = []
	print('Final prediction')
	for s in X:
		print(s, nn.predict(s))
		for ss in s:
			plot_x.append(ss)
			plot_y.append(nn.predict(s))

	print(plot_x)
	print(plot_y)

	# 印出最後權重
	nn.finalWeight()

	# 畫圖
	nn.plot_decision_regions(X,y, num=0)