import random
import matplotlib.pyplot as plt


#上升及下降次數
up_times = 0
down_times = 0

#上升及下降次數記錄
up_ary=[0]*10
down_ary=[0]*10

#初始化記錄當下及前一次產生的值
previous_number=0
present_number=0

#產生一個隨機亂數，第一行為講義上的方法，第二行為PYTHON上的方法
present_number=random.random()

#紀錄50000次的結果
for i in range(0,50000):
	previous_number = present_number
	present_number = random.random()
	
	#當新產生的值開始上升時，下次次數歸零
	if(present_number-previous_number>0):
		up_times=up_times+1
		down_ary[down_times] = down_ary[down_times] + 1
		down_times=0

	#當新產生的值開始下降時，上升次數歸零
	if(present_number-previous_number<0):
		down_times=down_times+1
		up_ary[up_times]=up_ary[up_times]+1
		up_times=0

#紀錄最後一次的結果
if(present_number-previous_number>0):
	up_ary[up_times]=up_ary[up_times]+1
if(present_number-previous_number<0):
	down_ary[down_times]=down_ary[down_times]+1

#輸出執行後的結果
for j in range(1,10):
	print("up {0} times: {1}   \t down {2} times: {3}".format(j,up_ary[j],j,down_ary[j]))


plt.style.use('ggplot')
width = 0.4
group1 = ['1','2','3','4','5','6','7','8','9']
group2 = ['1','2','3','4','5','6','7','8','9']
group1_index = range(len(group1))
group2_index = range(len(group2))
fig = plt.figure()
ax1=fig.add_subplot(1,1,1)
ax1.bar([i-width/2 for i in group1_index], up_ary[1:],width=width,align='center',color='darkblue')
ax1.bar([i+width/2 for i in group2_index], down_ary[1:],width=width,align='center',color='yellow')
ax1.xaxis.set_ticks_position('bottom')
ax1.yaxis.set_ticks_position('left')
plt.xticks(group1_index,group1,rotation=0,fontsize='small')
plt.xlabel('Continuous times')
plt.ylabel('Times')
plt.savefig('Random_Numbers_Distribution2.png', dpi=400, bbox_inches='tight')
plt.show()