dict={}
with open("seq2seq_report.csv",'r') as f:
    lines=f.readlines()
for i in range(52):
    dict[i+1]={}
for line in lines[1:]:
    name=line.split(',')[1].split(',')[0]
    i=line.split(',')[2]
    if(name=="train"):
        dict[int(i)][name]=float(line.split(',')[-1].split('\n')[0])
    if(name=="val"):
        dict[int(i)][name]=float(line.split(',')[-1].split('\n')[0])
    # if(name=="val"):
    #     dict[int(i)][name]=float(line.split(',')[-2])

print(dict)
import matplotlib.pyplot as plt
trains=[]
for i in range(13):
    trains.append(dict[i+1]["train"])
plt.plot(trains,label="train")
# plt.show()


# trains=[dict[i+2]['test'] for i in range(52)]
# plt.plot(trains,label="test")
# # plt.show()


trains=[dict[i+1]['val'] for i in range(13)]
plt.plot(trains,label="val")
plt.title("Bleu")
plt.xlabel("Epochs")
plt.ylabel("Bleu")
plt.legend()
plt.show()