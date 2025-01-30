import os
import random
import shutil
from itertools import islice


outputfolderpath = 'Dataset/SplitData'
inputfolderpath ='Dataset/all'
splitratio = {"train":0.7 , "val":0.2,"test":0.1}
classes = ["Fake","Real"]

try: 
    shutil.rmtree(outputfolderpath)
    print("Removed Directory")

except OSError as e:
    os.mkdir(outputfolderpath)


#Directories to create

os.makedirs(f"{outputfolderpath}/train/images",exist_ok=True)
os.makedirs(f"{outputfolderpath}/train/labels",exist_ok=True)
os.makedirs(f"{outputfolderpath}/val/images",exist_ok=True)
os.makedirs(f"{outputfolderpath}/val/labels",exist_ok=True)
os.makedirs(f"{outputfolderpath}/test/images",exist_ok=True)
os.makedirs(f"{outputfolderpath}/test/labels",exist_ok=True)


#get the names
listnames = os.listdir(inputfolderpath)
# print(listnames)
print(len(listnames))
uniquenames = []
for name in listnames:
    uniquenames.append(name.split('.')[0])
uniquenames = list(set(uniquenames))
# print(len(uniquenames))



#shuffle

random.shuffle(uniquenames)
# print(uniquenames)

#find the number of images each folders
lenData = len(uniquenames)
# print(f'Total Images {lenData }')

lentrain = int(lenData * splitratio["train"])
lenval = int(lenData * splitratio["val"])
lentest = int(lenData * splitratio["test"])

print(f'Total Images {lenData } \nSplit: {lentrain} , {lenval} , {lentest}')

#Put the remaining images in training

if lenData != lentrain +lentest + lenval:
    remaining = lenData - (lentrain + lentest + lenval)
    lentrain += remaining 

print(f'Total Images {lenData } \nSplit: {lentrain} , {lenval} , {lentest}')


#Split the list

lengthtosplit = [lentrain,lenval,lentest]
Input = iter(uniquenames)
Output = [list(islice(Input, elem)) for elem in lengthtosplit]
print(f'Total Images {lenData } \nSplit: {len(Output[0])} , {len(Output[1])} , {len(Output[2])}')


#copy the files
sequence = ['train','val','test']
for i,out in enumerate(Output):
    for filename in out:
        shutil.copy(f'{inputfolderpath}/{filename}.jpeg',f'{outputfolderpath}/{sequence[i]}/images/{filename}.jpeg')
        shutil.copy(f'{inputfolderpath}/{filename}.txt',f'{outputfolderpath}/{sequence[i]}/labels/{filename}.txt')


print("Split process Completed")

#creating data.yaml file 

dataYaml = f'path: ../Data\n\
train: ../train/images\n\
val: ../val/images\n\
test: ../test/images\n\
\n\
nc : {len(classes)}\n\
names:{classes} '

f = open(f"{outputfolderpath}/data.yaml","a")
f.write(dataYaml)
f.close()

print("Data.yaml file created da ")
