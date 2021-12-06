import numpy as np
input=np.array([1,4,1,1,1,1,5,1,1,5,1,4,2,5,1,2,3,1,1,1,1,5,4,2,1,1,3,1,1,1,1,1,1,1,2,1,1,1,1,1,5,1,1,1,1,1,1,1,1,1,4,1,1,1,1,5,1,4,1,1,4,1,1,1,1,4,1,1,5,5,1,1,1,4,1,1,1,1,1,3,2,1,1,1,1,1,2,3,1,1,2,1,1,1,3,1,1,1,2,1,2,1,1,2,1,1,3,1,1,1,3,3,5,1,4,1,1,5,1,1,4,1,5,3,3,5,1,1,1,4,1,1,1,1,1,1,5,5,1,1,4,1,2,1,1,1,1,2,2,2,1,1,2,2,4,1,1,1,1,3,1,2,3,4,1,1,1,4,4,1,1,1,1,1,1,1,4,2,5,2,1,1,4,1,1,5,1,1,5,1,5,5,1,3,5,1,1,5,1,1,2,2,1,1,1,1,1,1,1,4,3,1,1,4,1,4,1,1,1,1,4,1,4,4,4,3,1,1,3,2,1,1,1,1,1,1,1,4,1,3,1,1,1,1,1,1,1,5,2,4,2,1,4,4,1,5,1,1,3,1,3,1,1,1,1,1,4,2,3,2,1,1,2,1,5,2,1,1,4,1,4,1,1,1,4,4,1,1,1,1,1,1,4,1,1,1,2,1,1,2])
input=np.array([8])

i=0
j=len(input)

epoch=30


def spawn(days):
    age_counts = [0] * 9
    for age in input:
        age_counts[age] += 1

    for i in range(days):
        age_counts[(i + 7) % 9] += age_counts[i % 9]

    return sum(age_counts)
  



def solution1(data, days) -> int:
    # approach: assign fishes to day groups
    fishes = [0 for _ in range(9)]
    for fish in data:
        fishes[fish] += 1

    for day in range(days):
        print(fishes)
        creators = fishes.pop(0)
        fishes.append(creators)
        fishes[6] += creators
    return sum(fishes)


def solution2(data) -> int:
    return solution1(data, 256)
print(solution1(input,epoch))
while True:
    
    if(i==epoch):
        print(len(input))
        break
    #print(i,input)
    for x in range(0,j):
        if input[x]==0:
            input=np.append(input,[9])
            j=j+1
            input[x]=7
    i=i+1
    input=input-1


        

    