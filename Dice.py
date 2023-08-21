from time import time
from random import randint as rand
from collections import Counter

def calc_scores(hand, handdict): #hand is tuple
    #rank, points in rank, total points
    Rank = rank(handdict)
    arr = [Rank,rank_size(handdict,Rank),sum(hand)]
    score = (arr[2]-5) + 32 * (arr[1]-1) + 32*36*(arr[0])
    return tuple(arr+[score]) #(rank, total_size_in_rank, total_hand_size, overall_score)


def compare(hand1,hand2): #hand1 and hand2 are (tuple, dict)
    # -1: hand1>hand2
    # 0: hand1=hand2
    # 1: hand1<hand2
    scores1 = calc_scores(hand1[0],hand1[1])
    scores2 = calc_scores(hand2[0],hand2[1])
    ### Just use overall score ###
    x,y = scores1[3],scores2[3]
    return (x < y) - (y < x)
    #########################
    x,y = scores1[0],scores2[0] #rank
    diff = (x < y) - (y < x)
    if diff != 0:
        return diff
    x,y = scores1[1],scores2[1] #size in rank
    diff = (x < y) - (y < x)
    if diff != 0:
        return diff
    x,y = scores1[0],scores2[0] #total_hand_size
    diff = (x < y) - (y < x)
    if diff != 0:
        return diff



def verify_compare(arr_tdf): #input is ((sorted_tuple, its_dict, frequency))
    smaller = 0
    counter = 0
    arr = [] # return scores associated with each pair in arr_tdf
    tie = 0
    left = 0 # which is bigger
    right = 0
    matches = 0
    comps = 0
    #print(arr_tdf[0:10])
    for i in arr_tdf:
        for j in arr_tdf:
            #print(i)
            #print(j)
            change = 0
            cmp = compare((i[0],i[1]),(j[0],j[1]))
            arr.append(cmp)
            if cmp == 0 and i==j:
                q = i[2] * j[2]
                matches += q
            else:
                q = i[2] * j[2]
            comps += q
            if cmp == 0:
                if i != j:
                    pass#print(f"Different, tie: {i}     {j}")
                    tie += i[2] * j[2]
                else:
                    tie += i[2] * j[2]
            right += (cmp==1) * i[2] * j[2]
            left += (cmp==-1) * i[2] * j[2]
            counter += 1
            if not counter % 1000:
                pass#print('')
            if counter%1563 < 7:# and counter < 10000:
                pass#print(f"({cmp}) ~ {q} ~ {left} / {tie} / {right} ~ {i},{j}")
    #print(f"{left} / {tie} / {right} = {left+tie+right} ({counter})")
    #print(f"Matches: {matches} / {comps} comparisons")
    return tuple(arr)

def rank(hand):
    quantities = hand.values()
    if len(hand) == 1:
        return 8 # 5-of-a-kind = 8
    elif len(hand) == 2 and max(quantities) == 4:
        return 7 # 4-of-a-kind = 7
    elif len(hand) == 2 and max(quantities) == 3:
        return 6 # full house = 6
    elif len(hand) == 5:
        if 6 in hand and 1 not in hand:
            return 5 # 6-high straight = 5
        elif 1 in hand and 6 not in hand:
            return 4 # 5-high straight = 4
        return 0
    elif max(quantities) == 3:
        return 3 # 3-of-a-kind = 3
    elif len(hand) == 3:
        return 2 # 2 pairs
    else:
        return 1 # pair
#@profile
#def to_dict(arr):
#    return {j:arr.count(j) for i,j in enumerate(arr) if j not in arr[0:i]}
def to_dict(arr):
    return dict(Counter(arr))
    '''
    if len(arr) < 16:
        return {j:arr.count(j) for i,j in enumerate(arr) if j not in arr[0:i]}
    else:
        arr_iter = iter(arr)
        for i,j in arr_iter:
            count = 1
            while j==arr[i+1]:
                print()
    '''

def rank_size(hand,rank): #dict, assumes random order
    if rank in [0,4,5,6,8]: # add all dice
        return sum([i*j for i,j in hand.items()])
    # [1,2,3,7] remaining | pair, 2pair, 3oaK, 4oaK
    # if rank,quantity is consistant with rank -> quantity map for each key
    return sum([i*j for i,j in hand.items() if {1:2, 2:2, 3:3, 7:4}[rank] == j ])

def generate1():
    arr = []
    count = 0
    tmparr = [1,1,1,1,1]
    for i in range(6**5-1): #works
        count += 1
        tmparr[-1] += 1
        while 7 in tmparr:
            indx = tmparr.index(7)
            tmparr[indx] = 1
            tmparr[indx-1] += 1
        arr.append(tmparr.copy())
    return arr

def generate2(): #works, faster
    arr = []
    for a in range(1,7):
        for b in range(1,7):
            for c in range(1,7):
                for d in range(1,7):
                    for e in range(1,7):
                        arr.append([a,b,c,d,e])
    return arr

def method1(arr): #bad
    #print(len(arr))
    match = 0
    comps = 0
    for i,j in enumerate(arr):
        for i2,j2 in enumerate(arr[i::]): #includes comparison with self
            if j==j2:
                match += 1
            comps += 1
        if not i%1000:
            pass#print(i,"done")
    return (match,comps)

def method2(arr): #Good, must be sorted
    #print(arr[0:10])
    match = 0
    comps = 0
    for i in arr:
        for j in arr:
            if i==j:
                match += 1
            comps += 1
        #if not i%1000:
        #    pass#print(i,"done")
    return (match,comps)

#@profile
def method3(arr): #fast (bad)
    arr3 = arr.copy()
    for i in arr3:
        i.sort()
    arr3.sort()
    arr3 = [tuple(i) for i in arr3]
    arr3 = to_dict(arr3) #arr3 is now tuples (sorted) mapped to frequency
    arr_reduced = list(arr3.items())
    #print(arr_reduced[0:10])

    match = 0
    rmatch = 0
    comps = 0
    rcomps = 0
    for i,j in enumerate(arr_reduced):
        for i2,j2 in enumerate(arr_reduced[i::]):
            #print(i,i2,j,j2)
            
            #print('Compare:',j[0],j2[0])
            if j[0]==j2[0]:
                quantity = (j[1]**2 + j[1]) // 2
                #print('Compare:',j[0],j2[0])
                #print('Match',j[1],j2[1],quantity)
                match += quantity
                rmatch += 1
            else:
                quantity = j[1] * j2[1]
            comps += quantity
            rcomps += 1
        if not i%100:
            pass#print(i,"done")
    #print("Real match,comp:",rmatch,rcomps)
    return (match,comps)

#@profile
def method3_use_arr_dt(arr_dt): #fast (good)
    #print(arr_dt[0:10])
    match = 0
    rmatch = 0
    comps = 0
    rcomps = 0
    for i1,j1 in enumerate(arr_dt):
        for j2 in arr_dt[i1::]:
            if j1[0]==j2[0]:
                match += j1[1] * j2[1]
                rmatch += 1
            comps += j1[1] * j2[1]
            rcomps += 1
    #print("Real match,comp:",rmatch,rcomps)
    return (match,comps)

def odds_of_beating_any(arr, hand): #arr is { (hand): (quantity, score) }, hand is sorted tuple
    win = 0
    lose = 0
    tie = 0
    arrd = to_dict(arr)
    arrr = rank_size(arrd)
    for i,j in enumerate(arr_reduced):
        for i2,j2 in enumerate(arr_reduced[i::]):
            #print(i,i2,j,j2)
            val1,val2 = j[1],j2[1]
            
            #print('Compare:',j[0],j2[0])
            if j[0]==j2[0]:
                quantity = (val1**2 + val1) // 2
                #print('Compare:',j[0],j2[0])
                #print('Match',j[1],j2[1],quantity)
                match += quantity
                rmatch += 1
            else:
                quantity = val1 * val2
            comps += quantity
            rcomps += 1
        if not i%100:
            pass#print(i,"done")
    #print("Real match,comp:",rmatch,rcomps)
    return (match,comps)

#returns unsorted tuple with duplicates
def generate_all_rolls(hand,mask): #hand and mask are list or tuple. Mask: 1=replace
    indices = [i for i,j in enumerate(mask) if j==1]
    beginstr = f"{tuple([(str(j) if mask[i]==0 else f'i{i}') for i,j in enumerate(hand)])}"
    beginstr = beginstr.replace('\'','')
    endstr = ''.join([f" for i{i} in range(1,7)" for i,j in enumerate(mask) if j==1])
    fullstr = 'tuple([' + beginstr + endstr + '])'
    return eval(fullstr)

#works, do not sort masks
def generate_all_masks():
    beginstr = f"{tuple([(f'i{j}') for j in range(5)])}"
    beginstr = beginstr.replace('\'','')
    endstr = ''.join([f" for i{i} in range(2)" for i in range(5)])
    fullstr = 'tuple([' + beginstr + endstr + '])'
    #print("Full string:",fullstr)
    return eval(fullstr)

def save_to_file(arr1,arr2,mask,win): #tuple, tuple, tuple, float
    with open("best_moves_data.txt","a+") as f:
        string = f"{arr1}{arr2}{mask}{win}\n"
        string = string.replace(", ","")
        #print(string)
        f.write(string)

#@profile
def main():
    print("MAIN")
    rankmap = {0:"nothing", 1:"pair", 2:"2pair", 3:"3-of-a-kind",
               4:"5-high straight", 5:"6-high straight", 6:"full house",
               7:"4-of-a-kind", 8:"5-of-a-kind"}

    
    functions = [generate1, generate2]
    functions2 = [method1, method2, method3]
    
    arr = generate2()
    arr_unsorted = generate2()
    #print(arr == arr_unsorted)
    #print(arr is arr_unsorted)
    for i in arr:
        i.sort()
    ##################
    #test before sorting whole arr
    if 0:
        print("START")
        r = method1(arr)
        print(r)
        r = method2(arr)
        print(r)
        r = method3(arr)
        print(r)
        print("END")
    ##################
    #print(len(arr_unsorted))
    #print(arr_unsorted[0:10])
    arr.sort()
    #print(arr[0:10])
    arr_t = [tuple(i) for i in arr]
    # 252 unique sorted hands
    arr_d = to_dict(arr_t) # { (hand) : frequency }
    #print(f"aaaaa {arr_d[(1,2,3,4,5)]}")
    #print(arr_d)
    #print("arr_d:",[[i,j] for i,j in arr_d.items()][0:10])
    arr_dt = tuple((i,j) for i,j in arr_d.items()) # ( (hand, frequency) )
    arrm_t_d = { i : to_dict(i) for i in arr_t } # { (hand) : dict }
    arrm_t_ts = { tuple(i) : tuple(sorted(tuple(i))) for i in arr_unsorted } # { (unsorted) : (sorted) }
    arr_tdf = tuple([ ( i[0], to_dict(i[0]), i[1] ) for i in arr_dt ]) # ( (unique_sorted_tuple, its dict, frequency) )
    arr_dt_comps = verify_compare(arr_tdf) # ( -1, -1, 0, ... ) results from ^
    # { (hand1, hand2) : (comparison, quantity) }
    all_comps = { ( j1[0], j2[0] ) : (arr_dt_comps[i2+i1*252],j1[1]*j2[1]) for i1,j1 in enumerate(arr_dt) for i2,j2 in enumerate(arr_dt) }
    masks = generate_all_masks() #32 masks from 00000 -> 11111 ; 1=replace
    

    print("Done Computing Tables")
    if 1: #import ^ best moves
        with open("best_moves_data.txt","r+") as f:
            lines = f.readlines()
        # { (hand1, hand2) : (bestmask, win_chance) }
        arrm_tt_mf = { ( tuple([int(i) for i in x[1:6]]), tuple([int(i) for i in x[8:13]]) ) : ( tuple([int(i) for i in x[15:20]]), float(x[21:-1]) ) for x in lines }
        #print(1)
        print(f'{len(list(arrm_tt_mf.items()))} combinations imported into arrm_tt_mf.')
    total = 0
    count = 0
    as_list = list(arrm_tt_mf.items())
    for i in as_list:
        total += i[1][1]
        count += 1
    print(f"total={total}, count={count}, win%={total/count}")
    #print()
    #print(arrm_tt_mf[((1,3,4,4,6),(1,2,3,6,6))])
    # ( (hand, mask) : ( (combinations) )
    #print("COMP",list(all_comps.items())[0:1])
    #print(all_comps[((1,1,1,1,1),(1,1,1,1,1))])

    if 1:
        while 1:
            format_input = lambda x: tuple([int(i) for i in x])
            try:
                hand1 = format_input(input("Enter 2 hands: "))
                hand2 = format_input(input(">"))
                hand1 = tuple(sorted(hand1))
                hand2 = tuple(sorted(hand2))
                output = arrm_tt_mf[(hand1,hand2)]
                print(f"\n{hand1} vs {hand2}\nBest Move: {output[0]} ({output[1]*100:.4f}%)",'\n',sep='')
            except:
                pass
    
    # calculate arrm_best_move, export to file

    """
    with open("best_moves_data.txt","a+") as f:
        arr1 = (2,3,3,3,5)
        arr2 = (1,2,6,6,6)
        mask = (1,0,0,0,1)
        win = 100/377
        string = f"{arr1}{arr2}{mask}{win}\n"
        string = string.replace(", ","")
        print(string)
        f.write(string)
    """
    
    '''
    for i in lines:
        x = i[:-1]
        a = x[1:6]
        b = x[8:13]
        c = x[15:20]
        d = x[21:]
        print(f' a{a} b{b} c{c} d{d}')
    '''

    if 0: #calculate all best masks
        with open("best_moves_data.txt","a+") as f:
            counter = 0
            masks = generate_all_masks()
            bigTime = time()
            for arr1 in arr_tdf:
                smallTime = time()
                for arr2 in arr_tdf:
                    #arr1 = (1,3,4,5,6)
                    #arr2 = (1,2,3,6,6)
                    #arr2 = arrm_t_ts[arr2]
                    #arr2q = arr_d[arr2]
                    total_combs = 0
                    #d2 = arrm_t_d[arr2]
                    biggest_cmp = ((),0) #hand, times won
                    for mask in masks:
                        result = generate_all_rolls(arr1[0],mask) # unsorted dupes tuples
                        result2 = [arrm_t_ts[j] for i,j in enumerate(result)]
                        #print(result2)
                        result2 = to_dict(result2)
                        #print(f"Lens: {len(result)},{len(result2)}")
                        #print(mask)
                        total_cmp = 0
                        total_q = 6**mask.count(1)
                        #Time = time()
                        for i in result2.items():
                            cmp = all_comps[i[0],arr2[0]][0]
                            q = i[1]
                            total_cmp += (cmp==0) * .5 * q
                            total_cmp += (cmp==-1) * 1 * q
                        #print(time() - Time)
                        cmp_ratio = total_cmp / total_q
                        #print(f"Result: {cmp_ratio*100}%\n")
                        if cmp_ratio > biggest_cmp[1]:
                            biggest_cmp = (mask,cmp_ratio)
                        total_combs += len(result)
                    #save_to_file(arr1[0],arr2[0],biggest_cmp[0],biggest_cmp[1])
                    string = f"{arr1[0]}{arr2[0]}{biggest_cmp[0]}{biggest_cmp[1]}\n"
                    string = string.replace(", ","")
                    f.write(string)
                    counter += 1
                    if not (counter%1000):
                        print(f"{arr1[0]}, {arr2[0]} : Best: {biggest_cmp[0]} ({biggest_cmp[1]*100}%)")
                print(time() - smallTime)
            print(time() - bigTime)
    input()
    
        
    all_combinations = { (hand,mask) : combinations for i,j in enumerate() }
    #odds_vs_rand = { i[0] : (wins / 60_466_176) for i in arr_dt}
    print("all_comps:")
    hands = [tuple([rand(1,6) for i in range(5)]) for j in range(10)]
    for i,j in enumerate(hands):
        hands[i] = arrm_t_ts[j]
        print(hands[i])
    for i in hands:
        for j in hands:
            print(f"{i}, {j} = {all_comps[(i,j)]}")
    print(all_comps[((1,1,1,2,5),(1,1,1,3,4))])
    print("arr_t_tdf:",len(arr_tdf))
    for i in range(10):
        print(arr_tdf[i])
    # ( arr_dt[i][0], to_dict(arr_dt[i][0]), arr_dt[i][1] )
    print("arr_dt:",arr_dt[0:10])
    # iterate through arr_dt (ignore freq for now) 
    to_dict(generate_all_rolls())
    
    if 0: # test functions
        for j in functions:
            Time = time()
            arr2 = []
            for i in range(10**0):
                arr2 = j()
            print(j.__name__,time() - Time)
        for j in functions2:
            Time = time()
            for i in range(1):
                results = j(arr)
            print(results, j.__name__,time() - Time)
        if 1:
            Time = time()
            j = method3_use_arr_dt
            results = j(arr_dt)
            print(results, j.__name__,time() - Time)
    '''
    # lookup (1, 3, 4, 6, 6) in dict, get index, then iterate over array
    arr_reduced = list(arr3.items)
    print(arr_reduced[0:10])
    example_tuple = (1, 3, 4, 6, 6)
    indx = list(arr3.keys()).index(example_tuple)
    match = 0
    comps = 0
    for i,j in enumerate(arr_reduced,indx):
        if j == 
        comps += j
    #iterate over the rest, compare
    '''
    
    #arr4 = arr3.items()
    #print(arr4[0:10])
    #print(len(arr3),arr3)
    print('\n\n\n')
    #print(arr3)
        
    ''' #calculate odds of same 3 hands
    for j in range(10):
        match = 0
        tries = 2*10**6
        for i in range(tries):
            a = hand([rand(1,6) for k in range(5)])
            b = hand([rand(1,6) for k in range(5)])
            if a.dict == b.dict:
                match += 1
        print("Match: 1 in ",tries/match)
    '''

    '''
    for i,j in enumerate(arr): #del dupes
        if j in arr[0:i]:
                del(arr[i])
    '''
    #arr has no duplicates
    '''
    arr = [j for i,j in enumerate(arr) if j not in arr[i+1::]]
    print("len:",len(arr))
    for i in arr:
        if arr.count(i) != 1:
            print(i,arr.count(i),"ERROR")
    '''
    '''
    listed = []
    for i in arr:
        x = hand(i)
        score = x.score
        print(i,score)
        if score in listed:
            print("^ LISTEDDDDDDDDDDDDD")
        else:
            #print("UNIQUE",score,i,x.rank_name)
            listed.append(score)
            #print(listed)
    '''

    """
    arr_dicts = [to_dict(i) for i in arr]
    print(arr_dicts[0:10])
    scores = [rank(i) for i in arr_dicts]
    print(scores[0:10])
    score_count = to_dict(scores)
    #score_words = [rankmap[j]: for i,j in enumerate(scores)]
    print(score_count)
    for i,j in score_count.items():
        print(i,j)
        print({rankmap[i]:j})
    sc_words = {rankmap[i]:j for i,j in score_count.items()}

    hand([1,2,3,4,5])
    hand([6,6,6,6,6])
    hand([6,6,6,6,5])
    hand([1,1,1,1,2])
    hand([6,6,6,5,5])
    hand([1,1,1,2,2])
    hand([1,2,2,1,1])
    hand([2,3,4,5,6])
    hand([1,2,3,4,6])
    hand([1,3,4,5,6])
    hand([1,1,2,3,4])

    #input()
    #print(len(arr))
    """

if __name__ == "__main__":
    main()
    input()



