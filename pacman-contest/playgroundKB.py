import time
# a = time.time()

# time.sleep(5)
# print(time.time() -a)


# a = 1
# b = 0
# count = 0
# while a - b <= 1:
#     print("in loop")
#     count = count + 1

#     if count > 5:
#         break


# buckets = [[0 for col in range(5)] for row in range(10)]
# print(len(buckets[0])) 


# def test():
#     return "apple", "pear"

# a, b = test()
# (c, d) = test()

# print(a)
# print(b)
# print(c)
# print(d)


import operator
x = {("text1","a"): 2, ("text2","b"): 4, ("text3","c"): 3}
print(x)

sorted_x = sorted(x.items(), key=operator.itemgetter(1))

sorted_x.reverse()
print(sorted_x[0][0])
