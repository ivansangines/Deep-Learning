from math import pi
import sys

def computeAvg(a,b,c) :
 return (a + b + c)/3.0;

def doComplexMath() :
 num1 = 3 + 4j
 num2 = 6 + 3.5j
 res = num1 * num2
 return res;

def mapTest(mylist) :
 ys = map(lambda x: x * 2, mylist)
 #ys is a map object, so we need to convert it to a list
 result = []
 for elem in ys:
    result.append(elem)
 return result

 

def main():
 #Simple math
 print("Result = ",computeAvg(5,8,9))
 print("Result = ",doComplexMath())

 #working with strings
 s1 = "hello"
 s2 = s1.upper()
 print(s1 + "\n" + s2)

 s3 = "hello there how are you"
 pos = s3.find('how')
 print("index: " + str(pos)) #should print index 12

 s4 = 'helllo'
 s5 = s1[0:3] 
 print(s5)

 snum = "25"
 num1 = int(snum)
 print(num1 + 1)
 snum2 = "25.5"
 num2 = float(snum2)
 print(num2 + 1)

 #list and tupples examples
 fruits = ['apples', 'oranges', 'bananas', 'plums', 'pineapples']
 print(fruits)
 pfruits = fruits[2:4]
 print(pfruits)
 for fr in pfruits :
    print(fr)
 del fruits[2]
 fruits.remove('oranges')
 fruits.append('kiwi')
 print(fruits + pfruits)

 s1 = ('john','starkey',12341,3.5) # firstname, lastname, id, gpa
 print(s1[1])

 #map and lambda functions
 ys = mapTest([5,7,2,6])
 print(ys)



if __name__ == "__main__":
 sys.exit(int(main() or 0))
