import sys

def displayFile():
 fobj = open("C:/Users/ivans_000/Desktop/MASTER/Spring2019/Deep_Learning/Assignment1_Sangines/Students.txt","r")
 for line in fobj:
    print(line)

def copyFile():
 fobj = open("C:/Users/ivans_000/Desktop/MASTER/Spring2019/Deep_Learning/Assignment1_Sangines/Students.txt","r")
 fobj2 = open("C:/Users/ivans_000/Desktop/MASTER/Spring2019/Deep_Learning/Assignment1_Sangines/Students2.txt","w")
 for line in fobj:
    print(line)
    fobj2.write(line)
 fobj2.write("\n12344" + "\t" + "Gerard" + "\t" + "Way" + "\t" + "96" + "\t" + "89")
 fobj2.close()

def main():
 #displayFile()
 copyFile()

if __name__ == "__main__":
 sys.exit(int(main() or 0))
