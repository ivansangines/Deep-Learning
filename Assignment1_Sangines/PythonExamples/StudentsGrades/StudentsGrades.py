import sys
from Student import Student

def processGrades():
 fobj = open("C:/Users/ivans_000/Desktop/MASTER/Spring2019/Deep_Learning/Assignment1_Sangines/Students.txt","r")
 fobj2 = open("C:/Users/ivans_000/Desktop/MASTER/Spring2019/Deep_Learning/Assignment1_Sangines/StudentGrades.txt","w")

 for line in fobj:
    parts = line.split('\t')
    s1 = Student("","",0)
    s1.id = parts[0]
    s1.firstName = parts[1]
    s1.lastName = parts[2]
    s1.addTestScore(parts[3])
    s1.addTestScore(parts[4])
    s1.grade = s1.computeGrade()
    # now write id and grade to an output file
    fobj2.write(s1.id + "\t" + s1.lastName + "\t" + s1.grade + "\n")
 fobj2.close()
 print("done processing processGrades..")

def main():
 processGrades()

if __name__ == "__main__":
 sys.exit(int(main() or 0))