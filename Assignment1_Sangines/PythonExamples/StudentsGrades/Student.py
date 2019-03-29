class Student(object):
    """description of class"""
    def __init__(self, firstName, lastName, id):
        self.firstName = firstName
        self.lastName = lastName
        self.id = id
        self.test = [] #initially empty, filling it up with students.txt
        self.grade = ""
    
    def addTestScore (self, score):
        self.test.append(score)

    def computeGrade (self):
        res = 0
        for test in self.test:
            res += int(test)
        average = res/len(self.test)
        score = ""
        if average > 90:
            score = "A"
        elif average > 85:
            score = "A-"
        elif average > 80:
            score = "B"
        else:
            score = "B"
        return score

