import csv
anspath='./answer.txt'
answerfile = open(anspath, 'w')

testpath = './testdata.txt'
testfile = open(testpath, 'w')

with open ("C:/Users/Admin/Downloads/train.csv") as f:
    reader = csv.reader(f)
    next(reader)
    c = 0
    for row in reader:
        if c > 5000:
            answerfile.write(row[3])
            testfile.write(row[2])
            c += 1
        else:
            break

testfile.close()
answerfile.close()
      