for i in [1, 2, 3, 4, 5]:
    print i*i



for i in [1, 2, 3, 4, 5]:
    print i
    for j in [1, 2, 3, 4, 5]:
        print j # first line in "for j" block
        print i + j # last line in "for j" block
    print i # last line in "for i" block
print "done looping"




