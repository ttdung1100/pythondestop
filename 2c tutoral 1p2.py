file.write(file_text)
file.write("\n")
file.close()
    file_name = raw_input("Enter filename: ")
    if len(file_name) == 0:
        print "Next time please enter something"
        sys.exit()
try:
    file = open(file_name, "r")
    except IOError:
        print "There was an error reading file"
        sys.exit()


