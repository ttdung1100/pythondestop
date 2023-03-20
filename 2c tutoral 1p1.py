file = open(file_name, "w")
except IOError:
    print "There was an error writing to", file_name
    sys.exit()

    print "Enter '", file_finish,
    print "' When finished"
    
    while file_text != file_finish:
        file_text = raw_input("Enter text: ")
            if file_text == file_finish:file.close
# close the file
    break


