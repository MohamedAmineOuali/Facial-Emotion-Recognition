import os
import sys


# this function extruct only the actions units as an interger and without intensity
def label_preprocessing(labelRoot,
                        outputRoot):
    # verification of files
    for dir in os.listdir(labelRoot):
        for dir1 in os.listdir(labelRoot + dir):
            count = len([name for name in os.listdir(labelRoot + dir + '/' + dir1)])
            assert count == 1

    for root, dirs, files in os.walk(labelRoot):
        for filename in files:
            assert filename.endswith(".txt")
            with open(root + "/" + filename)as file:
                newFilename = file.name.replace(labelRoot, outputRoot)
                os.makedirs(os.path.dirname(newFilename), exist_ok=True)
                with open(newFilename, "w") as f:
                    for line in file:
                        try:
                            line = line.split()
                            actionUnit = int(float(line[0]))
                            print(actionUnit, file=f)
                        except:
                            print("error in this file"+f, file=sys.stderr)


outputRoot="/home/Amine/dataset/www.consortium.ri.cmu.edu/data/ck/CK+/AU/"
labelRoot="/home/Amine/dataset/www.consortium.ri.cmu.edu/data/ck/CK+/FACS/"


label_preprocessing(labelRoot,outputRoot)