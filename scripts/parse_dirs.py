"""
script to parse the directories and create a csv file for each directory
it will create a csv file for each directory with the name of the directory and the values of the forgetting measure
"""

# dir0,dir1,dir2
# task000 log0dir0,task000 log0dir1,task000 log0dir2,
# task000 log1dir0,task000 log1dir1,task000 log1dir2,
import sys
import os
import pdb

def parseFile(file, num):
    lines = []
    for line in file.readlines():
        if "Forgetting" in line and f"Task{num}" in line:
            val = line.split("= ")[1]
            val = float(val)
            lines.append(val)
    file.close()
    return lines
    
def parseDirs(dname, num):
    subdirs = os.listdir(dname)
    dest = open(f'{dname}_{num}.csv', 'w')
    dest.writelines(f"{d}," for d in subdirs)
    dest.writelines("\n")
    
    logs = {}
    for subdir in subdirs:
        subsubdirs = os.listdir(f"{dname}/{subdir}")
        for subsubdir in subsubdirs:
            files = os.listdir(f"{dname}/{subdir}/{subsubdir}")
            for file in files:
                if file.split(".")[-1] == "txt":
                    logno = file.split(".")[0]
                    if logno not in logs.keys():
                        logs[logno] = []
                    readFrom = open(f"{dname}/{subdir}/{subsubdir}/{file}", 'r')
                    logs[logno].append(parseFile(readFrom,num))
                    readFrom.close()
    
    if num == "000":
        for key in logs:
            for v in logs[key]:
                # pdb.set_trace() 
                dest.write(f"{str(v[0])},")
            dest.write("\n")
            for v in logs[key]:
                # pdb.set_trace() 
                dest.write(f"{str(v[1])},")
            dest.write("\n")
    else:        
        # pdb.set_trace() 
        for key in logs:
            for v in logs[key]:
                dest.write(f"{str(v[0])},")
            dest.write("\n")    
        
    dest.close()
    
    
    dest.close()

if __name__ == "__main__":
    parseDirs(sys.argv[1],"000")
    parseDirs(sys.argv[1],"001")
