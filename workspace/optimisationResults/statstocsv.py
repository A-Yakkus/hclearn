from pstats import Stats
import io
import sys

if __name__ == '__main__':
    args = sys.argv
    r = io.StringIO()
    if len(args) > 1:
        Stats(args[1], stream=r).strip_dirs().print_stats("cffun|DGStateAlan|go.py|gui|hcq|learnWeights|location.py|makeMaze|paths|plotPlaceCells|rbm|SURFExtractor")
    else:
        input("Profile Results:")
    r1 = r.getvalue()
    r2 = 'ncalls'+r1.split('ncalls')[-1]
    r2a = r2.split("\n")
    r2a[1] = r2a[1].replace(",","").replace("\"","")
    new_str = ""
    for i in r2a:
        new_str +=i+"\n"
    r2 = new_str
    #r2 = 'ncalls'+r2.split('ncalls')[-1]
    r3 = "\n".join([','.join(line.rstrip().split(None, 5)) for line in r2.split("\n")])
    if(args[2] is None):
        fname = input("File name:")
    else:
        fname = args[2]
    with open(fname, 'w+') as f:
        f.write(r3)
        f.close()
