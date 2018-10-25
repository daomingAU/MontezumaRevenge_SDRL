import os
import subprocess
import psutil
import signal

actionlist = ['move']
fluentlist = ['at','cost','picked']

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def extract_result():
    with open('result.tmp') as infile:
        content = infile.readlines()
        for info in content:
            if info.startswith('Answer'):
                result = content[content.index(info)+1].strip('\n')
                return result.split(" ")
    return None

def get_type(inputstring):
    prefix = inputstring[:inputstring.find('(')]
    for act in actionlist:
        if act == prefix:
            return "action"
    for flu in fluentlist:
        if flu == prefix:
            return "fluent"


def split_time(result):
    splittedtuple = []
    for res in result:
        if "=" in res:
            equalposition = res.rfind('=')
            value = res[equalposition+1:]
            timestamp = res[5:equalposition-1]
            atom="cost="+value
            splittedtuple.append((int(timestamp), atom, get_type(res)))
        else:    
            index = res.rfind(',')
            timestamp = res[index+1:][:-1]
            atompart = res[:index]
            atom = "".join(atompart)+")"
            splittedtuple.append((int(timestamp), atom, get_type(res)))
    return splittedtuple

def construct_lists(split,step):
    actions = ''
    fluents = ''
    for s in split:
        if s[0] == step:
            if s[2] == 'action':
                actions = actions+s[1]+' '
            else:
                fluents = fluents+s[1]+' '
    return actions, fluents

def compute_plan(clingopath = None, initial = "", goal = "", planning = "", qvalue = "", constraint = "", printout = False):
    if printout:
        print "Generate symbolic plan..."
    if initial == "":
        initial = "initial.lp"
    if planning == "":
        planning = "taxi.lp"
    if goal == "":
        goal = "goal.lp"
    show = "show.lp"
    files = initial+" "+planning+" "+qvalue+" "+constraint+" "+goal+" "+show

    if clingopath != None:
    #    os.system(clingopath+" "+files+" --timelimit=300 > result.tmp")
        clingconprocess = subprocess.Popen(clingopath+" "+files+" --time-limit=180 > result.tmp",shell=True)
        p = psutil.Process(clingconprocess.pid)
        try:
            p.wait(timeout=360)
        except psutil.TimeoutExpired:
        #    os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            p.kill()
            print bcolors.FAIL+"Planning timeout. Process killed."+bcolors.ENDC
            return None
    else:
        os.system(clingopath+" "+files+" > result.tmp")
   
    result = extract_result()
    if result == None:
        return None
    split = split_time(result)
    #print split
    maxtime = int(sorted(split, key=lambda item: item[0], reverse=True)[0][0])
    if printout:
        print "Find a plan in", maxtime, "steps"
    plan_trace= []
    for i in range(1, maxtime+1):
        actions, fluents = construct_lists(split, i)
        plan_trace.append((i,actions,fluents))
        if printout is True:
            print bcolors.OKBLUE+"[TIME STAMP]", i, bcolors.ENDC
            if fluents != '':
                print bcolors.OKGREEN+"[FLUENTS]"+ bcolors.ENDC, fluents
            if actions != '':
                print bcolors.OKGREEN+"[ACTIONS]"+bcolors.ENDC, bcolors.BOLD+actions+bcolors.ENDC
    return plan_trace
#compute_plan()       
    