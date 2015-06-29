# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

%pylab inline
import csv
import numpy as np
import random as rd
import matplotlib.pyplot as plt
import multiprocessing
import scipy.stats as stat
from joblib import Parallel, delayed

# <codecell>

##  Set the cutoff between training and testing.  Training will be everything up to AND INCLUDING
##  the date specified.
cutYear = 2011
cutMonth = 1
cutDay = 19

##  Specify the number of cores for parallelization.
##  Specify the number of iterations to simulate each game.  More iterations will produce a more accurate 
##  result at the cost of greater processing time.
nCores = 4
nIters = 10000

# <codecell>

##  Load the player log (contains player level data)
playerLogNames = [0]
playerLog = [0]*26769
with open('Data/2011-playerlogs.csv', 'rb') as f:
    reader = csv.reader(f)
    i = 0
    for row in reader:
        if i==0:
            playerLogNames = row
        else:
            playerLog[i-1] = row
        i = i+1

# <codecell>

##  Load the plays log (contains play by play)
playsLogNames = [0]
playsLog = [0]*634765
with open('Data/2011-plays.csv', 'rb') as ff:
    reader = csv.reader(ff)
    i = 0
    for row in reader:
        if i==0:
            playsLogNames = row
        else:
            playsLog[i-1] = row
        i = i+1

# <codecell>

##  Load the team log (contains team level data)
teamLogNames = [0]
teamLog = [0]*2622
with open('Data/2011-teamlogs.csv', 'rb') as fff:
    reader = csv.reader(fff)
    i = 0
    for row in reader:
        if i==0:
            teamLogNames = row
        else:
            teamLog[i-1] = row
        i = i+1
        
teamLog = teamLog[0:2460]

# <codecell>

def calcTime(year, month, day):
    thyme = 0
    if month == 10:
        thyme = day - 25
    if month == 11:
        thyme = 6 + day
    if month == 12:
        thyme = 6 + 30 + day
    if month == 1:
        thyme = 6 + 30 + 31 + day
    if month == 2:
        thyme = 6 + 30 + 31 + 31 + day
    if month == 3 and (year % 4) == 0:
        thyme = 6 + 30 + 31 + 31 + 29 + day
    if month == 3 and (year % 4) != 0:
        thyme = 6 + 30 + 31 + 31 + 28 + day
    if month == 4 and (year % 4) == 0:
        thyme = 6 + 30 + 31 + 31 + 29 + 31 + day
    if month == 4 and (year % 4) != 0:
        thyme = 6 + 30 + 31 + 31 + 28 + 31 + day
        
    return thyme

cutTime = calcTime(cutYear, cutMonth, cutDay)

# <codecell>

## Turn a set of values into cumulative proportions of the sum

def ratioMat(inMat):
    length = len(inMat)
    total = float(sum(inMat))
    ratMat = [0]*length
    cumRat = 0.0
    for i in range(0, length):
        cumRat = cumRat + (float(inMat[i]) / total)
        ratMat[i] = cumRat
    
    return ratMat

# <codecell>

## Given a set with cumulative proportions, select an element at random

def rollDice(inMat):
    length = len(inMat)
    
    u = rd.random()
    indexi = 0
    for i in range(0, length):
        if u > inMat[i]:
            indexi += 1
    return indexi  

# <codecell>

## Game list with dates and spread
## [Home, Away, Date, Spread]
gameList = [0]*1230
counter = 0
for i in range(0, 2460):
    if int(teamLog[i][7]) == 1:
        year = teamLog[i][3][:4]
        month = teamLog[i][3][4:6]
        day = teamLog[i][3][6:8]
        thyme = calcTime(int(year), int(month), int(day))
        home = teamLog[i][5]
        away = teamLog[i][6]
        spread = teamLog[i][15]
        margin = teamLog[i][25]
        gameList[counter] = [year, month, day, thyme, home, away, spread, margin]
        counter = counter + 1

# <codecell>

## teams: Array of team names
tm = set()
for i in range(0, 26769):
    tm.add(playerLog[i][6])
nTeam = len(tm)
teams = [0]*nTeam
for i in range(0, nTeam):
    teams[i] = tm.pop()

# <codecell>

## playerList: List of lists, each list contains the player names for a given team
playerList = [0]*nTeam
for i in range(0,nTeam):
    pList = set()
    for j in range(0, 26769):
        if playerLog[j][6]==teams[i]:
            pList.add(playerLog[j][5])
    pL = [0]*len(pList)
    for k in range(0, len(pList)):
        pL[k] = pList.pop()
    playerList[i] = pL

# <codecell>

##  Number of home and away games each team is trained on
numGamesTrain = [0]*nTeam
for i in range(0, nTeam):
    numGamesTrain[i] = [0, 0]

for i in range(0, 1230):
    if gameList[i][3] <= cutTime:
        t = teams.index(gameList[i][4])
        o = teams.index(gameList[i][5])
        numGamesTrain[t][0] = numGamesTrain[t][0] + 1
        numGamesTrain[o][1] = numGamesTrain[o][1] + 1

# <codecell>

## shotMatrix: nTeam x 4 x 7
##            [rim home atmp, short2 home atmp, mid2 home atmp, long2 home atmp, short3 home atmp, mid3 home atmp, long3 home atmp,
##             rim home conv, short2 home conv, mid2 home conv, long2 home conv, short3 home conv, mid3 home conv, long3 home conv,
##             rim away atmp, short2 away atmp, mid2 away atmp, long2 away atmp, short3 away atmp, mid3 away atmp, long3 away atmp,
##             rim away conv, short2 away conv, mid2 away conv, long2 away conv, short3 away conv, mid3 away conv, long3 away conv]

shotMatrix = [0]*nTeam
for i in range(0, nTeam):
    shotMatrix[i] = [[0]*7, [0]*7, [0]*7, [0]*7]

for i in range(0, 26769):
    year = int(playerLog[i][3][:4])
    month = int(playerLog[i][3][4:6])
    day = int(playerLog[i][3][6:8])
    thyme = calcTime(year, month, day)
    if thyme <= cutTime and playerLog[i][8] == '1':
        t = teams.index(playerLog[i][6])
    
        shotMatrix[t][0][0] += int(playerLog[i][40])
        shotMatrix[t][0][1] += int(playerLog[i][43])
        shotMatrix[t][0][2] += int(playerLog[i][46])
        shotMatrix[t][0][3] += int(playerLog[i][49])
        shotMatrix[t][0][4] += int(playerLog[i][31])
        shotMatrix[t][0][5] += int(playerLog[i][34])
        shotMatrix[t][0][6] += int(playerLog[i][37])
    
        shotMatrix[t][1][0] += int(playerLog[i][41])
        shotMatrix[t][1][1] += int(playerLog[i][44])
        shotMatrix[t][1][2] += int(playerLog[i][47])
        shotMatrix[t][1][3] += int(playerLog[i][50])
        shotMatrix[t][1][4] += int(playerLog[i][32])
        shotMatrix[t][1][5] += int(playerLog[i][35])
        shotMatrix[t][1][6] += int(playerLog[i][38])
        
    if thyme <= cutTime and playerLog[i][8] == '0':
        t = teams.index(playerLog[i][6])
    
        shotMatrix[t][2][0] += int(playerLog[i][40])
        shotMatrix[t][2][1] += int(playerLog[i][43])
        shotMatrix[t][2][2] += int(playerLog[i][46])
        shotMatrix[t][2][3] += int(playerLog[i][49])
        shotMatrix[t][2][4] += int(playerLog[i][31])
        shotMatrix[t][2][5] += int(playerLog[i][34])
        shotMatrix[t][2][6] += int(playerLog[i][37])
    
        shotMatrix[t][3][0] += int(playerLog[i][41])
        shotMatrix[t][3][1] += int(playerLog[i][44])
        shotMatrix[t][3][2] += int(playerLog[i][47])
        shotMatrix[t][3][3] += int(playerLog[i][50])
        shotMatrix[t][3][4] += int(playerLog[i][32])
        shotMatrix[t][3][5] += int(playerLog[i][35])
        shotMatrix[t][3][6] += int(playerLog[i][38])

# <codecell>

## rebMatrix: nTeam x 2 x 4
##            [def reb% field home, off reb% field home, def reb% ft home, off reb% ft home,
##             def reb% field away, off reb% field away, def reb% ft away, off reb% ft away]

## possMatrix: nTeam x 2 x 2
##            [TO% home, shooting foul% home,
##             TO% away, shooting foul% away]

## ftMatrix: nTeam x 2
##           [home ft%, away ft%]

rebMat = [0]*nTeam
for i in range(0, nTeam):
    rebMat[i] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    
possMat = [0]*nTeam
for i in range(0, nTeam):
    possMat[i] = [0, 0, 0, 0, 0, 0, 0, 0]
    
ftMat = [0]*nTeam
for i in range(0, nTeam):
    ftMat[i] = [0, 0, 0, 0]

count = 0
for i in range(0, 634765):
    year = int(playsLog[i][1][:4])
    month = int(playsLog[i][1][4:6])
    day = int(playsLog[i][1][6:8])
    thyme = calcTime(year, month, day)
    playType = playsLog[i][9]
    
    ####    REBOUNDING DATA    ####
    if thyme <= cutTime and (playType == 'Rebound'):
        t = teams.index(playsLog[i][2])
        o = teams.index(playsLog[i][3])
        home = teams.index(playsLog[i][1][9:])
        ##  If rebound is at home
        if t == home:
            ##  If rebound is off a field goal attempt
            if playsLog[i-1][10] == 'Two' or playsLog[i-1][10] == 'Three':
                ##  If it's a defensive rebound
                if playsLog[i][2] != playsLog[i-1][2]:
                    rebMat[t][0] = rebMat[t][0] + 1
                    rebMat[t][1] = rebMat[t][1] + 1
                    rebMat[o][3] = rebMat[o][3] + 1
                ##  Else, offensive rebound
                else:
                    rebMat[t][2] = rebMat[t][2] + 1
                    rebMat[t][3] = rebMat[t][3] + 1
                    rebMat[o][1] = rebMat[o][1] + 1
            ##  Else, rebound is off a free throw       
            else:
                ##  Defensive rebound
                if playsLog[i][2] != playsLog[i-1][2]:
                    rebMat[t][4] = rebMat[t][4] + 1
                    rebMat[t][5] = rebMat[t][5] + 1
                    rebMat[o][7] = rebMat[o][7] + 1
                ##  Offensive rebound
                else:
                    rebMat[t][6] = rebMat[t][6] + 1
                    rebMat[t][7] = rebMat[t][7] + 1
                    rebMat[o][5] = rebMat[o][5] + 1
                    
        ##  Else, rebounder is away
        else:
            ##  If rebound is off a field goal attempt
            if playsLog[i-1][10] == 'Two' or playsLog[i-1][10] == 'Three':
                ##  If it's a defensive rebound
                if playsLog[i][2] != playsLog[i-1][2]:
                    rebMat[t][8] = rebMat[t][8] + 1
                    rebMat[t][9] = rebMat[t][9] + 1
                    rebMat[o][11] = rebMat[o][11] + 1
                ##  Else, offensive rebound
                else:
                    rebMat[t][10] = rebMat[t][10] + 1
                    rebMat[t][11] = rebMat[t][11] + 1
                    rebMat[o][9] = rebMat[o][9] + 1
            ##  Else, rebound is off a free throw       
            else:
                ##  Defensive rebound
                if playsLog[i][2] != playsLog[i-1][2]:
                    rebMat[t][12] = rebMat[t][12] + 1
                    rebMat[t][13] = rebMat[t][13] + 1
                    rebMat[o][15] = rebMat[o][15] + 1
                ##  Offensive rebound
                else:
                    rebMat[t][14] = rebMat[t][14] + 1
                    rebMat[t][15] = rebMat[t][15] + 1
                    rebMat[o][13] = rebMat[o][13] + 1                    
    
    ####    POSSESSION DATA    ####
    quarter = int(playsLog[i][4])
    if thyme <= cutTime and (quarter <= 4):
        t = teams.index(playsLog[i][2])
        o = teams.index(playsLog[i][3])
        home = teams.index(playsLog[i][1][9:])
        ##  If possession is for the home team
        if t == home:
            # Made shot
            if playsLog[i][9] == 'Shot' and (playsLog[i][10] == 'Two' or playsLog[i][10] == 'Three') \
               and playsLog[i][11] == '1':
                possMat[t][0] = possMat[t][0] + 1
            # Missed shot, def rebound
            if playsLog[i][17] == 'Defensive' and (playsLog[i-1][10] == 'Two' or \
                                                   playsLog[i-1][10] == 'Three'):
                possMat[o][1] = possMat[o][1] + 1
            # Turnover
            if playsLog[i][9] == 'Turnover':
                possMat[t][2] = possMat[t][2] + 1
            # Shooting foul
            if playsLog[i][9] == 'Foul Drawn' and playsLog[i][19] == 'Shooting' and \
               playsLog[i+1][10] == 'Normal FT':
                possMat[t][3] = possMat[t][3] + 1
                
        ##  Else, possession is for the away team
        else:
            # Made shot
            if playsLog[i][9] == 'Shot' and (playsLog[i][10] == 'Two' or playsLog[i][10] == 'Three') \
               and playsLog[i][11] == '1':
                possMat[t][4] = possMat[t][4] + 1
            # Missed shot, def rebound
            if playsLog[i][17] == 'Defensive' and (playsLog[i-1][10] == 'Two' or \
                                                   playsLog[i-1][10] == 'Three'):
                possMat[o][5] = possMat[o][5] + 1
            # Turnover
            if playsLog[i][9] == 'Turnover':
                possMat[t][6] = possMat[t][6] + 1
            # Shooting foul
            if playsLog[i][9] == 'Foul Drawn' and playsLog[i][19] == 'Shooting' and \
               playsLog[i+1][10] == 'Normal FT':
                possMat[t][7] = possMat[t][7] + 1
                
    ####    FREE THROW DATA    ####        
    if thyme <= cutTime and (playsLog[i][10] == 'Normal FT' or playsLog[i][10] == 'Bonus FT'):
        t = teams.index(playsLog[i][2])
        home = teams.index(playsLog[i][1][9:])
        ##  If free throw is for the home team
        if t == home:
            ftMat[t][1] = ftMat[t][1] + 1
            if playsLog[i][11] == '1':
                ftMat[t][0] = ftMat[t][0] + 1
        ##  Else, free throw for the away team
        else:
            ftMat[t][3] = ftMat[t][3] + 1
            if playsLog[i][11] == '1':
                ftMat[t][2] = ftMat[t][2] + 1

##  Change counts to proportions
rebMatrix = [0]*nTeam
for i in range(0, nTeam):
    rebMatrix[i] = [0, 0]
    rebMatrix[i][0] = [0, 0, 0, 0]
    rebMatrix[i][1] = [0, 0, 0, 0]
    rebMatrix[i][0][0] = float(rebMat[i][0]) / rebMat[i][1]
    rebMatrix[i][0][1] = float(rebMat[i][2]) / rebMat[i][3]
    rebMatrix[i][0][2] = float(rebMat[i][4]) / rebMat[i][5]
    rebMatrix[i][0][3] = float(rebMat[i][6]) / rebMat[i][7]
    rebMatrix[i][1][0] = float(rebMat[i][8]) / rebMat[i][9]
    rebMatrix[i][1][1] = float(rebMat[i][10]) / rebMat[i][11]
    rebMatrix[i][1][2] = float(rebMat[i][12]) / rebMat[i][13]
    rebMatrix[i][1][3] = float(rebMat[i][14]) / rebMat[i][15]
    
possMatrix = [0]*nTeam
for i in range(0, nTeam):
    possMatrix[i] = [0, 0]
    possMatrix[i][0] = [0, 0]
    possMatrix[i][1] = [0, 0]
    possMatrix[i][0][0] = float(possMat[i][2]) / np.sum(possMat[i][0:3])
    possMatrix[i][0][1] = float(possMat[i][3]) / np.sum(possMat[i][0:3])
    possMatrix[i][1][0] = float(possMat[i][6]) / np.sum(possMat[i][4:7])
    possMatrix[i][1][1] = float(possMat[i][7]) / np.sum(possMat[i][4:7])
    
ftMatrix = [0]*nTeam
for i in range(0, nTeam):
    ftMatrix[i] = [0, 0]
    ftMatrix[i][0] = float(ftMat[i][0]) / ftMat[i][1]
    ftMatrix[i][1] = float(ftMat[i][2]) / ftMat[i][3]

# <codecell>

## timeMat: nTeam x 2
##          [time/pos home, time/pos away]

timeMat = [0]*nTeam
for i in range(0, nTeam):
    timeMat[i] = [0, 0]
    timeMat[i][0] = float(numGamesTrain[i][0])*48.0*60.0*.5 / \
                    float(np.sum(possMat[i][0:3]))
    timeMat[i][1] = float(numGamesTrain[i][1])*48.0*60.0*.5 / \
                    float(np.sum(possMat[i][4:7]))

# <codecell>

## defConstants: nTeam x 7
##            [rim def, short2 def, mid2 def, long2 def, short3 def, mid3 def, long3 def]
##            NOTE: Positive constant implies better than average defense.  For example, if Team A converts
##                  47% of short 3's and their opponent Team B has a short 3 defensive modifier of .02, then
##                  Team A's expected short 3 conversion rate becomes 45% versus team B.


## DefMatrix is nTeam x nGame x 8.  For a given team, each entry is the opponent followed by the opponents
##     conversion rate at each of the seven distances for that game.
defMatrix = [0]*nTeam
for i in range(0, nTeam):
    numGames = numGamesTrain[i][0] + numGamesTrain[i][1]
    defMatrix[i] = [0]*(numGames)
    for j in range(0, numGames):
        defMatrix[i][j] = [0]*8

counter = [0]*nTeam        
for i in range(0, 2460):
    year = int(teamLog[i][3][:4])
    month = int(teamLog[i][3][4:6])
    day = int(teamLog[i][3][6:8])
    thyme = calcTime(year, month, day)
    if thyme <= cutTime:
        t = teams.index(teamLog[i][6])
        defMatrix[t][counter[t]][0] = teamLog[i][5]
        
        if float(teamLog[i][56]) != 0:
            defMatrix[t][counter[t]][1] = float(teamLog[i][57]) / float(teamLog[i][56])
        else:
            defMatrix[t][counter[t]][1] = 'NA'
        if float(teamLog[i][59]) != 0:
            defMatrix[t][counter[t]][2] = float(teamLog[i][60]) / float(teamLog[i][59])
        else:
            defMatrix[t][counter[t]][1] = 'NA'
        if float(teamLog[i][62]) != 0:
            defMatrix[t][counter[t]][3] = float(teamLog[i][63]) / float(teamLog[i][62])
        else:
            defMatrix[t][counter[t]][1] = 'NA'
        if float(teamLog[i][65]) != 0:
            defMatrix[t][counter[t]][4] = float(teamLog[i][66]) / float(teamLog[i][65])
        else:
            defMatrix[t][counter[t]][1] = 'NA'
        if float(teamLog[i][47]) != 0:
            defMatrix[t][counter[t]][5] = float(teamLog[i][48]) / float(teamLog[i][47])
        else:
            defMatrix[t][counter[t]][1] = 'NA'
        if float(teamLog[i][50]) != 0:
            defMatrix[t][counter[t]][6] = float(teamLog[i][51]) / float(teamLog[i][50])
        else:
            defMatrix[t][counter[t]][1] = 'NA'
        if float(teamLog[i][53]) != 0:
            defMatrix[t][counter[t]][7] = float(teamLog[i][54]) / float(teamLog[i][53])
        else:
            defMatrix[t][counter[t]][1] = 'NA'
        
        counter[t] = counter[t] + 1

        
## Calculate defensive contants.  For a given shot distance, calculate average performance for the opponent.
##     Compare that average performance to the defensive performance; positive implies better than average
##     defense.  Average across all games.
defConstants = [0]*nTeam
for i in range(0, nTeam):
    numGames = len(defMatrix[i])
    defConstants[i] = [0, 0, 0, 0, 0, 0, 0]
    
    ## Rim
    defTotal = 0.0
    count = 0
    for j in range(0, numGames):
        if defMatrix[i][j][1] != 'NA':
            o = teams.index(defMatrix[i][j][0])
            avePerf = float(shotMatrix[o][1][0] + shotMatrix[o][3][0]) / \
                            (shotMatrix[o][0][0] + shotMatrix[o][2][0])
            defTotal = defTotal + (avePerf - defMatrix[i][j][1])
            count = count + 1
    defConstants[i][0] = defTotal / count
    
    ## Short 2
    defTotal = 0.0
    count = 0
    for j in range(0, numGames):
        if defMatrix[i][j][2] != 'NA':
            o = teams.index(defMatrix[i][j][0])
            avePerf = float(shotMatrix[o][1][1] + shotMatrix[o][3][1]) / \
                            (shotMatrix[o][0][1] + shotMatrix[o][2][1])
            defTotal = defTotal + (avePerf - defMatrix[i][j][2])
            count = count + 1
    defConstants[i][1] = defTotal / count
    
    ## Mid 2
    defTotal = 0.0
    count = 0
    for j in range(0, numGames):
        if defMatrix[i][j][3] != 'NA':
            o = teams.index(defMatrix[i][j][0])
            avePerf = float(shotMatrix[o][1][2] + shotMatrix[o][3][2]) / \
                            (shotMatrix[o][0][2] + shotMatrix[o][2][2])
            defTotal = defTotal + (avePerf - defMatrix[i][j][3])
            count = count + 1
    defConstants[i][2] = defTotal / count
    
    ## Long 2
    defTotal = 0.0
    count = 0
    for j in range(0, numGames):
        if defMatrix[i][j][4] != 'NA':
            o = teams.index(defMatrix[i][j][0])
            avePerf = float(shotMatrix[o][1][3] + shotMatrix[o][3][3]) / \
                            (shotMatrix[o][0][3] + shotMatrix[o][2][3])
            defTotal = defTotal + (avePerf - defMatrix[i][j][4])
            count = count + 1
    defConstants[i][3] = defTotal / count
    
    ## Short 3
    defTotal = 0.0
    count = 0
    for j in range(0, numGames):
        if defMatrix[i][j][5] != 'NA':
            o = teams.index(defMatrix[i][j][0])
            avePerf = float(shotMatrix[o][1][4] + shotMatrix[o][3][4]) / \
                            (shotMatrix[o][0][4] + shotMatrix[o][2][4])
            defTotal = defTotal + (avePerf - defMatrix[i][j][5])
            count = count + 1
    defConstants[i][4] = defTotal / count
    
    ## Mid 3
    defTotal = 0.0
    count = 0
    for j in range(0, numGames):
        if defMatrix[i][j][6] != 'NA':
            o = teams.index(defMatrix[i][j][0])
            avePerf = float(shotMatrix[o][1][5] + shotMatrix[o][3][5]) / \
                            (shotMatrix[o][0][5] + shotMatrix[o][2][5])
            defTotal = defTotal + (avePerf - defMatrix[i][j][6])
            count = count + 1
    defConstants[i][5] = defTotal / count
    
    ## Long 3
    defTotal = 0.0
    count = 0
    for j in range(0, numGames):
        if defMatrix[i][j][7] != 'NA':
            o = teams.index(defMatrix[i][j][0])
            avePerf = float(shotMatrix[o][1][6] + shotMatrix[o][3][6]) / \
                            (shotMatrix[o][0][6] + shotMatrix[o][2][6])
            defTotal = defTotal + (avePerf - defMatrix[i][j][7])
            count = count + 1
    defConstants[i][6] = defTotal / count

# <codecell>

#########################################################
#                                                       #
#                                                       #
#                                                       #
#                                                       #
#                                                       #
#           E N D   P R E P R O C E S S I N G           #
#                                                       #
#                                                       #
#                                                       #
#                                                       #
#                                                       #
#########################################################

# <codecell>

##  Function to simulate a single basketball game given two teams

def bballSim(homeTeam, awayTeam):
    home = teams.index(homeTeam)
    away = teams.index(awayTeam)
    
    ####  Team A shot and conversion matrices
    shotMatTeamA = shotMatrix[home][0]
    convMatTeamA = [0]*7
    for i in range(0, 7):
        convMatTeamA[i] = float(shotMatrix[home][1][i]) / float(shotMatTeamA[i])
        convMatTeamA[i] = convMatTeamA[i] - defConstants[away][i]
            
                
    ####  Team A shot and conversion matrices
    shotMatTeamB = shotMatrix[home][2]
    convMatTeamB = [0]*7
    for i in range(0, 7):
        convMatTeamB[i] = float(shotMatrix[home][3][i]) / float(shotMatTeamB[i])
        convMatTeamB[i] = convMatTeamB[i] - defConstants[home][i]
    
    ####  Team A and B turnover and rebound matricies.   
    timeA = timeMat[home][1]
    toA = possMatrix[home][0][0]
    foulA = possMatrix[home][0][1]
    rebA = (rebMatrix[home][0][1] + (1 - rebMatrix[away][1][0])) / 2.0
    rebFtA = (rebMatrix[home][0][3] + (1 - rebMatrix[away][1][2])) / 2.0
    ftA = ftMatrix[home][0]
    
    timeB = timeMat[away][1]
    toB = possMatrix[away][1][0]
    foulB = possMatrix[away][1][1]
    rebB = (rebMatrix[away][1][1] + (1 - rebMatrix[home][0][0])) / 2.0
    rebFtB = (rebMatrix[away][1][3] + (1 - rebMatrix[home][0][2])) / 2.0
    ftB = ftMatrix[away][1]

    ####  Conversions for processing speed
    shotMatTeamA = ratioMat(shotMatTeamA)
    shotMatTeamB = ratioMat(shotMatTeamB)
    
    scoreA = 0
    scoreB = 0
    possession = ''
    initPoss = ''
    for quarter in range(1,16):
        thyme = 0.0
        if quarter <= 4:
            thyme = 720.0
        else:
            thyme = 300.0
            
        
        ####  Figure out who has possession at the beginning of each quarter  ####
        if quarter == 1:
            u = rd.random()
            possIndex = 0
            if u > .5:
                possIndex = 1
            if possIndex == 0:
                initPoss = 'A'
            if possIndex == 1:
                initPoss = 'B'
            possession = initPoss
        if quarter == 2 or quarter == 3:
            if initPoss == 'A':
                possession = 'B'
            if initPoss == 'B':
                possession = 'A'
        if quarter == 4:
            possession = initPoss
        ####  END  Figuring out possession  ####

        ####  Run Successive possessions until the quarter ends  ####
        while thyme > 0:
            delta = [0]*3
            if possession == 'A':
                
                ####  RUN TEAM A POSSESSION  ####
                    
                    ####  Check to see if there is a turnover
                u = rd.random()
                if u < toA:
                    delta[0] = 0
                    delta[1] = 0
                    delta[2] = rd.gauss(timeA, 3)
                    possession = 'B'
                                        
                ####  Check to see if there is a shooting foul
                if u < (toA + foulA):
                    w = rd.random()
                    if w < ftA:
                        delta[0] = delta[0] + 1
                    w = rd.random()
                    if w < ftA:
                        delta[0] = delta[0] + 1
                        delta[1] = 0
                        delta[2] = rd.gauss(timeA, 3)
                        possession = 'B'
                    else:
                        v = rd.random()
                        if v < rebFtA:
                            delta[2] = rd.gauss(timeA, 3)
                            possession = 'A'
                        else:
                            delta[2] = rd.gauss(timeA, 3)
                            possession = 'B'
                
                ####  Play will end in a made shot or a defensive rebound
                else:
                    
                    while True:
                        
                        ####  Check the shot type
                        shotTypeIndex = rollDice(shotMatTeamA)
                        w = rd.random()
                    
                        ####  See if the shot went in
                        if w < convMatTeamA[shotTypeIndex]:
                            if shotTypeIndex <= 3:
                                delta[0] = 2
                                delta[1] = 0
                                delta[2] = rd.gauss(timeA, 3)
                                possession = 'B'
                                break
                            else:
                                delta[0] = 3
                                delta[1] = 0
                                delta[2] = rd.gauss(timeA, 3)
                                possession = 'B'
                                break
                    
                        #### If not, see who gets the rebound
                        v = rd.random()
                        if v > rebA:
                            delta[0] = 0
                            delta[1] = 0
                            delta[2] = rd.gauss(timeA, 3)
                            possession = 'B'
                            break
                ####  END TEAM A POSSESSION  ####
                
            else:
             
                ####  RUN TEAM B POSSESSION  ####
                    
                    ####  Check to see if there is a turnover
                u = rd.random()
                if u < toB:
                    delta[0] = 0
                    delta[1] = 0
                    delta[2] = rd.gauss(timeB, 3)
                    possession = 'A'
                                        
                ####  Check to see if there is a shooting foul
                if u < (toB + foulB):
                    w = rd.random()
                    if w < ftB:
                        delta[1] = delta[1] + 1
                    w = rd.random()
                    if w < ftB:
                        delta[1] = delta[1] + 1
                        delta[0] = 0
                        delta[2] = rd.gauss(timeB, 3)
                        possession = 'A'
                    else:
                        v = rd.random()
                        if v < rebFtB:
                            delta[2] = rd.gauss(timeB, 3)
                            possession = 'B'
                        else:
                            delta[2] = rd.gauss(timeB, 3)
                            possession = 'A'
                
                ####  Play will end in a made shot or a defensive rebound
                else:
                    
                    while True:
                        
                        ####  Check the shot type
                        shotTypeIndex = rollDice(shotMatTeamB)
                        w = rd.random()
                    
                        ####  See if the shot went in
                        if w < convMatTeamB[shotTypeIndex]:
                            if shotTypeIndex <= 3:
                                delta[0] = 0
                                delta[1] = 2
                                delta[2] = rd.gauss(timeB, 3)
                                possession = 'A'
                                break
                            else:
                                delta[0] = 0
                                delta[1] = 3
                                delta[2] = rd.gauss(timeB, 3)
                                possession = 'A'
                                break
                    
                        #### If not, see who gets the rebound
                        v = rd.random()
                        if v > rebB:
                            delta[0] = 0
                            delta[1] = 0
                            delta[2] = rd.gauss(timeB, 3)
                            possession = 'A'
                            break
                ####  END TEAM B POSSESSION  ####
                
            scoreA += delta[0]
            scoreB += delta[1]
            thyme -= delta[2]
        ####  END Running successive possessions ####
        if quarter >= 4 and scoreA != scoreB:
            break
        
    return [scoreA, scoreB, scoreA - scoreB]

# <codecell>

## Games to predict.
predGames = [0]*1230
countPreds = 0
for i in range(0, 1230):
    if gameList[i][3] > cutTime:
        predGames[countPreds] = gameList[i]
        countPreds = countPreds + 1
        
predGames = predGames[0:countPreds]
print countPreds

# <codecell>

###############################################################################################
#                                                                                             #
#                                 R U N   S I M U L A T I O N                                 #
#                                                                                             #
#  Total iterations will be teh number of games predicted on times the number of iterations   #
#  per game.  My 5 year old desktop with four cores can do approximately 2300 iterations per  #
#  second.  Have some idea how much time the simulation will take before running it.          #
#                                                                                             #
###############################################################################################

spreadMatrix = [0]*countPreds
for i in range(0, countPreds):
    homeTeam = predGames[i][4]
    awayTeam = predGames[i][5]

    out = Parallel(n_jobs=nCores)(delayed(bballSim)(homeTeam, awayTeam) for i in range(0, nIters))
    out = zip(*out)
    spreadMatrix[i] = out[2]

# <codecell>

#########################################################
#                                                       #
#                                                       #
#                                                       #
#                                                       #
#                    E X A M P L E S                    #
#                         A N D                         #
#        P R E L I M I N A R Y   A N A L Y S I S        #
#                                                       #
#                                                       #
#                                                       #
#                                                       #
#########################################################

# <codecell>

##  EXAMPLE OUTPUT 1
##  Produce histogram with 1,000 iterations
##  Red verticle line is the mean prediction
game = 0
homeTeam = predGames[game][4]
awayTeam = predGames[game][5]

out1 = Parallel(n_jobs=nCores)(delayed(bballSim)(homeTeam, awayTeam) for i in range(0, 1000))
out1 = zip(*out1)
spreadMat = out1[2]
ave = np.mean(spreadMat)
pointEst = float(np.round(2*ave)) / 2

binCount = 1 + abs(min(set(spreadMat))) + max(set(spreadMat))
plt.hist(spreadMat, bins = binCount, color = 'b')
plt.vlines(pointEst, 0, 45, color = 'r')
font = {'family' : 'serif', 'size' : 12}
plt.rc('font', **font)
plt.title(homeTeam + ' (home) vs. ' + awayTeam + ' (away)\n1,000 Iterations')
plt.xlabel('Spread')
plt.ylabel('Frequency')
plt.show()
#plt.savefig('1000iter.png')

# <codecell>

##  EXAMPLE OUTPUT 2
##  Produce histogram with 10,000 iterations
##  Red verticle line is the mean prediction
game = 0
homeTeam = predGames[game][4]
awayTeam = predGames[game][5]

out1 = Parallel(n_jobs=nCores)(delayed(bballSim)(homeTeam, awayTeam) for i in range(0, 10000))
out1 = zip(*out1)
spreadMat = out1[2]
ave = np.mean(spreadMat)
pointEst = float(np.round(2*ave)) / 2

binCount = 1 + abs(min(set(spreadMat))) + max(set(spreadMat))
plt.hist(spreadMat, bins = binCount, color = 'b')
plt.vlines(pointEst, 0, 400, color = 'r')
plt.title(homeTeam + ' (home) vs. ' + awayTeam + ' (away)\n10,000 Iterations')
plt.xlabel('Spread')
plt.ylabel('Frequency')
plt.show()
#plt.savefig('10000iter.png')

# <codecell>

##  EXAMPLE OUTPUT 3
##  Produce histogram with 100,000 iterations
##  Red verticle line is the mean prediction
game = 0
homeTeam = predGames[game][4]
awayTeam = predGames[game][5]

out1 = Parallel(n_jobs=nCores)(delayed(bballSim)(homeTeam, awayTeam) for i in range(0, 100000))
out1 = zip(*out1)
spreadMat = out1[2]
ave = np.mean(spreadMat)
pointEst = float(np.round(2*ave)) / 2

binCount = 1 + abs(min(set(spreadMat))) + max(set(spreadMat))
plt.hist(spreadMat, bins = binCount, color = 'b')
plt.vlines(pointEst, 0, 3500, color = 'r')
plt.title(homeTeam + ' (home) vs. ' + awayTeam + ' (away)\n100,000 Iterations')
plt.xlabel('Spread')
plt.ylabel('Frequency')
plt.show()
#plt.savefig('100000iter.png')

# <codecell>

##  EXAMPLE OUTPUT 4
##  Produce histogram with 1,000,000 iterations
##  Red verticle line is the mean prediction
game = 0
homeTeam = predGames[game][4]
awayTeam = predGames[game][5]

outMat = [0]*10
for i in range(0, 10):
    out1 = Parallel(n_jobs=nCores)(delayed(bballSim)(homeTeam, awayTeam) for i in range(0, 100000))
    out1 = zip(*out1)
    outMat[i] = out1[2]
    
spreadMat = outMat[0] + outMat[1] + outMat[2] + outMat[3] + outMat[4] + \
            outMat[5] + outMat[6] + outMat[7] + outMat[8] + outMat[9]
ave = np.mean(spreadMat)
pointEst = float(np.round(2*ave)) / 2

minimum = min(set(spreadMat))
maximum = max(set(spreadMat))
binCount = 1 + (-1 * minimum) + maximum
plt.hist(spreadMat, bins = binCount, color = 'b')
plt.vlines(pointEst, 0, 35000, color = 'r')
plt.title(homeTeam + ' (home) vs. ' + awayTeam + ' (away)\n1,000,000 Iterations')
plt.xlabel('Spread')
plt.ylabel('Frequency')
plt.show()
#plt.savefig('1000000iter.png')

# <codecell>

##  (For use with training data) Compare results with the actual outcomes.  How well did we do?

wins = 0
plays = 0
spreads = [0]*countPreds
for i in range(0, countPreds):
    spread = np.median(spreadMatrix[i])
    margin = float(predGames[i][7])
    spreads[i] = spread
            
    if abs(spread) >= 0:
        plays = plays + 1
        if np.sign(spread) == np.sign(margin):
            wins = wins + 1

winRate = float(wins) / plays
zValue = (float(wins) - (float(plays)/2)) / np.sqrt(plays*winRate*(1.0 - winRate))
pValue = 2.0 * (1 - stat.norm.cdf(zValue))
print "My wins:", int(wins) , "  My plays:", plays            
print "My win rate:", winRate
print "\nP value comparing my win rate to 50%:" , pValue
#plt.hist(mySpreads, bins = 12, color = 'r', alpha = .5)
#plt.hist(vegasSpreads, bins = 18, color = 'b', alpha = .5)
#plt.show()

# <codecell>

##  Report on how accurate my predictions were versus the Vegas lines for games in the training set.  Make a
##  histogram of deviation versus frequency.


## Record the differences in prediction versus outcome.
resultsMatrix = [0]*countPreds
for i in range(0, countPreds):
    vegasSpread = (-1 * float(predGames[i][6]))
    #meSpread = np.round(np.mean(spreadMatrix[i]))
    meSpread = np.median(spreadMatrix[i])
    margin = float(predGames[i][7])
    vegasDiff = vegasSpread - margin
    meDiff = meSpread - margin
    
    goodPred = 0
    if abs(vegasDiff) > abs(meDiff):
        goodPred = 1
    if abs(vegasDiff) < abs(meDiff):
        goodPred = -1
    resultsMatrix[i] = [vegasDiff, meDiff, goodPred]

## Take absolute value of differences and truncate anything over 30
accuracy = [0]*countPreds
vegasAccuracy = [0]*countPreds
for i in range(0, countPreds):
    accuracy[i] = abs(resultsMatrix[i][1])
    if accuracy[i] > 30:
        accuracy[i] = 30
    vegasAccuracy[i] = abs(resultsMatrix[i][0])
    if vegasAccuracy[i] > 30:
        vegasAccuracy[i] = 30

## Visualize
binCountMe = len(np.bincount(accuracy))
binCountVegas = len(np.bincount(vegasAccuracy))
plt.hist(accuracy, bins = binCountMe, alpha = .5, color='r', label='Me')
plt.hist(vegasAccuracy, bins = binCountVegas, alpha = .5, color='b', label='Vegas')
plt.title('Absolute Deviation in Spread, Me (Red) vs. Vegas (Blue)')
plt.xlabel('Absolute Deviation in Spread')
plt.ylabel('Frequency')
plt.legend(loc = "upper right", numpoints=1)
#plt.savefig('AbsDev.png')
plt.show()

# <codecell>

## Given a distribution X and value x, find P(x <= X)

def ecdf(vec, value):
    ind = 0
    sVec = list(np.sort(vec))
    length = len(sVec)

    a = 1
    while True:
        if value+a in sVec or a == 100:
            break
        else:
            a = a+1
    if a == 100:
        ind = length
    else:
        ind = sVec.index(value+a)

    return float(ind) / length

# <codecell>

## Given a percentile, return decile

def decile(value):
    if value >= 0 and value < .1:
        return 0
    elif value >= .1 and value < .2:
        return 1
    elif value >= .2 and value < .3:
        return 2
    elif value >= .3 and value < .4:
        return 3
    elif value >= .4 and value < .5:
        return 4
    elif value >= .5 and value < .6:
        return 5
    elif value >= .6 and value < .7:
        return 6
    elif value >= .7 and value < .8:
        return 7
    elif value >= .8 and value < .9:
        return 8
    elif value >= .9 and value <= 1:
        return 9

# <codecell>

deciles = [0]*10
percentile = [0]*countPreds
for i in range(0, countPreds):
    margin = int(predGames[i][7])
    perc = ecdf(spreadMatrix[i], margin)
    dec = decile(perc)
    deciles[dec] = deciles[dec]+1
    percentile[i] = perc

sd = np.sqrt(607.0*.05*.95)
av = 607.0 / 20
plt.hist(percentile, bins = 20, color = 'y', alpha = .8)
plt.hlines(av, 0, 1, color = 'r')
plt.hlines(av + 2.0*sd, 0, 1, color = 'b')
plt.hlines(av - 2.0*sd, 0, 1, color = 'b')
plt.title('20-Quantile Frequency')
plt.xlabel('20-Quantile')
plt.ylabel('Frequency')
#plt.savefig('vigintile.png')
plt.show()
#print deciles

# <codecell>


