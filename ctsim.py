'''
Cluster threshold simulation.
Replication and extension of Woo et al. 2014, NeuroImage 91: 412-419.

Orignal author: Matthew R G Brown
2015 April
'''

from __future__ import print_function
import os, sys, inspect
sys.path.append(os.path.expanduser('~/code/python'))
import compman
from compman import CompMan, InvalidStateError, InvalidMetaparameterError
import nibabel as nib
import numpy as np
import numpy.linalg
import scipy as sc
import scipy.ndimage
import scipy.linalg
import scipy.stats
import pickle
import spmhrf
from collections import OrderedDict
from collections import namedtuple
import pdb
import subprocess
import matplotlib.pyplot as plt

version = '0.0.2'
testMode = False

afniPath = '/home/mbrown/afni/'

# ------------------------------
def getBaseDataPath():
    if 'uname' in dir(os):
        if os.uname()[0] == 'Darwin' and os.uname()[1] == 'lens.local':
            return '/locdata'
        else:
            return '/data'
    else:
        raise Exception('Not coded for this platform.')

# ------------------------------
SubjectAnalysisKey = namedtuple('SubjectAnalysisKey', 'subjectIndex runIndex seedRegionIndex designIndex trialWeightsIndex')
BetweenSubjectAnalysisKey = namedtuple('BetweenSubjectAnalysisKey', 'permutationIndex seedRegionIndex designIndex trialWeightsIndex')
ClusterResultsKey = namedtuple('ClusterResultsKey', 'localPThreshold measureFunction trialWeightsIndex')

# ------------------------------
class SimulationMan(CompMan):
    def __init__(self,cmMetaParam,scratchPath,dataPath):
        cmDesc     = 'simulation'
        cmCodeTag  = os.path.splitext(os.path.split(inspect.getfile(inspect.currentframe()))[1])[0] + '_' + type(self).__name__
        cmBasePath = scratchPath
        CompMan.__init__(self,cmDesc,cmCodeTag,cmMetaParam,cmBasePath=cmBasePath)
        self.dataPath = dataPath
        self.configure()

    def configure_exp1(self):
        # Do not change ordering, it matters for hashTag
        self.makeOutputFilesFunction = self.makeOutputFiles_exp1
        self.makeDataManagers_1()
        self.makeSeedRegionManagers(numSeedRegionsSets=10)
        self.makeSummaryStatsManagers()
        self.makeDesignManagers(numDesigns=10)
        self.makeTrialWeightsManagers_exp1()
        self.makeSubjectManagers()
        self.makeBetweenSubjectManagers()
        self.makeResultsManagers_woo01()

    def configure_exp2(self):
        # Do not change ordering, it matters for hashTag
        self.makeOutputFilesFunction = self.makeOutputFiles_exp1
        self.makeDataManagers_1()
        self.makeSeedRegionManagers(numSeedRegionsSets=10)
        self.makeSummaryStatsManagers()
        self.makeDesignManagers(numDesigns=10)
        self.makeTrialWeightsManagers_exp2()
        self.makeSubjectManagers()
        self.makeBetweenSubjectManagers()
        self.makeResultsManagers_woo01()

    def configure_exp3(self):
        # Do not change ordering, it matters for hashTag
        self.makeOutputFilesFunction = self.makeOutputFiles_exp1
        self.makeDataManagers_1()
        self.makeSeedRegionManagers(numSeedRegionsSets=10)
        self.makeSummaryStatsManagers()
        self.makeDesignManagers(numDesigns=10)
        self.makeTrialWeightsManagers_exp2()
        self.makeSubjectManagers()
        self.makeBetweenSubjectManagers()
        self.makeResultsManagers_emogonogo01()

    def makeDataManagers_1(self):
        self.setConfig('resDataSourceMan'   , ResDataSourceMan('era20_20150415c', self.dataPath))
        self.setConfig('subjectSeriesMan'   , SubjectSeriesMan('era20', self.dataPath, self.resDataSourceMan))
        self.setConfig('maskMan'            , MaskMan('era20_combo1',self.dataPath))
        #self.setConfig('maskMan'            , MaskMan('era20_tae',self.dataPath)) # for testing only

    def makeSeedRegionManagers(self,numSeedRegionsSets):
        self.setConfig('numSeedRegionSets'  , numSeedRegionsSets)
        self.setConfig('seedRegionManList'  , [SeedRegionMan('woo2014_v001',self.maskMan,self.cmBasePath,index)
                                               for index in range(self.numSeedRegionSets)])
        self.setConfig('preprocSeedRegionManList', [PreprocSeedRegionMan(self.resDataSourceMan.preprocTag,man,self.cmBasePath)
                                                    for man in self.seedRegionManList])

    def makeSummaryStatsManagers(self):
        self.setConfig('summaryStatsMan', SummaryStatsMan('mnstd',self.subjectSeriesMan,self.maskMan,self.cmBasePath))

    def makeDesignManagers(self,numDesigns):
        self.setConfig('numDesigns'   , numDesigns)
        self.setConfig('designManList', [DesignMan('ds001',self.cmBasePath,index) for index in range(self.numDesigns)])
        self.setConfig('contrastMan'  , ContrastMan('tt1loc'))

    def makeTrialWeightsManagers_exp1(self):
        self.setConfig('trialWeightsManList', [TrialWeightsMan('tw001_0p8')])

    def makeTrialWeightsManagers_exp2(self):
        self.setConfig('trialWeightsManList', [TrialWeightsMan('tw001_0p0'),
                                               TrialWeightsMan('tw001_0p01'),TrialWeightsMan('tw001_0p05'),TrialWeightsMan('tw001_0p1'),
                                               TrialWeightsMan('tw001_0p3'),TrialWeightsMan('tw001_0p6'),TrialWeightsMan('tw001_0p9'),
                                               TrialWeightsMan('tw001_1p2'),TrialWeightsMan('tw001_1p5')])

    def makeTrialWeightsManagers_unused(self):
        self.setConfig('trialWeightsManList', [TrialWeightsMan('tw001_0p4'),TrialWeightsMan('tw001_0p8'),TrialWeightsMan('tw001_1p2')])

    def makeSubjectManagers(self):
        self.setConfig('simSubjectDataManParam','sd001')
        self.setConfig('simSubjectAnalysisManParam','sa001')
        if testMode:
            (simSubjectDataManDict, subjectAnalysisManDict) = self.makeTestSubjectAnalysisFiles()
        else:
            (simSubjectDataManDict, subjectAnalysisManDict) = self.makeSubjectAnalysisFiles()
        self.setConfig('simSubjectDataManDict' , simSubjectDataManDict)
        self.setConfig('subjectAnalysisManDict', subjectAnalysisManDict)

    def makeBetweenSubjectManagers(self):
        self.setConfig('betweenSubjectDesignMan'   , BetweenSubjectDesignMan('bds001',self.cmBasePath))
        self.setConfig('betweenSubjectContrastMan' , BetweenSubjectContrastMan('offsetloc'))
        self.setConfig('betweenSubjectAnalysisManParam','bsa001')
        if testMode:
            (permutationMan,betweenSubjectAnalysisManDict) = self.makeTestBetweenSubjectAnalysisFiles()
        else:
            (permutationMan,betweenSubjectAnalysisManDict) = self.makeBetweenSubjectAnalysisFiles()
        self.setConfig('permutationMan',permutationMan)
        self.setConfig('betweenSubjectAnalysisManDict',betweenSubjectAnalysisManDict)

    def makeResultsManagers_woo01(self):
        self.setConfig('clusterResultsMan',ClusterResultsMan('woo01',self.betweenSubjectAnalysisManDict,self.cmBasePath))

    def makeResultsManagers_emogonogo01(self):
        self.setConfig('clusterResultsMan',ClusterResultsMan('emogonogo01',self.betweenSubjectAnalysisManDict,self.cmBasePath))

    def makeTestSubjectAnalysisFiles(self):
        '''
        For testing only. Do not use in production.
        Return (simSubjectDataManDict, subjectAnalysisManDict).
        '''
        runIndex          = 0
        seedRegionIndex   = 0
        designIndex       = 0
        trialWeightsIndex = 0
        simSubjectDataManDict = OrderedDict()
        subjectAnalysisManDict = OrderedDict()
        for subjectIndex in [2, 14]:
            simSubjectDataMan =  SimSubjectDataMan(self.simSubjectDataManParam,
                                                   self.subjectSeriesMan.getOutputByIndex(subjectIndex),
                                                   runIndex,
                                                   self.summaryStatsMan,
                                                   self.preprocSeedRegionManList[seedRegionIndex],
                                                   self.designManList[designIndex],
                                                   self.trialWeightsManList[trialWeightsIndex],
                                                   self.cmBasePath) 
            subjectAnalysisMan = SubjectAnalysisMan(self.simSubjectAnalysisManParam,
                                                    simSubjectDataMan,
                                                    self.designManList[designIndex],
                                                    self.contrastMan,
                                                    self.maskMan,
                                                    self.cmBasePath)
            key = SubjectAnalysisKey(subjectIndex,runIndex,seedRegionIndex,designIndex,trialWeightsIndex)
            simSubjectDataManDict[key] = simSubjectDataMan
            subjectAnalysisManDict[key] = subjectAnalysisMan
        return (simSubjectDataManDict, subjectAnalysisManDict)

    def makeSubjectAnalysisFiles(self):
        '''
        Return (simSubjectDataManDict, subjectAnalysisManDict).
        '''
        simSubjectDataManDict = OrderedDict()
        subjectAnalysisManDict = OrderedDict()
        for subjectIndex in range(self.subjectSeriesMan.getNumSubjects()):
            for runIndex in range(self.resDataSourceMan.getNumRuns()):
                for seedRegionIndex in range(self.numSeedRegionSets):
                    for designIndex in range(self.numDesigns):
                        for trialWeightsIndex in range(len(self.trialWeightsManList)):
                            simSubjectDataMan =  SimSubjectDataMan(self.simSubjectDataManParam,
                                                                   self.subjectSeriesMan.getOutputByIndex(subjectIndex),
                                                                   runIndex,
                                                                   self.summaryStatsMan,
                                                                   self.preprocSeedRegionManList[seedRegionIndex],
                                                                   self.designManList[designIndex],
                                                                   self.trialWeightsManList[trialWeightsIndex],
                                                                   self.cmBasePath) 
                            subjectAnalysisMan = SubjectAnalysisMan(self.simSubjectAnalysisManParam,
                                                                    simSubjectDataMan,
                                                                    self.designManList[designIndex],
                                                                    self.contrastMan,
                                                                    self.maskMan,
                                                                    self.cmBasePath)
                            key = SubjectAnalysisKey(subjectIndex,runIndex,seedRegionIndex,designIndex,trialWeightsIndex)
                            simSubjectDataManDict[key] = simSubjectDataMan
                            subjectAnalysisManDict[key] = subjectAnalysisMan
        return (simSubjectDataManDict, subjectAnalysisManDict)

    def makeTestBetweenSubjectAnalysisFiles(self):
        '''
        For testing only. Do not use in production.
        Returns betweenSubjectAnalysisManDict
        '''
        permutationIndex  = 0
        seedRegionIndex   = 0
        designIndex       = 0
        trialWeightsIndex = 0
        betweenSubjectAnalysisManDict = OrderedDict()
        testBetweenSubjectAnalysisMan = BetweenSubjectAnalysisMan(self.betweenSubjectAnalysisManParam,
                                                                  self.subjectAnalysisManDict.values(),
                                                                  self.betweenSubjectDesignMan,
                                                                  self.betweenSubjectContrastMan,
                                                                  self.maskMan,
                                                                  self.cmBasePath)
        key = BetweenSubjectAnalysisKey(permutationIndex,seedRegionIndex,designIndex,trialWeightsIndex)
        betweenSubjectAnalysisManDict[key] = testBetweenSubjectAnalysisMan
        permutationMan = None
        return (permutationMan,betweenSubjectAnalysisManDict)

    def makeBetweenSubjectAnalysisFiles(self):
        '''
        Returns betweenSubjectAnalysisManDict
        '''
        numSubjects = self.subjectSeriesMan.getNumSubjects()
        numRuns     = self.resDataSourceMan.getNumRuns()
        permutationMan = PermutationMan('pm1',self.numSeedRegionSets,self.numDesigns,numSubjects,numRuns,self.cmBasePath)
        permutationMan.makeOutputFiles(forceRebuild=False)
        runIndexPermutationListDict = permutationMan.getOutput()

        betweenSubjectAnalysisManDict = OrderedDict()
        for seedRegionIndex in range(self.numSeedRegionSets):
            for designIndex in range(self.numDesigns):
                runIndexPermutationList = runIndexPermutationListDict[(seedRegionIndex,designIndex)]
                for trialWeightsIndex in range(len(self.trialWeightsManList)):
                    for permutationIndex in range(numRuns):
                        subjectAnalysisManList = []
                        for subjectIndex in range(numSubjects):
                            runIndex = runIndexPermutationList[subjectIndex][permutationIndex]
                            saKey = (subjectIndex,runIndex,seedRegionIndex,designIndex,trialWeightsIndex)
                            subjectAnalysisManList += [self.subjectAnalysisManDict[saKey]]
                        betweenSubjectAnalysisMan = BetweenSubjectAnalysisMan(self.betweenSubjectAnalysisManParam,
                                                                              subjectAnalysisManList,
                                                                              self.betweenSubjectDesignMan,
                                                                              self.betweenSubjectContrastMan,
                                                                              self.maskMan,
                                                                              self.cmBasePath)
                        key = BetweenSubjectAnalysisKey(permutationIndex,seedRegionIndex,designIndex,trialWeightsIndex)
                        betweenSubjectAnalysisManDict[key] = betweenSubjectAnalysisMan
        return (permutationMan,betweenSubjectAnalysisManDict)

    def makeOutputFiles(self,forcerebuild=False,**kwArgs):
        self.makeOutputFilesFunction(forcerebuild,**kwArgs)

    def makeOutputFiles_exp1(self,forcerebuild=False,index=None,numSplits=None):
        '''
        Pass in index and numSplits to do only a portion of the 
        subject analyses (and NO between subject analyses).
        Useful for manually "parallelizing" computations by running
        multiple python instances.
        '''
        for m in self.seedRegionManList:
            m.makeOutputFiles(forcerebuild)
        for m in self.preprocSeedRegionManList:
            m.makeOutputFiles(forcerebuild)
        self.summaryStatsMan.makeOutputFiles(forcerebuild) # takes a long time
        for m in self.designManList:
            m.makeOutputFiles(forcerebuild)
        if index is None or numSplits is None:
            for m in self.subjectAnalysisManDict.values():
                m.makeOutputFiles(forcerebuild)
            for m in self.betweenSubjectAnalysisManDict.values():
                m.makeOutputFiles(forcerebuild)
            self.clusterResultsMan.makeOutputFiles()
        else:
            samList = self.subjectAnalysisManDict.values()
            for m in getPortion(samList,index=index,numSplits=numSplits):
                m.makeOutputFiles(forcerebuild)
            # no between-subject analyses or cluster results

    def computeResidualsSmoothness(self):
        '''
        Computes smoothness (FWHM) using AFNI's 3dFWHMx
        '''
        maskFilePath = self.maskMan.getOutputFile()
        resDataSourceMan = ResDataSourceMan('era20_20150415b', self.dataPath)
        # note: different from self.resDataSourceMan, 'era20_20150415b' removed mean spatial intensity
        subjectSeriesMan = SubjectSeriesMan('era20', self.dataPath, resDataSourceMan)
        # note: different from self.subjectSeriesMan
        subjectTagList = subjectSeriesMan.subjectTagList
        fwhmList = []
        for subjectTag in subjectTagList:
            for residualsFilePath in subjectSeriesMan.getOutput(subjectTag).getOutputFilesList():
                print(residualsFilePath)
                pathTo3dFWHMx = os.path.join(afniPath,'3dFWHMx')
                cmd = '{0} -mask {1} -input {2} -detrend'.format(pathTo3dFWHMx,maskFilePath,residualsFilePath)
                p = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE) 
                (out,err) = p.communicate()
                fwhm = np.mean([float(x) for x in out.split()])
                print(fwhm)
                fwhmList += [fwhm]

# ------------------------------
def getPortion(theList,index,numSplits):
    '''
    Returns approximately 1 / numSplits of list (slice).
    "Left-over" part is tacked onto last portion.
    index in [0,numSplits-1], can be an int or a 2-tuple
    or 2-list
    '''
    if type(index) in (list,tuple):
        index0 = index[0]
        index1 = index[1]
    else:
        index0 = index
        index1 = index
    if index0 < 0 or index0 >= numSplits or index1 < 0 or index1 >= numSplits:
        raise IndexError('index must be in [0 ... numSplits-1]')
    length = len(theList)
    (portionLength,leftOver) = divmod(length,numSplits)
    startIndex = index0*portionLength
    stopIndex  = (index1+1)*portionLength
    if index == numSplits-1:
        stopIndex = length
    return theList[startIndex:stopIndex]

# ------------------------------
class PermutationMan(CompMan):
    def __init__(self,cmMetaParam,numSeedRegionSets,numDesigns,numSubjects,numRuns,cmBasePath):
        cmDesc     = 'runindexpermutation'
        cmCodeTag  = os.path.splitext(os.path.split(inspect.getfile(inspect.currentframe()))[1])[0] + '_' + type(self).__name__
        CompMan.__init__(self,cmDesc,cmCodeTag,cmMetaParam,cmBasePath=cmBasePath)
        self.setConfig('numSeedRegionSets',numSeedRegionSets)
        self.setConfig('numDesigns',numDesigns)
        self.setConfig('numSubjects',numSubjects)
        self.setConfig('numRuns',numRuns)
        self.setConfig('permutationGenerationFunction',None)
        self.configure()
        self.runIndexPermutationListDict = self.loadOutputFiles()

    def configure_pm1(self):
        self.setConfig('permutationGenerationFunction',self.permutationGenerationFunction_v1)

    def getOutput(self):
        return self.runIndexPermutationListDict

    def getOutputFile(self):
        outputPath = self.getOutputPath()
        prefix     = self.getTagPrefix(True)
        fileName   = '{0}.sav'.format(prefix)
        return os.path.join(outputPath,fileName)

    def makeOutputFiles(self,forceRebuild=False):
        print('Generating permutation dict')
        self.makeOutputPath()
        filePath = self.getOutputFile()
        if not os.path.isfile(filePath) or forceRebuild:
            runIndexPermutationListDict = self.permutationGenerationFunction()
            with open(filePath,'w') as f:
                pickle.dump(runIndexPermutationListDict,f)
            print('    Saved {}'.format(filePath))
        else:
            print('    File already exists: {}'.format(filePath))
        self.saveConfigCSVFile(forceRebuild)
        self.runIndexPermutationListDict = self.loadOutputFiles()

    def loadOutputFiles(self):
        filePath = self.getOutputFile()
        if os.path.isfile(filePath):
            with open(filePath,'r') as f:
                runIndexPermutationListDict = pickle.load(f)
        else:
            runIndexPermutationListDict = None
        return runIndexPermutationListDict

    def permutationGenerationFunction_v1(self):
        runIndexPermutationListDict = OrderedDict()
        for seedRegionIndex in range(self.numSeedRegionSets):
            for designIndex in range(self.numDesigns):
                runIndexPermutationList = [list(np.random.permutation(self.numRuns)) for i in range(self.numSubjects)]
                runIndexPermutationListDict[(seedRegionIndex,designIndex)] = runIndexPermutationList
        return runIndexPermutationListDict

# ------------------------------
class SubjectManML(CompMan):
    '''
    Subject manager for residuals output files from Matlab.
    '''
    def __init__(self,experimentMetaparameter,subjectTag,subjectIndex,cmBasePath,resDataSourceMan):
        cmDesc      = 'subject'
        cmCodeTag   = os.path.splitext(os.path.split(inspect.getfile(inspect.currentframe()))[1])[0] + '_' + type(self).__name__
        cmMetaParam = experimentMetaparameter + '_' + subjectTag
        CompMan.__init__(self,cmDesc,cmCodeTag,cmMetaParam,cmBasePath=cmBasePath)
        self.setConfig('experimentMetaparameter' , experimentMetaparameter)
        self.setConfig('subjectTag'              , subjectTag)
        self.setConfig('resDataSourceMan'        , resDataSourceMan)
        self.subjectIndex = subjectIndex
        self.cacheHashTag()

    def getOutputFilesList(self):
        '''
        Returns list of residuals files.
        '''
        return self.resDataSourceMan.getOutputFilesList(self.subjectTag,self.subjectIndex)

# ------------------------------
class SubjectSeriesMan(CompMan):
    def __init__(self,cmMetaParam,cmBasePath,resDataSourceMan=None):
        cmDesc    = 'subjectseries'
        cmCodeTag = os.path.splitext(os.path.split(inspect.getfile(inspect.currentframe()))[1])[0] + '_' + type(self).__name__
        CompMan.__init__(self,cmDesc,cmCodeTag,cmMetaParam,cmBasePath=cmBasePath)
        self.setConfig('resDataSourceMan', resDataSourceMan)
        self.configure()

    def __iter__(self):
        index = 0
        while index < len(self.subjectTagList):
            yield(self.getOutput(self.subjectTagList[index]))
            index += 1

    def configure_era20(self):
        '''
        era20: 20 young adult participants from 2010 emotional
               Go/NoGo fMRI experiment with good data.
        '''
        if self.cmMetaParam != 'era20':
            raise InvalidMetaparameterError(self.cmMetaParam)
        self.setConfig('subjectTagList', [ '20100128_emogonogo',
                                           '20100225_emogonogo',
                                           '20100308_emogonogo_fixed',
                                           '20100615_emogonogo',
                                           '20100616_emogonogo',
                                           '20100621_emogonogo',
                                           '20100623_emogonogo',
                                           '20100624_emogonogo',
                                           '20100716_emogonogo',
                                           '20100719_emogonogo',
                                           '20100723_emogonogo',
                                           '20100922_emogonogo',
                                           '20110217_emogonogo',
                                           '20110302_emogonogo',
                                           '20110304_emogonogo',
                                           '20110405_emogonogo',
                                           '20110407_emogonogo',
                                           '20110503_emogonogo',
                                           '20110504_emogonogo',
                                           '20110506_emogonogo'])

    def getOutputByIndex(self,index):
        '''
        Note: index counts from 0
        '''
        return self.getOutput(self.subjectTagList[index])

    def getOutput(self,subjectTag):
        if subjectTag not in self.subjectTagList:
            return None
        foreignIndex = self.subjectTagList.index(subjectTag) + 1 # NB: subject foreignIndex counts from 1
        return SubjectManML(self.cmMetaParam,subjectTag,foreignIndex,self.cmBasePath,self.resDataSourceMan)

    def getNumSubjects(self):
        return len(self.subjectTagList)

# ------------------------------
class ResDataSourceMan(CompMan):
    '''
    Custom-coded interface for Matlab-generated residuals 4D NIFTI
    (.nii) files. For details, see
    MakeParameters_emogonogo_residuals_xxx.m files from 2010_emogonogo
    analysis Matlab code.

    cmMetaParam:
    'era20_20150414a' no slice-time correction
    'era20_20150415a' slice-time correction
    'era20_20150415b' slice-time correction, no percent signal scaling
    'era20_20150415c' slice-time correction, no percent signal scaling,
                      voxel means retained

    '''
    def __init__(self,cmMetaParam,cmBasePath):
        '''
        Note: There is some clunkyness getting this to mesh with Matlab stuff.
        '''
        cmDesc    = 'residuals'
        cmCodeTag = 'MakeResiduals_emogonogo' # Matlab code, not Python
        CompMan.__init__(self,cmDesc,cmCodeTag,cmMetaParam,cmBasePath=cmBasePath)
        self.configure()

    def configure_era20_20150414a(self):
        if self.cmMetaParam != 'era20_20150414a':
            raise InvalidMetaparameterError(self.cmMetaParam)
        self.setConfig('preprocTag'    , 'preproc_20131205')
        self.setConfig('percentScale'  , True)
        self.setConfig('keepVoxelMean' , False)
        self.setConfig('runList'       , ['run1','run2','run3','run4'])

    def configure_era20_20150415a(self):
        if self.cmMetaParam != 'era20_20150415a':
            raise InvalidMetaparameterError(self.cmMetaParam)
        self.setConfig('preprocTag'    , 'preproc_20131206')
        self.setConfig('percentScale'  , True)
        self.setConfig('keepVoxelMean' , False)
        self.setConfig('runList'       , ['run1','run2','run3','run4'])

    def configure_era20_20150415b(self):
        if self.cmMetaParam != 'era20_20150415b':
            raise InvalidMetaparameterError(self.cmMetaParam)
        self.setConfig('preprocTag'    , 'preproc_20131206')
        self.setConfig('percentScale'  , False)
        self.setConfig('keepVoxelMean' , False)
        self.setConfig('runList'       , ['run1','run2','run3','run4'])

    def configure_era20_20150415c(self):
        if self.cmMetaParam != 'era20_20150415c':
            raise InvalidMetaparameterError(self.cmMetaParam)
        self.setConfig('preprocTag'    , 'preproc_20131206')
        self.setConfig('percentScale'  , False)
        self.setConfig('keepVoxelMean' , True)
        self.setConfig('runList'       , ['run1','run2','run3','run4'])

    def getHashTag(self,includeExtraConfigDummy):
        '''
        Overrides CompMan's getHashTag().
        Returns custom hashtag matching stuff coming from Matlab.
        '''
        if self.cmMetaParam == 'era20_20150414a':
            hashtag = '62842927'
        elif self.cmMetaParam == 'era20_20150415a':
            hashtag = '679233458'
        elif self.cmMetaParam == 'era20_20150415b':
            hashtag = '360470451'
        elif self.cmMetaParam == 'era20_20150415c':
            hashtag = '3261238954'
        else:
            raise InvalidMetaparameterError(compound_metaparameter)
        return hashtag

    def getOutputFilesList(self,subjectTag,subjectIndex):
        '''
        Returns list of residuals .nii files for subject denoted by subjectIndex and subjectTag.
        subject_index counts from 1.
        '''
        filesList    = []
        prefix       = self.getTagPrefix(True)
        tsep         = self.cmSep
        outputPath   = self.getOutputPath()
        for run in self.runList:
            fileName = prefix + tsep + 'subj{0:02}_{1}_{2}.nii'.format(subjectIndex,run,subjectTag) # NB: subjectIndex counts from 1
            filePath = os.path.join(outputPath,fileName)
            filesList += [filePath]
        for f in filesList:
            if not os.path.isfile(f):
                print('One or more output files are not present. You must manually copy them over.')
        return filesList

    def getNumRuns(self):
        return len(self.runList)

# ------------------------------
class MaskMan(CompMan):
    '''
    Mask volume (binary, 0's and 1's) manager.
    '''
    def __init__(self,cmMetaParam,cmBasePath):
        cmDesc    = 'mask'
        cmCodeTag = os.path.splitext(os.path.split(inspect.getfile(inspect.currentframe()))[1])[0] + '_' + type(self).__name__
        CompMan.__init__(self,cmDesc,cmCodeTag,cmMetaParam,cmBasePath=cmBasePath)
        self.configure()

    def configure_era20_tae(self):
        if self.cmMetaParam != 'era20_tae':
            raise InvalidMetaParameterError(self.cmMetaParam)
        self.setConfig('longDescription' , 'Custom mask from thresholding average fMRI EPI volume')
        self.setConfig('maskFileName'    , 'mask_era20_thresholded_average_epi.nii')
        self.customOutputPath = os.path.join(self.cmBasePath,'mask/mask_era20_tae')
        self.maskFilePath     = os.path.join(self.getOutputPath(),self.maskFileName)

    def configure_fsl_mask1(self):
        if self.cmMetaParam != 'fsl_mask1':
            raise InvalidMetaParameterError(self.cmMetaParam)
        self.setConfig('longDescription' , 'FSL MNI T1 mask')
        self.setConfig('maskFileName'    , 'MNI152_T1_2mm_brain_mask.nii.gz')
        baseDataPath          = getBaseDataPath()
        self.customOutputPath = os.path.join(baseDataPath,'atlases/fsl-mni152-templates')
        self.maskFilePath     = os.path.join(self.getOutputPath(),self.maskFileName)

    def configure_fsl_custvent(self):
        if self.cmMetaParam != 'fsl_custvent':
            raise InvalidMetaParameterError(self.cmMetaParam)
        self.setConfig('longDescription' , 'MB custom ventricles extracted from FSL MNI T1 mask')
        self.setConfig('maskFileName'    , 'ventricles_from_FSL_MNI152_T1_2mm_brain_mask.nii')
        baseDataPath          = getBaseDataPath()
        self.customOutputPath = os.path.join(baseDataPath,'atlases/mb_custom')
        self.maskFilePath     = os.path.join(self.getOutputPath(),self.maskFileName)

    def configure_era20_combo1(self):
        if self.cmMetaParam != 'era20_combo1':
            raise InvalidMetaParameterError(self.cmMetaParam)
        self.setConfig('longDescription', ('Custom mask from thresholding average fMRI EPI volume '
                                           'with ventricles removed based on FSL MNI T1 Mask'))
        self.setConfig('maskFileName'   , 'mask_era20_combo1.nii')
        self.setConfig('maskMan1'       , MaskMan('era20_tae',self.cmBasePath))
        self.setConfig('maskMan2'       , MaskMan('fsl_custvent',None))
        self.customOutputPath = None
        self.maskFilePath     = os.path.join(self.getOutputPath(),self.maskFileName)
        if not os.path.isfile(self.maskFilePath):
            inFilePath1 = self.maskMan1.maskFilePath
            inFilePath2 = self.maskMan2.maskFilePath
            self.makeFile_era20_combo1(inFilePath1,inFilePath2)

    def makeFile_era20_combo1(self,inFilePath1,inFilePath2):
        nii1 = nib.load(inFilePath1)
        nii2 = nib.load(inFilePath2)
        smat1 = nii1.get_sform()
        smat2 = nii2.get_sform()
        mat = np.dot(np.linalg.inv(smat1),smat2)
        d = nii1.get_data()
        xa,ya,za = np.where(nii2.get_data()==1)
        for x,y,z in zip(xa,ya,za):
            icoord = np.round(np.dot(mat,[x,y,z,1.]))
            d[icoord[0],icoord[1],icoord[2]] = 0
        newnii = nib.nifti1.Nifti1Image(d,nii1.get_affine(),nii1.header)
        newnii.header['descrip'] = np.array('MB custom mask')
        self.makeOutputPath()
        nib.save(newnii,self.maskFilePath)
        print('Saved {}'.format(self.maskFilePath))
        self.saveConfigCSVFile()

    def getOutput(self):
        return nib.load(self.maskFilePath)

    def getOutputPath(self):
        if self.customOutputPath is None:
            return CompMan.getOutputPath(self)
        else:
            return self.customOutputPath

    def getOutputFile(self):
        return self.maskFilePath

# ------------------------------
class SeedRegionMan(CompMan):
    def __init__(self,localMetaparameter,maskMan,cmBasePath,index):
        cmDesc      = 'seedregion'
        cmCodeTag   = os.path.splitext(os.path.split(inspect.getfile(inspect.currentframe()))[1])[0] + '_' + type(self).__name__
        cmMetaParam = self.generateCompoundMetaParameter(localMetaparameter,[maskMan])
        CompMan.__init__(self,cmDesc,cmCodeTag,cmMetaParam,cmBasePath=cmBasePath)
        self.setConfig('localMetaparameter' , localMetaparameter)
        self.setConfig('maskMan'            , maskMan)
        self.setExConfig('index'            , index)
        self.configure(localMetaparameter)

    def configure_woo2014_v001(self):
        if self.localMetaparameter != 'woo2014_v001':
            raise InvalidMetaParameterError(self.localMetaparameter)
        self.setConfig('longDescription', ('Seed region names and centre coordinates from Woo et al., 2014, ' 
                                           'NeuroImage 91: 412-419. Spherical seed regions with random radii '
                                           '6-15mm in 1mm increments'))
        self.setConfig('seedGeneratorFunction' , self.generateSteppedSphereRegionVolume)
        self.setConfig('minRadius'             , 6) # in mm
        self.setConfig('maxRadius'             , 15) # in mm
        (regionNameList,regionCentreList) = self.getRegionParamsWoo2014()
        self.regionNameList   = regionNameList
        self.regionCentreList = regionCentreList  

    def getIndex(self):
        return self.index

    def getRegionParamsWoo2014(self):
        '''
        Seed region names and centre coordinate from Woo et al. 2014 NeuroImage paper.
        '''
        regionNameList = ['Caudate (L)',
                          'Caudate (R)',
                          'Dorsal anterior cingulate cortex (ACC)',
                          'Hippocampus',
                          'Inferior parietal lobe',
                          'Fusiform gyrus',
                          'Inferior frontal gyrus (L)',
                          'Anterior insular cortex (L)',
                          'Dorsal parietal insular cortex (L)',
                          'Putamen (L)',
                          'Supramarginal gyrus (L)',
                          'Orbitofrontal cortex',
                          'Orbitofrontal cortex',
                          'Parahippocampal cortex',
                          'Periaqueductal gray',
                          'Posterior cingulate cortex',
                          'Inferior frontal gyrus (R)',
                          'Anterior insular cortex (R)',
                          'Dorsal parietal insular cortex (R)',
                          'Putamen (R)',
                          'Subgenual ACC',
                          'Rostral ACC',
                          'Rostral-dorsal ACC',
                          'SII (L)',
                          'SII (R)',
                          'Striatum (L)',
                          'Striatum (R)',
                          'Supplementary motor area',
                          'Thalamus' ]
        regionCentreList = [ (-10,16,6),
                             (10,16,8),
                             (-4,16,40),
                             (-20,-6,-22),
                             (-48,-42,44),
                             (-34,-40,-20),
                             (-44,36,-8),
                             (-36,20,-8),
                             (-42,-14,2),
                             (-22,10,-2),
                             (-56,-44,28),
                             (-10,44,-20),
                             (-28,44,-16),
                             (-24,-24,-20),
                             (-2,-30,-10),
                             (-4,-48,28),
                             (46,34,-8),
                             (44,12,-8),
                             (40,-14,8),
                             (24,10,-2),
                             (-2,32,-6),
                             (-4,38,10),
                             (-4,24,22),
                             (-56,-6,8),
                             (56,-4,8),
                             (-10,14,-10),
                             (10,12,-12),
                             (-4,10,58),
                             (4,-4,-8) ]
        return (regionNameList,regionCentreList)

    def getOutput(self):
        return nib.load(self.getOutputFile())

    def getOutputFile(self):
        outputPath           = self.getOutputPath()
        (prefix,sep,hashtag) = self.getTagPrefixHashSep(True)
        fileName             = '{0}{1}regionset_{2:04}{3}{4}.nii'.format(prefix,sep,self.index,sep,hashtag)
        return os.path.join(outputPath,fileName)

    def makeOutputFiles(self,forceRebuild=False):
        print('Creating region .nii file')
        self.makeOutputPath()
        filePath = self.getOutputFile()
        if not os.path.isfile(filePath) or forceRebuild:
            regionNii = self.seedGeneratorFunction()
            nib.save(regionNii,filePath)
            print('    Saved {}'.format(filePath))
        else:
            print(    'File already exists: {}'.format(filePath))
        self.saveConfigCSVFile(forceRebuild)

    def generateSteppedSphereRegionVolume(self):
        mask = self.maskMan.getOutput()
        (iarray,jarray,karray,xarray,yarray,zarray) = getIndexRealCoordinates(mask)
        regionData = np.zeros(mask.get_header().get_data_shape(),dtype=np.uint16)
        minRadius = self.minRadius 
        maxRadius = self.maxRadius 
        for (num,centre) in enumerate(self.regionCentreList,1):
            radius = np.round(np.random.rand() * (maxRadius - minRadius) + minRadius)
            inSphere = getDistances(xarray,yarray,zarray,centre) <= radius
            regionData[np.logical_and(np.logical_and(inSphere,mask.get_data()>0),regionData==0)] = num
        regionNii = nib.Nifti1Image(regionData,mask.get_affine(),mask.get_header())
        regionNii.get_header().set_data_dtype(np.uint16)
        regionNii.get_header()['descrip'] = 'Simulated regions'
        regionNii.get_header()['glmin'] = regionData.min()
        regionNii.get_header()['glmax'] = regionData.max()
        regionNii.set_qform(regionNii.get_sform())
        return regionNii

# ------------------------------
class PreprocSeedRegionMan(CompMan):
    def __init__(self,cmMetaParam,seedRegionMan,cmBasePath):
        cmDesc    = 'preprocseedregion'
        cmCodeTag = os.path.splitext(os.path.split(inspect.getfile(inspect.currentframe()))[1])[0] + '_' + type(self).__name__
        CompMan.__init__(self,cmDesc,cmCodeTag,cmMetaParam,cmBasePath=cmBasePath)
        self.setConfig('seedRegionMan' , seedRegionMan)
        self.setExConfig('index'       , self.seedRegionMan.getIndex())
        self.configure()

    def getIndex(self):
        return self.index

    def configure_preproc_20131205(self):
        if self.cmMetaParam != 'preproc_20131205':
            raise InvalidMetaParameterError(self.cmMetaParam)
        self.setConfig('longDescription'       , 'Preprocessing including 8mm FWHM Gaussian spatial smoothing')
        self.setConfig('preprocessingFunction' , self.preprocessRegionVolume_gaussianSmoothing)
        self.setConfig('gaussianFWHM'          , 8.) # in mm

    def configure_preproc_20131206(self):
        if self.cmMetaParam != 'preproc_20131206':
            raise InvalidMetaParameterError(self.cmMetaParam)
        self.setConfig('longDescription'       , 'Preprocessing including 8mm FWHM Gaussian spatial smoothing')
        self.setConfig('preprocessingFunction' , self.preprocess_gaussianSmoothing)
        self.setConfig('gaussianFWHM'          , 8.) # in mm

    def getOutputFile(self):
        outputPath           = self.getOutputPath()
        (prefix,sep,hashtag) = self.getTagPrefixHashSep(True)
        fileName             = '{0}{1}regionset_{2:04}{3}{4}.nii'.format(prefix,sep,self.getIndex(),sep,hashtag)
        return os.path.join(outputPath,fileName)

    def makeOutputFiles(self,forceRebuild=False):
        print('Preprocessing region .nii files')
        self.makeOutputPath()
        origFilePath = self.seedRegionMan.getOutputFile()
        preprocFilePath = self.getOutputFile()
        if not os.path.isfile(preprocFilePath) or forceRebuild:
            preprocNii = self.preprocessingFunction(origFilePath)
            nib.save(preprocNii,preprocFilePath)
            print('    Saved {}'.format(preprocFilePath))
        else:
            print('    File already exists: {}'.format(preprocFilePath))
        self.saveConfigCSVFile(forceRebuild)

    def preprocess_gaussianSmoothing(self,origFilePath):
        '''
        Gaussian smoothing.
        '''
        nii = nib.load(origFilePath)
        scal = 2. * np.sqrt(2.*np.log(2.))
        if type(self.gaussianFWHM) in (list,tuple):
            sigma = [(x / scal) / nii.get_header()['pixdim'][i] for (i,x) in enumerate(self.gaussianFWHM,1)]
        else:
            sigma = [(self.gaussianFWHM / scal) / nii.get_header()['pixdim'][i] for i in range(1,4)]
        preprocData = sc.ndimage.gaussian_filter(np.array(nii.get_data()>0,dtype=np.float32),sigma)
        mask = self.seedRegionMan.maskMan.getOutput()
        preprocData[mask.get_data() == 0] = 0.
        preprocNii = nib.Nifti1Image(preprocData,nii.get_affine(),nii.get_header())
        preprocNii.get_header().set_data_dtype(np.float32)
        preprocNii.get_header().set_intent(0)
        preprocNii.get_header()['descrip'] = 'Preproc region 8mm Gaussian'
        preprocNii.get_header()['glmin'] = preprocData.min()
        preprocNii.get_header()['glmax'] = preprocData.max()
        preprocNii.set_qform(preprocNii.get_sform())
        return preprocNii

# ------------------------------
# REDO below here
class SummaryStatsMan(CompMan):
    def __init__(self,cmMetaParam,subjectSeriesMan,maskMan,cmBasePath):
        cmDesc    = 'summarystats'
        cmCodeTag = os.path.splitext(os.path.split(inspect.getfile(inspect.currentframe()))[1])[0] + '_' + type(self).__name__
        CompMan.__init__(self,cmDesc,cmCodeTag,cmMetaParam,cmBasePath=cmBasePath)
        self.setConfig('longDescription'  , None)
        self.setConfig('subjectSeriesMan' , subjectSeriesMan)
        self.setConfig('maskMan'          , maskMan)
        self.configure()

    def configure_mnstd(self):
        if self.cmMetaParam != 'mnstd':
            raise InvalidMetaparameterError(self.cmMetaParam)
        self.setConfig('longDescription', ('Mean and mean of standard deviation over time for '
                                           'all voxels inside mask, for each .nii file'))
        self.setConfig('summaryStatsFunction', self.makeSummaryStatsDict_MeanStd)
        #self.makeOutputFiles(self,forceRebuild=False)
        self.stats = self.loadOutputFiles()

    def getOutputFile(self):
        outputPath = self.getOutputPath()
        prefix     = self.getTagPrefix(True)
        fileName   = '{0}.sav'.format(prefix)
        return os.path.join(outputPath,fileName)

    def makeOutputFiles(self,forceRebuild=False):
        print('Computing summary stats for .nii files')
        self.makeOutputPath()
        filePath = self.getOutputFile()
        if not os.path.isfile(filePath) or forceRebuild:
            stats = self.makeSummaryStatsDict()
            with open(filePath,'w') as f:
                pickle.dump(stats,f)
            print('    Saved {}'.format(filePath))
        else:
            print('    File already exists: {}'.format(filePath))
        self.saveConfigCSVFile(forceRebuild)
        self.stats = self.loadOutputFiles()

    def loadOutputFiles(self):
        filePath = self.getOutputFile()
        if os.path.isfile(filePath):
            with open(filePath,'r') as f:
                stats = pickle.load(f)
        else:
            stats = None
        return stats

    def makeSummaryStatsDict(self):
        return self.summaryStatsFunction()

    def makeSummaryStatsDict_MeanStd(self):
        mask      = self.maskMan.getOutput()
        maskVol   = mask.get_data() > 0
        filesList = []
        meanList  = []
        stdList   = []
        for subjectMan in self.subjectSeriesMan:
            for f in subjectMan.getOutputFilesList():
                print(f)
                filesList += [f]
                nii = nib.load(f)
                d = nii.get_data()
                d_masked = d[maskVol]
                mn = np.mean([float(d_masked[:,i].mean()) for i in range(d_masked.shape[1])]) # avoid precision error
                meanList += [mn]
                st = float(d_masked.std(axis=1).mean())
                stdList  += [st]
                del(nii)
                del(d)
        stats = {}
        stats['files'] = filesList
        stats['means'] = meanList
        stats['stds']  = stdList
        return stats

    def getStats(self,fileName):
        '''
        Returns dict with 'mean' and 'std' keys for file fileName.
        '''
        fileInd = [os.path.split(f)[1] for f in self.stats['files']].index(os.path.split(fileName)[1])
        return {'mean': self.stats['means'][fileInd], 'std': self.stats['stds'][fileInd]}

    def getScaledStats(self,fileName):
        '''
        Returns dict with scaled 'mean' and 'std' keys for file fileName.
        Scaled mean is always 100.
        '''
        fileInd = [os.path.split(f)[1] for f in self.stats['files']].index(os.path.split(fileName)[1])
        return {'mean': 100., 'std': 100.*self.stats['stds'][fileInd] / self.stats['means'][fileInd]}

    def getMeanStats(self):
        '''
        Returns dict with mean of means ('mean' key) and mean of std ('std') key.
        '''
        return {'mean': np.mean(self.stats['means']), 'std': np.mean(self.stats['stds'])}

    def getScaledMeanStats(self):
        '''
        Returns dict with scaled mean of means ('mean' key) and mean of std ('std') key.
        Scaled mean is always 100.
        '''
        mn = self.stats['means']
        sd = self.stats['stds']
        scaled_sd = [100 * s / m for (m,s) in zip(mn,sd)]
        return {'mean':100., 'std': np.mean(scaled_sd)}

    def getOutput(self,scale=False):
        if not scale:
            return self.getMeanStats()
        else:
            return self.getScaledMeanStats()

# ------------------------------
class DesignMan(CompMan):
    '''
    Design information for one or more runs.
    '''
    def __init__(self,cmMetaParam,cmBasePath,index):
        cmDesc    = 'design'
        cmCodeTag = os.path.splitext(os.path.split(inspect.getfile(inspect.currentframe()))[1])[0] + '_' + type(self).__name__
        CompMan.__init__(self,cmDesc,cmCodeTag,cmMetaParam,cmBasePath=cmBasePath)
        self.setExConfig('index', index)
        self.setConfig('makeDesignDictFunction' , None)
        self.setConfig('numRuns'                , None)
        self.setConfig('trialTypeList'          , None)
        self.setConfig('timeUnits'              , None)
        self.setConfig('volumeTime'             , None)
        self.configure()

    def getIndex(self):
        return self.index

    def getNumRuns(self):
        return self.numRuns

    def getTrialTypeList(self):
        return self.trialTypeList

    def getTimeUnits(self):
        return self.timeUnits

    def getVolumeTime(self):
        return self.volumeTime

    def configure_ds001(self):
        if self.cmMetaParam != 'ds001':
            raise InvalidMetaparameterError(self.cmMetaParam)
        self.setConfig('longDescription', ('Single trial type, rapid event-related design. Uniform random '
                                           'ITI 2-10 seconds. Assumes volume time = 2s. Populated trials '
                                           'starting in [10.,280.] second interval.'))
        self.setConfig('makeDesignDictFunction' , self.makeDesignDict_DesignScheme001)
        self.setConfig('numRuns'                , 1)
        self.setConfig('trialTypeList'          , ['trialtype1'])
        self.setConfig('timeUnits'              , 'seconds')
        self.setConfig('volumeTime'             , 2.)
        self.setConfig('minITI'                 , 2.)
        self.setConfig('maxITI'                 , 10.)
        self.setConfig('minStartTime'           , 10.) # seconds
        self.setConfig('maxEndTime'             , 280.) # seconds
        self.designDictList = self.loadDesignDictList()

    def getOutputFile(self,runIndex):
        if runIndex < 0 or runIndex >= self.numRuns:
            raise IndexError()
        outputPath           = self.getOutputPath()
        (prefix,sep,hashtag) = self.getTagPrefixHashSep(True)
        fileName             = '{0}{1}variant{2:03}{3}{4}{5}run_{6:02}.sav'.format(prefix,sep,self.getIndex(),sep,hashtag,sep,runIndex)
        return os.path.join(outputPath,fileName)

    def makeOutputFiles(self,forceRebuild=False):
        print('Creating design files')
        self.makeOutputPath()
        for runIndex in range(self.numRuns):
            filePath = self.getOutputFile(runIndex)
            if not os.path.isfile(filePath) or forceRebuild:
                designDict = self.makeDesignDictFunction()
                with open(filePath,'w') as f:
                    pickle.dump(designDict,f)
                print('    Saved {}'.format(filePath))
            else:
                print('    File already exists: {}'.format(filePath))
        self.saveConfigCSVFile(forceRebuild)
        self.designDictList = self.loadDesignDictList()

    def loadDesignDictList(self):
        designDictList = []
        for runIndex in range(self.numRuns):
            filePath = self.getOutputFile(runIndex)
            if os.path.isfile(filePath):
                with open(filePath,'r') as f:
                    designDict = pickle.load(f)
            else:
                designDict = None
            designDictList += [designDict]
        return designDictList

    def makeDesignDict_DesignScheme001(self):
        intervals   = np.arange(self.minITI,self.maxITI+1,self.volumeTime)
        time        = self.minStartTime - self.minITI
        trialStarts = [] # in seconds
        while time <= self.maxEndTime:
            time = time + intervals[np.random.randint(0,intervals.size)]
            if time > self.maxEndTime:
                break
            trialStarts += [time]
        designDict                  = OrderedDict() 
        designDict['timeUnits']     = self.timeUnits
        designDict['trialTypeList'] = self.trialTypeList
        designDict['trialtype1']    = trialStarts
        return designDict

    def getOutput(self,runIndex):
        return self.designDictList[runIndex]

# ------------------------------
class ContrastMan(CompMan):
    '''
    Statistical contrast manager

    getOutPut() returns contrastDictDict, a dictionary of dictionaries,
    each specifying one contrast by weighting each trial type.
    Keys of contrastDictDict are self.contrastNameList.
    Keys of each dict inside contrastDictDict include everything in
    self.trialTypeList as well as 'contrastType'.
    '''
    def __init__(self,cmMetaParam):
        cmDesc    = 'contrast'
        cmCodeTag = os.path.splitext(os.path.split(inspect.getfile(inspect.currentframe()))[1])[0] + '_' + type(self).__name__
        CompMan.__init__(self,cmDesc,cmCodeTag,cmMetaParam)
        self.setConfig('trialTypeList'                , None)
        self.setConfig('contrastNameList'             , None)
        self.setConfig('contrastTypeList'             , None)
        self.setConfig('makeContrastDictDictFunction' , None)
        self.configure()
        self.contrastDictDict = self.makeContrastDictDictFunction()

    def getTrialTypeList(self):
        return self.trialTypeList

    def getContrastNameList(self):
        return self.contrastNameList

    def getContrastTypeList(self):
        return self.contrastTypeList

    def configure_tt1loc(self):
        if self.cmMetaParam != 'tt1loc':
            raise InvalidMetaparameterError(self.cmMetaParam)
        self.setConfig('trialTypeList'                , ['trialtype1'])
        self.setConfig('contrastNameList'             , ['localizer'])
        self.setConfig('contrastTypeList'             , ['ttest'])
        self.setConfig('makeContrastDictDictFunction' , self.makeContrastDictDict_TrialType1Localizer)

    def getOutput(self):
        '''
        Returns contrastDictDict.
        '''
        return self.makeContrastDictDictFunction()

    def makeContrastDictDict_TrialType1Localizer(self):
        contrastDict1                  = OrderedDict()
        contrastDict1['contrastType']  = 'ttest'
        contrastDict1['trialTypeList'] = self.trialTypeList
        contrastDict1['trialtype1']    = 1.0
        contrastDictDict               = OrderedDict()
        contrastDictDict['localizer']  = contrastDict1
        return contrastDictDict

# ------------------------------
class DesignMatrixMan(CompMan):
    def __init__(self,cmMetaParam):
        cmDesc    = 'designmatrix'
        cmCodeTag = os.path.splitext(os.path.split(inspect.getfile(inspect.currentframe()))[1])[0] + '_' + type(self).__name__
        CompMan.__init__(self,cmDesc,cmCodeTag,cmMetaParam)
        self.setConfig('longDescription'      , None)
        self.setConfig('designMatrixFunction' , None)
        self.configure(self.cmMetaParam)

    class TrialStartsPastEndError(Exception):
        def __init__(self,*args,**kwArgs):
            Exception.__init__(self,*args,**kwArgs)

    def configure_d01n00(self):
        self.setConfig('longDescription', ('SPM canonical difference of gammas HRF, one predictor '
                                           'per trialtype. No run offset predictor. No nuisance predictors.'))
        self.setConfig('designMatrixFunction'   , self.makeDesignMatrix_v1)
        self.setConfig('hrfFunction'            , self.spmCanonicalHRF)
        self.setConfig('hrfSamplingPeriod'      , None)
        self.setConfig('contrastVectorFunction' , self.spmCanonicalHRFContrastVector)
        self.setConfig('includeRunOffset'       , False)
        self.setConfig('includeRunLinearDrift'  , False)
        self.setConfig('includeRunSinusoids'    , False)
        self.setConfig('includeRunMotionParams' , False)

    def configure_d01n01(self):
        self.setConfig('longDescription', ('SPM canonical difference of gammas HRF, one predictor per trialtype. '
                                           'Includes run offset predictor. No other nuisance predictors.'))
        self.setConfig('designMatrixFunction'   , self.makeDesignMatrix_v1)
        self.setConfig('hrfFunction'            , self.spmCanonicalHRF)
        self.setConfig('hrfSamplingPeriod'      , None)
        self.setConfig('contrastVectorFunction' , self.spmCanonicalHRFContrastVector)
        self.setConfig('includeRunOffset'       , True)
        self.setConfig('includeRunLinearDrift'  , False)
        self.setConfig('includeRunSinusoids'    , False)
        self.setConfig('includeRunMotionParams' , False)

    def spmCanonicalHRF(self,volumeTime):
        '''
        Returns (hrfCurve,timePoints)
        '''
        (hrfCurve,params,timePoints) = spmhrf.spm_hrf(volumeTime)
        hrfCurve = np.reshape(hrfCurve,(hrfCurve.size,1))
        return (hrfCurve,timePoints)

    def getOutput(self,designMan,numVolumesList,deleteStartVolumes,allowTrialStartsPastEnd=False):
        return self.designMatrixFunction(designMan,numVolumesList,deleteStartVolumes,allowTrialStartsPastEnd)

    def makeDesignMatrix_v1(self,designMan,numVolumesList,deleteStartVolumes,allowTrialStartsPastEnd=False):
        '''
        Returns (designMat,indexDict)
        '''
        (designMatrix,indexDict) = self.makeDesignMatrix_taskPredictors(designMan,
                                                                        numVolumesList,
                                                                        deleteStartVolumes,
                                                                        self.hrfFunction,
                                                                        self.hrfSamplingPeriod,
                                                                        allowTrialStartsPastEnd)
        # Add offsets and nuisance predictors:
        funcList = []
        if self.includeRunOffset:
            funcList += [self.makeDesignMatrix_runOffsets]
        if self.includeRunLinearDrift:
            funcList += [self.makeDesignMatrix_runLinearDrifts]
        if self.includeRunSinusoids:
            funcList += [self.makeDesignMatrix_runSinusoids]
        if self.includeRunMotionParams:
            funcList += [self.makeDesignMatrix_runMotionParams]
        for func in funcList:
            (newMatrix,newIndexDict) = func(numVolumesList,deleteStartVolumes,self.getMaxIndex(indexDict)+1)
            designMatrix = np.concatenate((designMatrix,newMatrix),axis=1)
            indexDict    = self.combineIndexDict(indexDict,newIndexDict)

        return (designMatrix,indexDict)

    def getMaxIndex(self,indexDict):
        maxIndex = 0;
        for indexList in indexDict.itervalues():
            mx = max(indexList)
            maxIndex = max(maxIndex,mx)
        return maxIndex

    def combineIndexDict(self,indexDict1,indexDict2):
        newIndexDict = indexDict1.copy()
        for (key,value) in indexDict2.iteritems():
            if key in newIndexDict:
                raise Exception('Two indexDicts must have disjoint keys')
            newIndexDict[key] = value
        return newIndexDict

    def makeDesignMatrix_taskPredictors(self,designMan,numVolumesList,deleteStartVolumes,hrfFunction,hrfSamplingPeriod,allowTrialStartsPastEnd):
        '''
        Returns (designMat,indexDict)
        designMat:
            numVolumes X (num_trial_types * num_hrf_curves)
        No constant offset or nuisance predictors.
        indexDict, keyed by trial types, list of indices into
            designMat's columns

        designMan: DesignMan CompMan object
        numVolumesList: list of run durations in volumes (not seconds)
        deleteStartVolumes: number of volumes to delete at start of
                            each run for spin saturation
        volumeTime and hrfSamplingPeriod are in seconds.
        hrfSamplingPeriod defaults to volumeTime, can be set smaller
            (eg: volumeTime / 10.) if trials not synchronized with
            volume collection times. Best to set hrfSamplingPeriod
            to an integral quotient of volumeTime.
        allowTrialStartsPastEnd: If true, does not raise exception
            when trial(s) starts past end of modeled time course
            (volumeTime * numVolumes seconds), in which case trial
            is silently omitted.
        Does not check whether trial HRF predictor(s) are non-zero
        past end of modeled time course (volumeTime * numVolumes
        seconds). Such points are silently omitted.
        '''
        designMat = None
        indexDict = None
        if designMan.getNumRuns() != len(numVolumesList):
            raise Exception('Arguments mismatch for number of runs.')
        if designMan.timeUnits != 'seconds':
            raise ValueError('Code requires time units of seconds')
        volumeTime = designMan.getVolumeTime()
        if hrfSamplingPeriod is None:
            hrfSamplingPeriod = volumeTime
        trialTypeList = designMan.getTrialTypeList()
        for runIndex in range(designMan.getNumRuns()):
            if runIndex >= 2:
                raise Exception('Code not tested with num runs >= 2. Remove this exception, proceed carefully, and do quality checking now.')
            designDict      = designMan.getOutput(runIndex)
            numVolumes      = numVolumesList[runIndex]
            lastVolumeOnset = volumeTime * (numVolumes-1)
            if not allowTrialStartsPastEnd:
                for trialType in trialTypeList:
                    if np.any(np.array(designDict[trialType]) > lastVolumeOnset):
                        raise self.TrialStartsPastEndError()

            (hrfMat,hrfTimePoints) = hrfFunction(hrfSamplingPeriod)
            nHRFCurves             = hrfMat.shape[1]
            superSamplingTimes     = np.arange(0,numVolumes*volumeTime+1,hrfSamplingPeriod)
            superSamplingTimes     = np.reshape(superSamplingTimes,(superSamplingTimes.size,1))
            superSampledDesignMat  = np.zeros((superSamplingTimes.size,nHRFCurves*len(trialTypeList)))
            pos = 0
            tempIndexDict = {}
            for trialType in trialTypeList:
                indexList  = []
                onsets     = designDict[trialType]
                onsets     = np.reshape(onsets,(1,len(onsets)))
                indicator  = np.zeros(superSamplingTimes.size,dtype=np.bool)
                ind        = np.argmin(np.abs(superSamplingTimes-onsets),axis=0)
                if allowTrialStartsPastEnd:
                    ind = ind[ind<indicator.size]
                indicator[ind] = 1
                for i in range(hrfMat.shape[1]):
                    predictor = np.convolve(indicator,hrfMat[:,i],mode='full')
                    predictor = predictor[:superSamplingTimes.size]
                    superSampledDesignMat[:,pos] = predictor
                    indexList += [pos]
                    pos += 1
                tempIndexDict[trialType] = indexList
            volumeOnsets  = np.arange(0,numVolumes*volumeTime,volumeTime)
            volumeOnsets  = np.reshape(volumeOnsets,(1,volumeOnsets.size))
            ind           = np.argmin(np.abs(superSamplingTimes-volumeOnsets),axis = 0)
            tempDesignMat = superSampledDesignMat[ind,:]
            tempDesignMat = tempDesignMat[deleteStartVolumes:,:]
            if designMat is None:
                designMat = tempDesignMat
            else:
                designMat = np.concatenate((designMat,tempDesignMat),axis=0)
            if indexDict is None:
                indexDict = tempIndexDict
            else:
                if indexDict != tempIndexDict:
                    raise Exception('Mismatch between indexDict from different runs.')
        return (designMat,indexDict)

    def makeDesignMatrix_runOffsets(self,numVolumesList,deleteStartVolumes,indexOffset):
        numRuns   = len(numVolumesList)
        matrix    = np.zeros((sum(numVolumesList)-numRuns*deleteStartVolumes,numRuns))
        indexDict = {}
        timePos   = 0
        for runIndex in range(numRuns):
            numVols = numVolumesList[runIndex] - deleteStartVolumes
            matrix[timePos:timePos+numVols,runIndex] = 1
            indexDict['offsetRun{0:02}'.format(runIndex)] = [runIndex + indexOffset]
            timePos += numVols
        return (matrix,indexDict)

    def makeDesignMatrix_runLinearDrifts(self,numVolumesList,deleteStartVolumes,indexOffset):
        raise NotImplementedError('Code this!')

    def makeDesignMatrix_runSinusoids(self,numVolumesList,deleteStartVolumes,indexOffset):
        raise NotImplementedError('Code this!')

    def makeDesignMatrix_runMotionParams(self,numVolumesList,deleteStartVolumes,indexOffset):
        raise NotImplementedError('Code this!')

    def contrastVector(self,designMatrix,indexDict,contrastDict):
        return self.contrastVectorFunction(designMatrix,indexDict,contrastDict)

    def spmCanonicalHRFContrastVector(self,designMatrix,indexDict,contrastDict):
        conVec = np.zeros(designMatrix.shape[1],np.float)
        for trialType in contrastDict['trialTypeList']:
            weight = contrastDict[trialType]
            ind    = indexDict[trialType]
            if np.any(conVec[ind] != 0.):
                raise ValueError('contrastDict contains duplicate indices')
            conVec[ind] = float(weight) / len(ind)
        return conVec

    def mapTrialNamesToIndices(self,designDict,hrfFunction):
        '''
        Returns indexDict
        indexDict, keyed by trial types, list of indices into
            design matrix's columns
        '''
        (hrfMat,hrfTimePoints) = hrfFunction(1.0)
        nHRFCurves = hrfMat.shape[1]
        pos = 0
        indexDict = {}
        for trialType in trialTypeList:
            indexDict[trialType] = range(pos,pos+nHRFCurves)
            pos += nHRFCurves
        return indexDict

# ------------------------------
class TrialWeightsMan(CompMan):
    '''
    Weights to different trial types for creating simulated signals. Trial type
    names must match those in DesignSeriesMan.
    '''
    def __init__(self,cmMetaParam):
        cmDesc    = 'trialweights'
        cmCodeTag = os.path.splitext(os.path.split(inspect.getfile(inspect.currentframe()))[1])[0] + '_' + type(self).__name__
        CompMan.__init__(self,cmDesc,cmCodeTag,cmMetaParam)
        self.configure()

    def configure_tw001_0p0(self):
        if self.cmMetaParam != 'tw001_0p0':
            raise InvalidMetaparameterError(self.cmMetaParam)
        weight = 0.0
        self.configure_tw001_helper(weight)

    def configure_tw001_0p01(self):
        if self.cmMetaParam != 'tw001_0p01':
            raise InvalidMetaparameterError(self.cmMetaParam)
        weight = 0.01
        self.configure_tw001_helper(weight)

    def configure_tw001_0p05(self):
        if self.cmMetaParam != 'tw001_0p05':
            raise InvalidMetaparameterError(self.cmMetaParam)
        weight = 0.05
        self.configure_tw001_helper(weight)

    def configure_tw001_0p1(self):
        if self.cmMetaParam != 'tw001_0p1':
            raise InvalidMetaparameterError(self.cmMetaParam)
        weight = 0.1
        self.configure_tw001_helper(weight)

    def configure_tw001_0p3(self):
        if self.cmMetaParam != 'tw001_0p3':
            raise InvalidMetaparameterError(self.cmMetaParam)
        weight = 0.3
        self.configure_tw001_helper(weight)

    def configure_tw001_0p4(self):
        if self.cmMetaParam != 'tw001_0p4':
            raise InvalidMetaparameterError(self.cmMetaParam)
        weight = 0.4
        self.configure_tw001_helper(weight)

    def configure_tw001_0p6(self):
        if self.cmMetaParam != 'tw001_0p6':
            raise InvalidMetaparameterError(self.cmMetaParam)
        weight = 0.6
        self.configure_tw001_helper(weight)

    def configure_tw001_0p8(self):
        if self.cmMetaParam != 'tw001_0p8':
            raise InvalidMetaparameterError(self.cmMetaParam)
        weight = 0.8
        self.configure_tw001_helper(weight)

    def configure_tw001_0p9(self):
        if self.cmMetaParam != 'tw001_0p9':
            raise InvalidMetaparameterError(self.cmMetaParam)
        weight = 0.9
        self.configure_tw001_helper(weight)

    def configure_tw001_1p2(self):
        if self.cmMetaParam != 'tw001_1p2':
            raise InvalidMetaparameterError(self.cmMetaParam)
        weight = 1.2
        self.configure_tw001_helper(weight)

    def configure_tw001_1p5(self):
        if self.cmMetaParam != 'tw001_1p5':
            raise InvalidMetaparameterError(self.cmMetaParam)
        weight = 1.5
        self.configure_tw001_helper(weight)

    def configure_tw001_helper(self,weight):
        self.setConfig('longDescription', 'Single trial type')
        d                     = {}
        d['trialTypeList']    = ['trialtype1']
        d['trialtype1']       = [weight]
        self.setConfig('trialWeightsDict', d)

    def getTrialWeightsDict(self):
        return self.trialWeightsDict

# ------------------------------
class SimSubjectDataMan(CompMan):
    def __init__(self,cmMetaParam,subjectMan,runIndex,summaryStatsMan,preprocSeedRegionMan,designMan,trialWeightsMan,cmBasePath):
        cmDesc    = 'simsubjectdata'
        cmCodeTag = os.path.splitext(os.path.split(inspect.getfile(inspect.currentframe()))[1])[0] + '_' + type(self).__name__
        CompMan.__init__(self,cmDesc,cmCodeTag,cmMetaParam,cmBasePath=cmBasePath)
        self.setConfig('longDescription'      , None)
        self.setConfig('subjectMan'           , subjectMan)
        self.setConfig('runIndex'             , runIndex)
        self.setConfig('summaryStatsMan'      , summaryStatsMan)
        self.setConfig('preprocSeedRegionMan' , preprocSeedRegionMan)
        self.setConfig('designMan'            , designMan)
        self.setConfig('trialWeightsMan'      , trialWeightsMan)
        self.setConfig('designMatrixMan'      , None)
        self.configure(self.cmMetaParam)

    def configure_sd001(self):
        if self.cmMetaParam != 'sd001':
            raise InvalidMetaparameterError(self.cmMetaParam)
        self.setConfig('longDescription', 'Adds simulated task activation signal to saved residuals.')
        self.setConfig('simDataFunction', self.makeSimulatedData_sd001)
        self.setConfig('designMatrixMan', DesignMatrixMan('d01n00'))

    def getOutput(self):
        '''
        Returns an Nibabel NIFTI object.
        '''
        return self.simDataFunction()

    def getOutputFile(self):
        outputPath = self.getOutputPath()
        prefix     = self.getTagPrefix(True)
        fileName   = '{0}.nii'.format(prefix)
        return os.path.join(outputPath,fileName)

    def makeOutputFiles(self,forceRebuild=False):
        print('Creating simulated data .nii file')
        self.makeOutputPath()
        filePath = self.getOutputFile()
        if not os.path.isfile(filePath) or forceRebuild:
            simNii = self.getOutput()
            nib.save(simNii,filePath)
            print('    Saved {}'.format(filePath))
        else:
            print(    'File already exists: {}'.format(filePath))
        self.saveConfigCSVFile(forceRebuild)

    def makeSimulatedData_sd001(self):
        resFilePath = self.subjectMan.getOutputFilesList()[self.runIndex]
        nii         = nib.load(resFilePath)
        resData     = nii.get_data()
        resData     = resData * 100 / self.summaryStatsMan.getStats(resFilePath)['mean'] # scale by within-mask mean intensity
        scaledStd   = self.summaryStatsMan.getScaledMeanStats()['std'] # NB: same for all subjects

        numVolumesList           = [nii.header['dim'][4]]
        deleteStartVolumes       = 0
        allowTrialStartsPastEnd  = False
        (designMatrix,indexDict) = self.designMatrixMan.getOutput(self.designMan,numVolumesList,deleteStartVolumes,allowTrialStartsPastEnd)

        weightsDict = self.trialWeightsMan.getTrialWeightsDict()
        # Assumes weightsDict is coherent with hrfFunction

        designDict = self.designMan.getOutput(0)
        signal     = np.zeros(designMatrix.shape[0])
        pos        = 0
        for trialType in designDict['trialTypeList']:
            for w in weightsDict[trialType]:
                signal += designMatrix[:,pos] * w * scaledStd
                pos += 1

        preprocSeedFilePath = self.preprocSeedRegionMan.getOutputFile()
        preprocSeedNii      = nib.load(preprocSeedFilePath)
        if not np.all(preprocSeedNii.affine == nii.affine):
            raise Exception('preproc seed file and residuals file have different affine transforms')
        seedData = preprocSeedNii.get_data()
        simData  = resData + np.repeat(np.reshape(seedData,seedData.shape+(1,)),signal.size,axis=3) * np.reshape(signal,(1,1,1,signal.size))
        #for testing: simData = resData + (np.repeat(np.reshape(seedData,seedData.shape+(1,)),signal.size,axis=3) > 0) * np.reshape(signal,(1,1,1,signal.size)) * 1000

        simNii                         = nib.Nifti1Image(simData,nii.get_affine(),nii.get_header())
        simNii.get_header()['descrip'] = 'Simulated data'
        simNii.get_header()['glmin']   = simData.min()
        simNii.get_header()['glmax']   = simData.max()
        simNii.set_qform(simNii.get_sform())
        return simNii

# ------------------------------
class SubjectAnalysisMan(CompMan):
    def __init__(self,cmMetaParam,subjectDataMan,designMan,contrastMan,maskMan,cmBasePath):
        cmDesc    = 'simsubjectanalysis'
        cmCodeTag = os.path.splitext(os.path.split(inspect.getfile(inspect.currentframe()))[1])[0] + '_' + type(self).__name__
        CompMan.__init__(self,cmDesc,cmCodeTag,cmMetaParam,cmBasePath=cmBasePath)
        self.setConfig('longDescription' , None)
        self.setConfig('subjectDataMan'  , subjectDataMan)
        self.setConfig('designMan'       , designMan)
        self.setConfig('designMatrixMan' , None)
        self.setConfig('contrastMan'     , contrastMan)
        self.setConfig('maskMan'         , maskMan)
        self.setConfig('analysisFunction', None)
        self.configure()

    def configure_sa001(self):
        if self.cmMetaParam != 'sa001':
            raise InvalidMetaparameterError(self.cmMetaParam)
        self.setConfig('longDescription'               , 'Analysis of data from one subject.')
        self.setConfig('analysisFunction'              , self.doAnalysis_Procedure001)
        self.setConfig('deleteStartVolumes'            , 0)
        self.setConfig('percentSignalChangeScale'      , True)
        self.setConfig('designMatrixMan'               , DesignMatrixMan('d01n01'))
        self.setConfig('allowTrialStartsPastEnd'       , False)
        self.setConfig('autoCorrelationCorrection'     , True)
        self.setConfig('autoCorrelationCorrectionMode' , 'allinmask')
        self.setConfig('autoCorrelationMaxLag'         , 5)

    def getOutput(self):
        return self.analysisFunction()

    def getOutputFilesDicts(self):
        '''
        Returns (glmFilesDict,contrastFilesDict,spmFilesDict)
        '''
        outputPath   = self.getOutputPath()
        prefix       = self.getTagPrefix(True)
        glmFilesDict = OrderedDict()
        for name in ('beta','variance'):
            glmFilesDict[name] = os.path.join(outputPath,'{0}{1}{2}.nii'.format(prefix,self.cmSep,name))
        contrastFilesDict = OrderedDict()
        spmFilesDict      = OrderedDict()
        contrastDictDict  = self.contrastMan.getOutput()
        for name in contrastDictDict.iterkeys():
            contrastFilesDict[name] = os.path.join(outputPath,'{0}{1}contrast_{2}.nii'.format(prefix,self.cmSep,name))
            spmFilesDict[name]      = os.path.join(outputPath,'{0}{1}spm_{2}.nii'.format(prefix,self.cmSep,name))
        return (glmFilesDict,contrastFilesDict,spmFilesDict)

    def makeOutputFiles(self,forceRebuild=False):
        print('Analysing subject')
        self.makeOutputPath()
        filesDictTuple = self.getOutputFilesDicts()
        if forceRebuild or not self.checkFilesExist(filesDictTuple):
            niiDictTuple = self.getOutput()
            for (filesDict,niiDict) in zip(filesDictTuple,niiDictTuple):
                for key in filesDict.iterkeys():
                    nib.save(niiDict[key],filesDict[key])
                    print('Saved file: ',filesDict[key])
        else:
            print('Files already exist')
        self.saveConfigCSVFile(forceRebuild)

    def checkFilesExist(self,filesDictTuple):
        '''
        Returns False if any file in filesDictTuple does NOT exist,
        otherwise returns True.
        filesDictTuple is a tuple of dictionaries whose values are
        file paths.
        '''
        for filesDict in filesDictTuple:
            for filePath in filesDict.itervalues():
                if not os.path.isfile(filePath):
                    return False
        return True

    def doAnalysis_Procedure001(self):
        '''
        Returns (glmNiiDict,contrastNiiDict,spmNiiDict).
        '''
        # Data
        print('Loading data')
        dataNii      = self.subjectDataMan.getOutput()
        data         = dataNii.get_data()
        flatData     = np.reshape(data,(np.prod(data.shape[:3]),data.shape[3])).transpose()
        flatMaskData = self.getMaskVariables(dataNii)
        flatData     = self.applyMask(flatData,flatMaskData)
        if self.percentSignalChangeScale:
            print('Percent signal change scaling data')
            flatData = self.percentSignalChangeScaleData(flatData)

        # GLM
        print('Fitting GLM')
        (designMatrix,indexDict)   = self.getDesignMatrix(dataNii)
        (flatBeta,flatVar,flatRes) = self.fitGLM(flatData,designMatrix)
        if self.autoCorrelationCorrection:
            print('Whitening for autocorrelation correction and re-fitting GLM')
            autoCorrCoef = self.computeAutoCorrelations(flatRes)
            (flatBeta,flatVar,flatRes,whiteDesignMatrix) = self.fitGLMWithWhitening(flatData,designMatrix,autoCorrCoef)
            designMatrix = whiteDesignMatrix
        beta    = self.unflatten(self.reverseMask(flatBeta,flatMaskData),dataNii)
        var     = self.unflatten(self.reverseMask(flatVar,flatMaskData),dataNii)
        betaNii = self.makeNiiOutput(dataNii,beta,'beta')
        varNii  = self.makeNiiOutput(dataNii,var,'variance')
        glmNiiDict             = OrderedDict()
        glmNiiDict['beta']     = betaNii
        glmNiiDict['variance'] = varNii

        # Contrasts and SPMs
        print('Computing statistics')
        contrastDictDict = self.contrastMan.getOutput()
        contrastNiiDict  = OrderedDict()
        spmNiiDict       = OrderedDict()
        for (contrastName,contrastDict) in contrastDictDict.iteritems():
            (flatContrast,flatSpm) = self.computeContrast(flatBeta,flatVar,designMatrix,indexDict,contrastName)
            contrast     = self.unflatten(self.reverseMask(flatContrast,flatMaskData),dataNii)
            spm          = self.unflatten(self.reverseMask(flatSpm,flatMaskData),dataNii)
            contrastNii  = self.makeNiiOutput(dataNii,contrast,'contrast')
            spmNii       = self.makeNiiOutput(dataNii,spm,contrastDict['contrastType'])
            contrastNiiDict[contrastName] = contrastNii
            spmNiiDict[contrastName]      = spmNii

        return (glmNiiDict,contrastNiiDict,spmNiiDict)

    def getMaskVariables(self,dataNii):
        '''
        Returns (maskNii,maskData,flatMaskData)
        '''
        if self.maskMan is None:
            flatMaskData = None
        else:
            maskNii      = self.maskMan.getOutput()
            if (np.any(maskNii.get_header()['dim'][1:4] != dataNii.get_header()['dim'][1:4]) or
                np.any(maskNii.get_header()['pixdim'][1:4] != dataNii.get_header()['pixdim'][1:4])):
                raise Exception('dataNii and maskNii have mismatching dimensions.')
            maskData     = maskNii.get_data()
            flatMaskData = maskData.ravel()
        return flatMaskData

    def applyMask(self,flatArray,flatMaskData):
        '''
        flatArray: #whatever x #voxels, #whatever can be 1
        Return masked flatArray
        '''
        if flatMaskData is None:
            return flatArray
        if flatArray.ndim == 1:
            return flatArray[flatMaskData>0]
        if flatArray.ndim == 2:
            return flatArray[:,flatMaskData>0]
        raise Exception('flatArray must have ndim 1 or 2')

    def reverseMask(self,flatArray,flatMaskData):
        '''
        Returns unmaskedFlatArray
        '''
        if flatMaskData is None:
            unmaskedFlatArray = flatArray
        else:
            numVoxels = flatMaskData.size
            if flatArray.ndim == 1:
                unmaskedFlatArray = np.zeros(numVoxels,np.float)
                unmaskedFlatArray[flatMaskData>0] = flatArray
            elif flatArray.ndim == 2:
                unmaskedFlatArray = np.zeros((flatArray.shape[0],numVoxels),np.float)
                unmaskedFlatArray[:,flatMaskData>0] = flatArray
            else:
                raise Exception('flatArray must have ndim 1 or 2')
        return unmaskedFlatArray

    def unflatten(self,flatArray,dataNii):
        '''
        Returns array
        '''
        shape = dataNii.get_data().shape[:3]
        if flatArray.ndim == 1:
            array = np.reshape(flatArray,shape)
        elif flatArray.ndim == 2:
            array = np.reshape(flatArray.transpose(),shape+(flatArray.shape[0],))
        else:
            raise Exception('flatArray must have ndim 1 or 2')
        return array

    def percentSignalChangeScaleData(self,flatData):
        '''
        flatData must be #timepoints X #voxels (i.e. each column a timecourse)
        '''
        meanData   = np.mean(flatData,axis=0)
        scaledData = 100 * (flatData - meanData) / meanData
        return scaledData

    def getDesignMatrix(self,dataNii):
        '''
        Return (designMatrix,indexDict).
        '''
        numVolumesList = [dataNii.header['dim'][4]]
        return self.designMatrixMan.getOutput(self.designMan,numVolumesList,self.deleteStartVolumes,self.allowTrialStartsPastEnd)

    def fitGLM(self,flatData,designMatrix):
        '''
        GLM fitting.
        Returns (flatBeta,flatVar)
        '''
        hatMatrix = np.dot(np.linalg.inv(np.dot(designMatrix.transpose(),designMatrix)),designMatrix.transpose())
        flatBeta  = np.dot(hatMatrix,flatData)
        flatEst   = np.dot(designMatrix,flatBeta)
        flatRes   = flatData - flatEst
        df        = designMatrix.shape[0] - designMatrix.shape[1]
        flatVar   = np.sum(flatRes**2,axis=0) / df
        return (flatBeta,flatVar,flatRes)

    def fitGLMWithWhitening(self,flatData,designMatrix,autoCorrCoef):
        '''
        GLM fitting with whitening of data and design matrix to
        correct for autocorrelations.

        Returns (flatBeta,flatVar)
        '''
        whitMat = self.computeWhiteningMatrix(designMatrix.shape[0],autoCorrCoef)
        flatWhiteData = np.dot(whitMat,flatData)
        whiteDesignMatrix = np.dot(whitMat,designMatrix)
        return self.fitGLM(flatWhiteData,whiteDesignMatrix) + (whiteDesignMatrix,)

    def computeContrast(self,flatBeta,flatVar,designMatrix,indexDict,contrastName):
        '''
        Returns (flatContrast,flatSpm).
        '''
        contrastType = self.contrastMan.getOutput()[contrastName]['contrastType']
        if contrastType == 'ttest':
            return self.computeContrast_tTest(flatBeta,flatVar,designMatrix,indexDict,contrastName)
        else:
            raise NotImplementedError('Not coded for contrastType {0}'.format(contrastType))

    def computeContrast_tTest(self,flatBeta,flatVar,designMatrix,indexDict,contrastName):
        contrastDict = self.contrastMan.getOutput()[contrastName]
        conVec       = self.designMatrixMan.contrastVector(designMatrix,indexDict,contrastDict)
        flatContrast = np.dot(conVec,flatBeta)
        mixMat       = np.linalg.inv(np.dot(designMatrix.transpose(),designMatrix))
        quadForm     = np.dot(conVec,np.dot(mixMat,conVec))
        denominator  = np.sqrt(quadForm) * np.sqrt(flatVar)
        flatSpm      = flatContrast / denominator # t values
        return (flatContrast,flatSpm)

    def computeAutoCorrelations(self,flatRes):
        '''
        flatRes: #timepoints x #voxels residuals matrix
        Returns autoCorrCoef (coefficients for lag 0, 1, 2, 3, ...)
        '''
        if self.autoCorrelationCorrectionMode == 'allinmask':
            index = np.arange(flatRes.shape[1])
        else:
            raise NotImplementedError('Not coded for autoCorrelationCorrectionMode {0}'.format(self.autoCorrelationCorrectionMode))
        autoCorrCoef    = np.zeros(self.autoCorrelationMaxLag+1,np.float)
        autoCorrCoef[0] = 1. # correlation = 1.0 for lag 0, obviously
        sys.stdout.write('Computing autocorrelations: ')
        sys.stdout.flush()
        for offset in range(1,self.autoCorrelationMaxLag+1):
            sys.stdout.write('{0} '.format(offset))
            sys.stdout.flush()
            r1 = flatRes[:-offset,index].transpose().ravel()
            r2 = flatRes[offset:,index].transpose().ravel()
            c  = np.corrcoef(r1,r2)
            autoCorrCoef[offset] = c[0,1]
        print('')
        return autoCorrCoef

    def computeWhiteningMatrix(self,numTimePoints,autoCorrCoef):
        '''
        Computes whitening matrix based on Worseley et al. 2002,
        NeuroImage 15:1-5, Appendix A.3.

        Returns whiteMat.
        '''
        Ainv     = np.linalg.cholesky(sc.linalg.toeplitz(autoCorrCoef))
        p1       = Ainv.shape[1]
        A        = np.linalg.inv(Ainv)
        col      = np.zeros(numTimePoints)
        col[:p1] = A[-1,::-1]
        row      = np.zeros(numTimePoints);
        row[0]   = col[0]
        Vmhalf   = sc.linalg.toeplitz(col,row)
        Vmhalf[:p1,:p1] = A
        return Vmhalf

    def makeNiiOutput(self,dataNii,array,mode):
        '''
        dataNii: data reference Nibabel nifti object
        array:   numX x numY x numZ x numWhatever array
                 numWhatever can be 1

        Returns nii
        '''
        nii = nib.nifti1.Nifti1Image(array,dataNii.get_affine(),dataNii.header)
        nii.get_header()['pixdim'][4] = 0.
        nii.get_header()['glmin'] = 0
        nii.get_header()['glmax'] = 0
        nii.get_header()['cal_min'] = 0.
        nii.get_header()['cal_max'] = 0.
        if mode == 'beta':
            nii.get_header()['descrip'] = 'GLM beta weights'
            nii.get_header().set_intent('beta')
        elif mode == 'variance':
            nii.get_header()['descrip'] = 'GLM residual variance' 
            nii.get_header().set_intent('none')
        elif mode == 'contrast':
            nii.get_header()['descrip'] = 'Contrast scalar' 
            nii.get_header().set_intent('none')
        elif mode == 'ttest':
            nii.get_header()['descrip'] = 't map' 
            nii.get_header().set_intent('t test')
        elif mode == 'ftest':
            nii.get_header()['descrip'] = 'F map' 
            nii.get_header().set_intent('f test')
        else:
            nii.get_header()['descrip'] = 'Analysis output' 
            nii.get_header().set_intent('none')
        return nii

# ------------------------------
class BetweenSubjectDesignMan(CompMan):
    '''
    Stores between subjects (second-level) model design parameters.
    '''
    def __init__(self,cmMetaParam,cmBasePath):
        cmDesc    = 'betweensubjectdesign'
        cmCodeTag = os.path.splitext(os.path.split(inspect.getfile(inspect.currentframe()))[1])[0] + '_' + type(self).__name__
        CompMan.__init__(self,cmDesc,cmCodeTag,cmMetaParam,cmBasePath=cmBasePath)
        self.setConfig('longDescription'        , None)
        self.setConfig('makeDesignDictFunction' , None)
        self.setConfig('predictorList'          , None)
        self.setConfig('predictorTypeList'      , None)
        self.configure()

    def configure_bds001(self):
        if self.cmMetaParam != 'bds001':
            raise InvalidMetaparameterError(self.cmMetaParam)
        self.setConfig('longDescription', 'Between subject analysis. Simple offest model (column of ones).')
        self.setConfig('makeDesignDictFunction' , self.makeDesignDict_BetweenSubjectDesignScheme001)
        self.setConfig('predictorList'          , ['offset'])
        self.setConfig('predictorTypeList'      , ['indicator']) # can be 'indicator' or 'covariate'

    def makeDesignDict_BetweenSubjectDesignScheme001(self,subjectAnalysisManList):
        numSubjects                     = len(subjectAnalysisManList)
        designDict                      = OrderedDict()
        designDict['predictorList']     = self.predictorList
        designDict['predictorTypeList'] = self.predictorTypeList
        designDict['offset']            = range(numSubjects)
        return designDict

    def getOutput(self,subjectAnalysisManList):
        return self.makeDesignDictFunction(subjectAnalysisManList)

    def getPredictorList(self):
        return self.predictorList

    def getPredictorTypeList(self):
        return self.predictorTypeList

# ------------------------------
class BetweenSubjectContrastMan(CompMan):
    '''
    Statistical contrast manager for between subject analysis.

    getOutPut() returns contrastDictDict, a dictionary of dictionaries,
    each specifying one contrast by weighting each second level predictor.
    Keys of contrastDictDict are self.contrastNameList.
    Keys of each dict inside contrastDictDict include everything in
    self.predictorList as well as 'contrastType'.
    '''
    def __init__(self,cmMetaParam):
        cmDesc    = 'betweensubjectcontrast'
        cmCodeTag = os.path.splitext(os.path.split(inspect.getfile(inspect.currentframe()))[1])[0] + '_' + type(self).__name__
        CompMan.__init__(self,cmDesc,cmCodeTag,cmMetaParam)
        self.setConfig('predictorList'                , None)
        self.setConfig('contrastNameList'             , None)
        self.setConfig('contrastTypeList'             , None)
        self.setConfig('makeContrastDictDictFunction' , None)
        self.configure()
        self.contrastDictDict = self.makeContrastDictDictFunction()

    def getPredictorList(self):
        return self.predictorList

    def getContrastNameList(self):
        return self.contrastNameList

    def getContrastTypeList(self):
        return self.contrastTypeList

    def configure_offsetloc(self):
        if self.cmMetaParam != 'offsetloc':
            raise InvalidMetaparameterError(self.cmMetaParam)
        self.setConfig('predictorList'                , ['offset'])
        self.setConfig('contrastNameList'             , ['localizer'])
        self.setConfig('contrastTypeList'             , ['ttest'])
        self.setConfig('makeContrastDictDictFunction' , self.makeContrastDictDict_Offset1Localizer)

    def getOutput(self):
        '''
        Returns contrastDictDict.
        '''
        return self.makeContrastDictDictFunction()

    def makeContrastDictDict_Offset1Localizer(self):
        contrastDict1                  = OrderedDict()
        contrastDict1['contrastType']  = 'ttest'
        contrastDict1['predictorList'] = self.predictorList
        contrastDict1['offset']        = 1.0
        contrastDictDict               = OrderedDict()
        contrastDictDict['localizer']  = contrastDict1
        return contrastDictDict

# ------------------------------
class BetweenSubjectDesignMatrixMan(CompMan):
    def __init__(self,cmMetaParam):
        cmDesc    = 'betweensubjectdesignmatrix'
        cmCodeTag = os.path.splitext(os.path.split(inspect.getfile(inspect.currentframe()))[1])[0] + '_' + type(self).__name__
        CompMan.__init__(self,cmDesc,cmCodeTag,cmMetaParam)
        self.setConfig('longDescription'        , None)
        self.setConfig('designMatrixFunction'   , None)
        self.setConfig('contrastVectorFunction' , None)
        self.configure(self.cmMetaParam)

    def configure_bd01(self):
        self.setConfig('longDescription', ('Populates between subject design matrix using predidctor variables'
                                           'from BetweenSubjectDesignMan. Indicator predictors left alone. Covariate predictors'
                                           'mean-centred.'))
        self.setConfig('designMatrixFunction'   , self.makeBetweenSubjectDesignMatrix_v1)
        self.setConfig('contrastVectorFunction' , self.betweenSubjectContrastVector_V1)

    def getOutput(self,betweenSubjectDesignMan,subjectAnalysisManList):
        return self.designMatrixFunction(betweenSubjectDesignMan,subjectAnalysisManList)

    def makeBetweenSubjectDesignMatrix_v1(self,bsDesignMan,subjectAnalysisManList):
        '''
        Returns (designMat,indexDict)
        designMat:
            numSubjects X numBetweenSubjectPredictors
        indexDict, keyed by predictor, list of indices into
            designMat's columns

        bsDesignMan: BetweenSubjectDesignMan CompMan object
        '''
        predictorList     = bsDesignMan.getPredictorList()
        predictorTypeList = bsDesignMan.getPredictorTypeList()
        numPredictors     = len(predictorList)
        designDict        = bsDesignMan.getOutput(subjectAnalysisManList)
        numSubjects       = len(subjectAnalysisManList)
        designMat         = np.zeros((numSubjects,numPredictors),dtype = np.float)
        indexDict         = {}
        for (colIndex,(predictor,predictorType)) in enumerate(zip(predictorList,predictorTypeList)):
            if predictorType == 'indicator':
                designMat[designDict[predictor],colIndex] = 1.
            elif predictorType == 'covariate':
                designMat[:,colIndex] = designDict[predictor] - np.mean(designDict[predictor])
            else:
                raise ValueError('Invalid predictorType: {0}'.format(predictorType))
            indexDict[predictor] = colIndex
        return (designMat,indexDict)

    def getMaxIndex(self,indexDict):
        maxIndex = 0;
        for indexList in indexDict.itervalues():
            mx = max(indexList)
            maxIndex = max(maxIndex,mx)
        return maxIndex

    def combineIndexDict(self,indexDict1,indexDict2):
        newIndexDict = indexDict1.copy()
        for (key,value) in indexDict2.iteritems():
            if key in newIndexDict:
                raise Exception('Two indexDicts must have disjoint keys')
            newIndexDict[key] = value
        return newIndexDict

    def contrastVector(self,designMatrix,indexDict,contrastDict):
        return self.contrastVectorFunction(designMatrix,indexDict,contrastDict)

    def betweenSubjectContrastVector_V1(self,designMatrix,indexDict,contrastDict):
        conVec = np.zeros(designMatrix.shape[1],np.float)
        for predictor in contrastDict['predictorList']:
            weight = contrastDict[predictor]
            ind    = indexDict[predictor]
            if np.any(conVec[ind] != 0.):
                raise ValueError('contrastDict contains duplicate indices')
            conVec[ind] = float(weight)
        return conVec

# ------------------------------
class BetweenSubjectAnalysisMan(CompMan):
    def __init__(self,cmMetaParam,subjectAnalysisManList,betweenSubjectDesignMan,betweenSubjectContrastMan,maskMan,cmBasePath):
        cmDesc    = 'simbetweensubjectanalysis'
        cmCodeTag = os.path.splitext(os.path.split(inspect.getfile(inspect.currentframe()))[1])[0] + '_' + type(self).__name__
        CompMan.__init__(self,cmDesc,cmCodeTag,cmMetaParam,cmBasePath=cmBasePath)
        self.setConfig('longDescription'               , None)
        self.setConfig('subjectAnalysisManList'        , subjectAnalysisManList)
        self.setConfig('betweenSubjectDesignMan'       , betweenSubjectDesignMan)
        self.setConfig('betweenSubjectContrastMan'     , betweenSubjectContrastMan)
        self.setConfig('betweenSubjectDesignMatrixMan' , None)
        self.setConfig('maskMan'                       , maskMan)
        self.setConfig('analysisFunction'              , None)
        self.configure()

    def configure_bsa001(self):
        if self.cmMetaParam != 'bsa001':
            raise InvalidMetaparameterError(self.cmMetaParam)
        self.setConfig('longDescription'               , 'Between subject analysis modelling mean effect (offset column only).')
        self.setConfig('analysisFunction'              , self.doBetweenSubjectAnalysis_Procedure001)
        self.setConfig('betweenSubjectDesignMatrixMan' , BetweenSubjectDesignMatrixMan('bd01'))

    def getOutput(self):
        return self.analysisFunction()

    def getOutputFilesDicts(self):
        '''
        Returns (glmFilesDict,contrastFilesDict,spmFilesDict).
        '''
        outputPath   = self.getOutputPath()
        prefix       = self.getTagPrefix(True)
        glmFilesDict = OrderedDict()
        (_,subjectContrastFilesDict,_) = self.subjectAnalysisManList[0].getOutputFilesDicts()
        firstLevelContrastList = subjectContrastFilesDict.keys()
        for firstLevelContrast in firstLevelContrastList:
            for name in ('beta','variance'):
                glmFilesDict[(firstLevelContrast,name)] = os.path.join(outputPath,'{0}{1}fl_{2}{3}sl_{4}.nii'.format(prefix,self.cmSep,firstLevelContrast,self.cmSep,name))
        contrastFilesDict = OrderedDict()
        spmFilesDict      = OrderedDict()
        contrastDictDict  = self.betweenSubjectContrastMan.getOutput()
        for firstLevelContrast in firstLevelContrastList:
            for name in contrastDictDict.iterkeys():
                contrastFilesDict[(firstLevelContrast,name)] = os.path.join(outputPath,'{0}{1}fl_{2}{3}sl_contrast_{4}.nii'.format(prefix,self.cmSep,firstLevelContrast,self.cmSep,name))
                spmFilesDict[(firstLevelContrast,name)]      = os.path.join(outputPath,'{0}{1}fl_{2}{3}sl_spm_{4}.nii'.format(prefix,self.cmSep,firstLevelContrast,self.cmSep,name))
        return (glmFilesDict,contrastFilesDict,spmFilesDict)

    # Duplicated between SimSubjectAnalysisMan and BetweenSubjectAnalysisMan
    def makeOutputFiles(self,forceRebuild=False):
        print('Performing between subjects analysis')
        self.makeOutputPath()
        filesDictTuple = self.getOutputFilesDicts()
        if forceRebuild or not self.checkFilesExist(filesDictTuple):
            niiDictTuple = self.getOutput()
            for (filesDict,niiDict) in zip(filesDictTuple,niiDictTuple):
                for key in filesDict.iterkeys():
                    nib.save(niiDict[key],filesDict[key])
                    print('Saved file: ',filesDict[key])
        else:
            print('Files already exist')
        self.saveConfigCSVFile(forceRebuild)

    # Duplicated between SimSubjectAnalysisMan and BetweenSubjectAnalysisMan
    def checkFilesExist(self,filesDictTuple):
        '''
        Returns False if any file in filesDictTuple does NOT exist,
        otherwise returns True.
        filesDictTuple is a tuple of dictionaries whose values are
        file paths.
        '''
        for filesDict in filesDictTuple:
            for filePath in filesDict.itervalues():
                if not os.path.isfile(filePath):
                    return False
        return True

    def doBetweenSubjectAnalysis_Procedure001(self):
        '''
        Returns (glmNiiDict,contrastNiiDict,spmNiiDict).
        Those dicts are keyed by (firstLevelContrast,<name>) pairs, where
        <name> is 'beta', 'variance', or the second level contrast name.
        '''
        # Data
        (_,contrastFilesDict,_) = self.subjectAnalysisManList[0].getOutputFilesDicts()
        firstLevelContrastList = contrastFilesDict.keys()
        numSubjects     = len(self.subjectAnalysisManList)
        flatMaskData    = None
        glmNiiDict      = OrderedDict()
        contrastNiiDict = OrderedDict()
        spmNiiDict      = OrderedDict()
        for firstLevelContrast in firstLevelContrastList:
            print('Loading data for first level contrast {0}'.format(firstLevelContrast))
            data = None
            for (subjectIndex,subjectAnalysisMan) in enumerate(self.subjectAnalysisManList):
                (_,contrastFilesDict,_) = subjectAnalysisMan.getOutputFilesDicts()
                firstLevelContrastNiiFilePath = contrastFilesDict[firstLevelContrast]
                dataNii     = nib.load(firstLevelContrastNiiFilePath)
                subjectData = dataNii.get_data() # contrast amplitude data from one subject
                if data is None:
                    data = np.zeros(tuple(subjectData.shape)+(numSubjects,),np.float)
                data[:,:,:,subjectIndex] = subjectData
            if flatMaskData is None:
                flatMaskData = self.getMaskVariables(dataNii)
            flatData = np.reshape(data,(np.prod(data.shape[:3]),data.shape[3])).transpose()
            flatData = self.applyMask(flatData,flatMaskData)

            # GLM
            print('Fitting between subjects GLM for first level contrast {0}'.format(firstLevelContrast))
            (designMatrix,indexDict)   = self.getDesignMatrix()
            (flatBeta,flatVar,flatRes) = self.fitGLM(flatData,designMatrix)
            beta    = self.unflatten(self.reverseMask(flatBeta,flatMaskData),dataNii)
            var     = self.unflatten(self.reverseMask(flatVar,flatMaskData),dataNii)
            betaNii = self.makeNiiOutput(dataNii,beta,'beta')
            varNii  = self.makeNiiOutput(dataNii,var,'variance')
            glmNiiDict[(firstLevelContrast,'beta')]     = betaNii
            glmNiiDict[(firstLevelContrast,'variance')] = varNii

            # Contrasts and SPMs
            print('Computing between subjects statistics for first level contrast {0}'.format(firstLevelContrast))
            secondLevelContrastDictDict = self.betweenSubjectContrastMan.getOutput()
            for (secondLevelContrastName,secondLevelContrastDict) in secondLevelContrastDictDict.iteritems():
                (flatContrast,flatSpm) = self.computeContrast(flatBeta,flatVar,designMatrix,indexDict,secondLevelContrastName)
                contrast     = self.unflatten(self.reverseMask(flatContrast,flatMaskData),dataNii)
                spm          = self.unflatten(self.reverseMask(flatSpm,flatMaskData),dataNii)
                contrastNii  = self.makeNiiOutput(dataNii,contrast,'contrast')
                spmNii       = self.makeNiiOutput(dataNii,spm,secondLevelContrastDict['contrastType'])
                contrastNiiDict[(firstLevelContrast,secondLevelContrastName)] = contrastNii
                spmNiiDict[(firstLevelContrast,secondLevelContrastName)]      = spmNii

        return (glmNiiDict,contrastNiiDict,spmNiiDict)

    def getMaskVariables(self,dataNii):
        '''
        Returns (maskNii,maskData,flatMaskData)
        '''
        if self.maskMan is None:
            flatMaskData = None
        else:
            maskNii      = self.maskMan.getOutput()
            if (np.any(maskNii.get_header()['dim'][1:4] != dataNii.get_header()['dim'][1:4]) or
                np.any(maskNii.get_header()['pixdim'][1:4] != dataNii.get_header()['pixdim'][1:4])):
                raise Exception('dataNii and maskNii have mismatching dimensions.')
            maskData     = maskNii.get_data()
            flatMaskData = maskData.ravel()
        return flatMaskData

    def applyMask(self,flatArray,flatMaskData):
        '''
        flatArray: #whatever x #voxels, #whatever can be 1
        Return masked flatArray
        '''
        if flatMaskData is None:
            return flatArray
        if flatArray.ndim == 1:
            return flatArray[flatMaskData>0]
        if flatArray.ndim == 2:
            return flatArray[:,flatMaskData>0]
        raise Exception('flatArray must have ndim 1 or 2')

    def reverseMask(self,flatArray,flatMaskData):
        '''
        Returns unmaskedFlatArray
        '''
        if flatMaskData is None:
            unmaskedFlatArray = flatArray
        else:
            numVoxels = flatMaskData.size
            if flatArray.ndim == 1:
                unmaskedFlatArray = np.zeros(numVoxels,np.float)
                unmaskedFlatArray[flatMaskData>0] = flatArray
            elif flatArray.ndim == 2:
                unmaskedFlatArray = np.zeros((flatArray.shape[0],numVoxels),np.float)
                unmaskedFlatArray[:,flatMaskData>0] = flatArray
            else:
                raise Exception('flatArray must have ndim 1 or 2')
        return unmaskedFlatArray

    def unflatten(self,flatArray,dataNii):
        '''
        Returns array
        '''
        shape = dataNii.get_data().shape[:3]
        if flatArray.ndim == 1:
            array = np.reshape(flatArray,shape)
        elif flatArray.ndim == 2:
            array = np.reshape(flatArray.transpose(),shape+(flatArray.shape[0],))
        else:
            raise Exception('flatArray must have ndim 1 or 2')
        return array

    def getDesignMatrix(self):
        '''
        Return (designMatrix,indexDict).
        '''
        return self.betweenSubjectDesignMatrixMan.getOutput(self.betweenSubjectDesignMan,self.subjectAnalysisManList)

    def fitGLM(self,flatData,designMatrix):
        '''
        GLM fitting.
        Returns (flatBeta,flatVar)
        '''
        hatMatrix = np.dot(np.linalg.inv(np.dot(designMatrix.transpose(),designMatrix)),designMatrix.transpose())
        flatBeta  = np.dot(hatMatrix,flatData)
        flatEst   = np.dot(designMatrix,flatBeta)
        flatRes   = flatData - flatEst
        df        = designMatrix.shape[0] - designMatrix.shape[1]
        flatVar   = np.sum(flatRes**2,axis=0) / df
        return (flatBeta,flatVar,flatRes)

    def computeContrast(self,flatBeta,flatVar,designMatrix,indexDict,contrastName):
        '''
        Returns (flatContrast,flatSpm).
        '''
        contrastType = self.betweenSubjectContrastMan.getOutput()[contrastName]['contrastType']
        if contrastType == 'ttest':
            return self.computeContrast_tTest(flatBeta,flatVar,designMatrix,indexDict,contrastName)
        else:
            raise NotImplementedError('Not coded for contrastType {0}'.format(contrastType))

    def computeContrast_tTest(self,flatBeta,flatVar,designMatrix,indexDict,contrastName):
        contrastDict = self.betweenSubjectContrastMan.getOutput()[contrastName]
        conVec       = self.betweenSubjectDesignMatrixMan.contrastVector(designMatrix,indexDict,contrastDict)
        flatContrast = np.dot(conVec,flatBeta)
        mixMat       = np.linalg.inv(np.dot(designMatrix.transpose(),designMatrix))
        quadForm     = np.dot(conVec,np.dot(mixMat,conVec))
        denominator  = np.sqrt(quadForm) * np.sqrt(flatVar)
        flatSpm      = flatContrast / denominator # t values
        return (flatContrast,flatSpm)

    def makeNiiOutput(self,dataNii,array,mode):
        '''
        dataNii: data reference Nibabel nifti object
        array:   numX x numY x numZ x numWhatever array
                 numWhatever can be 1

        Returns nii
        '''
        nii = nib.nifti1.Nifti1Image(array,dataNii.get_affine(),dataNii.header)
        nii.get_header()['pixdim'][4] = 0.
        nii.get_header()['glmin'] = 0
        nii.get_header()['glmax'] = 0
        nii.get_header()['cal_min'] = 0.
        nii.get_header()['cal_max'] = 0.
        if mode == 'beta':
            nii.get_header()['descrip'] = 'Between subjects GLM beta weights'
            nii.get_header().set_intent('beta')
        elif mode == 'variance':
            nii.get_header()['descrip'] = 'Between subjects GLM residual variance' 
            nii.get_header().set_intent('none')
        elif mode == 'contrast':
            nii.get_header()['descrip'] = 'Between subjects Contrast scalar' 
            nii.get_header().set_intent('none')
        elif mode == 'ttest':
            nii.get_header()['descrip'] = 'Between subjects t map' 
            nii.get_header().set_intent('t test')
        elif mode == 'ftest':
            nii.get_header()['descrip'] = 'Between subjects F map' 
            nii.get_header().set_intent('f test')
        else:
            nii.get_header()['descrip'] = 'Between subjects analysis output' 
            nii.get_header().set_intent('none')
        return nii

# ------------------------------
class ClusterResultsMan(CompMan):
    def __init__(self,cmMetaParam,betweenSubjectAnalysisManDict,cmBasePath):
        cmDesc    = 'clusterresults'
        cmCodeTag = os.path.splitext(os.path.split(inspect.getfile(inspect.currentframe()))[1])[0] + '_' + type(self).__name__
        CompMan.__init__(self,cmDesc,cmCodeTag,cmMetaParam,cmBasePath=cmBasePath)
        self.setConfig('longDescription'               , None)
        self.setConfig('betweenSubjectAnalysisManDict' , betweenSubjectAnalysisManDict)
        #self.betweenSubjectAnalysisManDict = betweenSubjectAnalysisManDict
        self.setConfig('analysisFunction'              , None)
        self.configure()
        self.resultsDict = self.loadOutputFiles()

    def configure_woo01(self):
        if self.cmMetaParam != 'woo01':
            raise InvalidMetaparameterError(self.cmMetaParam)
        self.setConfig('longDescription'  , 'Summary of cluster results using Woo et al. 2014 measures and one more.')
        self.setConfig('analysisFunction' , self.clusterResultsAnalysis_v1)
        plist  = [0.1,0.05,0.02,0.01,0.001,0.0001,0.00001]
        cplist = [1-.5*p for p in plist]
        tfunc  = lambda x : scipy.stats.t.cdf(x,19) # df = 19
        tlist  = [muldivSearch(cp,tfunc) for cp in cplist]
        self.setConfig('localPThresholdList'      , plist)
        self.setConfig('localTThresholdList'      , tlist)
        self.setConfig('clusterSizeThresholdList' , [648,289,144,95,29,9,3])
            # Computed using fMRIMonteCluster.m in Matlab using FWHM value computed using SimulationMan.computeResidualsSmoothness()
            # using FWHM = 12.1331 mm
        self.measureFunctionList = [self.confusionMatrix,self.clusterTrueFalseLiberal,self.clusterTrueFalse50,self.clusterFWELiberal,self.clusterFWE50]

    def configure_emogonogo01(self):
        if self.cmMetaParam != 'emogonogo01':
            raise InvalidMetaparameterError(self.cmMetaParam)
        self.setConfig('longDescription'  , 'Summary of cluster results using Woo et al. 2014 measures and one more.')
        self.setConfig('analysisFunction' , self.clusterResultsAnalysis_v1)
        plist  = [0.1,0.05,0.02,0.01,0.001,0.0001,0.00001]
        cplist = [1-.5*p for p in plist]
        tfunc  = lambda x : scipy.stats.t.cdf(x,19) # df = 19
        tlist  = [muldivSearch(cp,tfunc) for cp in cplist]
        self.setConfig('localPThresholdList'      , plist)
        self.setConfig('localTThresholdList'      , tlist)
        self.setConfig('clusterSizeThresholdList' , [6426,2276,748,354,49,9,1])
            # Computed using scriptComputeClusterSizeThresholdRandomization.m in Matlab using
            # permutationOrSampling = 'permutation' with variables from 21 adolescent participants
            # 2010 emogonogo fMRI study, with numIterations = 5000.
            # With 50000 iterations, computed: [6535,2283,743,348,51,8,1], these were not available when I ran this Python code.
            # Cluster size threshold values using sampling with replacement: [7553,2202,693,312,33,4,0]
            # I chose to use the values from permutation testing instead, because they are more conservative except for
            # the local_p = 0.1 case, which we would never use for actualy fMRI analysis.
        self.measureFunctionList = [self.confusionMatrix,self.clusterTrueFalseLiberal,self.clusterTrueFalse50,self.clusterFWELiberal,self.clusterFWE50]

    def getTrialWeightsIndexSet(self):
        return set([key.trialWeightsIndex for key in self.betweenSubjectAnalysisManDict.iterkeys()])

    def getOutput(self):
        return self.resultsDict

    def getOutputFilesDict(self):
        '''
        Returns outputFilesDict
        '''
        outputPath   = self.getOutputPath()
        prefix       = self.getTagPrefix(True)
        sep          = self.cmSep
        outputFilesDict = OrderedDict()
        for localPThreshold in self.localPThresholdList:
            for mFunc in self.measureFunctionList:
                for trialWeightsIndex in self.getTrialWeightsIndexSet():
                    key = ClusterResultsKey(localPThreshold,mFunc.__name__,trialWeightsIndex)
                    pstr = str(localPThreshold).replace('.','p')
                    fstr = mFunc.__name__
                    wstr = 'twi{0}'.format(trialWeightsIndex)
                    fileName = '{0}{1}results_{2}_{3}_{4}.sav'.format(prefix,sep,pstr,fstr,wstr)
                    outputFilesDict[key] = os.path.join(outputPath,fileName)
        return outputFilesDict

    def makeOutputFiles(self,forceRebuild=False):
        print('Computing cluster results')
        self.makeOutputPath()
        filesDict = self.getOutputFilesDict()
        if not self.checkFilesExist(filesDict) or forceRebuild:
            resultsDict = self.analysisFunction()
            for localPThreshold in self.localPThresholdList:
                for mFunc in self.measureFunctionList:
                    for trialWeightsIndex in self.getTrialWeightsIndexSet():
                        key      = (localPThreshold,mFunc.__name__,trialWeightsIndex)
                        filePath = filesDict[key]
                        results  = resultsDict[key]
                        with open(filePath,'w') as f:
                            pickle.dump(results,f)
                        print('    Saved {}'.format(filePath))
        else:
            print('    Files already exists')
        self.saveConfigCSVFile(forceRebuild)
        self.resultsDict = self.loadOutputFiles()

    def loadOutputFiles(self):
        resultsDict = OrderedDict()
        filesDict = self.getOutputFilesDict()
        if self.checkFilesExist(filesDict):
            for localPThreshold in self.localPThresholdList:
                for mFunc in self.measureFunctionList:
                    for trialWeightsIndex in self.getTrialWeightsIndexSet():
                        key = ClusterResultsKey(localPThreshold,mFunc.__name__,trialWeightsIndex)
                        filePath = filesDict[key]
                        with open(filePath,'r') as f:
                            results = pickle.load(f)
                        resultsDict[key] = results
        else:
            return None
        return resultsDict

    def checkFilesExist(self,filesDict):
        '''
        Returns False if any file in filesDict does NOT exist,
        otherwise returns True.
        '''
        for filePath in filesDict.itervalues():
            if not os.path.isfile(filePath):
                return False
        return True

    def clusterResultsAnalysis_v1(self):
        '''
        Returns resultsDict,
        a dict of dicts, each containing a one or more lists of results from all iterations.
        All first and second level contrast results are combined together.
        '''
        resultsDict = self._initializeResultsDict()
        for (bsaKey,betweenSubjectAnalysisMan) in self.betweenSubjectAnalysisManDict.iteritems():
            trialWeightsIndex = bsaKey.trialWeightsIndex
            seedRegionMan = betweenSubjectAnalysisMan.subjectAnalysisManList[0].subjectDataMan.preprocSeedRegionMan.seedRegionMan
            seedRegionVol = seedRegionMan.getOutput().get_data()
            maskVol = seedRegionMan.maskMan.getOutput().get_data()
            (_,firstLevelContrastFilesDict,_) = betweenSubjectAnalysisMan.subjectAnalysisManList[0].getOutputFilesDicts()
            firstLevelContrastList = firstLevelContrastFilesDict.keys()
            (glmFilesDict,contrastFilesDict,spmFilesDict) = betweenSubjectAnalysisMan.getOutputFilesDicts()
            secondLevelContrastDictDict = betweenSubjectAnalysisMan.betweenSubjectContrastMan.getOutput()
            for firstLevelContrastName in firstLevelContrastList:
                for secondLevelContrastName in secondLevelContrastDictDict.keys():
                    spmFileKey  = (firstLevelContrastName,secondLevelContrastName)
                    spmFilePath = spmFilesDict[spmFileKey]
                    spmNii      = nib.load(spmFilePath)
                    spmData     = spmNii.get_data()
                    print('clusterResultsAnalysis: Computing results for {0}'.format(spmFilePath))
                    for (localPThreshold,localTThreshold,clusterSizeThreshold) in zip(self.localPThresholdList,self.localTThresholdList,self.clusterSizeThresholdList):
                        (binaryVol,labelVol,numLabels) = self.thresholdSPM(spmData,localTThreshold,clusterSizeThreshold)
                        for mFunc in self.measureFunctionList:
                            key = ClusterResultsKey(localPThreshold,mFunc.__name__,trialWeightsIndex)
                            results = mFunc(seedRegionVol,binaryVol,labelVol,numLabels,maskVol)
                            if resultsDict[key] is None:
                                resultsDict[key] = self._initializeResultsDictDict(results)
                            for (innerkey,val) in results.iteritems():
                                resultsDict[key][innerkey] += [val]
        return resultsDict

    def _initializeResultsDict(self):
        resultsDict = {}
        for localPThreshold in self.localPThresholdList:
            for mFunc in self.measureFunctionList:
                for trialWeightsIndex in self.getTrialWeightsIndexSet():
                    key = ClusterResultsKey(localPThreshold,mFunc.__name__,trialWeightsIndex)
                    resultsDict[key] = None
        return resultsDict

    def _initializeResultsDictDict(self,results):
        d = OrderedDict()
        for key in results.keys():
            d[key] = []
        return d

    def thresholdSPM(self,spmData,localTThreshold,clusterSizeThreshold):
        '''
        Returns (binaryVol,labelVol,numLabels).
        labelVol == 0 is background.
        labelVol == 1, 2, 3, etc. is clusters with labels 1, 2, 3, ...
        numLabels = # labels (i.e. clusters) INCLUDING the background (label 0)
        '''
        binaryVol = abs(spmData) >= localTThreshold
        structure = scipy.ndimage.generate_binary_structure(3,3)
        (labelVolPreCT,maxLabelsPreCT) = scipy.ndimage.label(binaryVol,structure=structure) # PreCT = pre-cluster-thresholding
        for labelInd in range(1,maxLabelsPreCT+1): # ignore label 0 = background
            if np.sum(labelVolPreCT == labelInd) < clusterSizeThreshold:
                binaryVol[labelVolPreCT == labelInd] = False
        (labelVol,maxLabel) = scipy.ndimage.label(binaryVol,structure=structure)
        numLabels = maxLabel + 1
        return (binaryVol,labelVol,numLabels)

    def confusionMatrix(self,seedRegionVol,binaryVol,labelVol,numLabels,maskVol):
        '''
        Return confusion matrix coded as a dictionary.
        '''
        confDict  = OrderedDict()
        numVoxels = int(np.sum(maskVol>0)) # int converts from numpy.memmap
        confDict['numVoxels'] = numVoxels
        confDict['truePositive']  = float(np.sum(np.logical_and(maskVol>0,np.logical_and(seedRegionVol>0 ,binaryVol>0 )))) / numVoxels
        confDict['trueNegative']  = float(np.sum(np.logical_and(maskVol>0,np.logical_and(seedRegionVol==0,binaryVol==0)))) / numVoxels
        confDict['falsePositive'] = float(np.sum(np.logical_and(maskVol>0,np.logical_and(seedRegionVol==0,binaryVol>0 )))) / numVoxels
        confDict['falseNegative'] = float(np.sum(np.logical_and(maskVol>0,np.logical_and(seedRegionVol>0 ,binaryVol==0)))) / numVoxels
        return confDict

    def clusterTrueFalseLiberal(self,seedRegionVol,binaryVol,labelVol,numLabels,maskVol):
        '''
        Return dict containing proportion of true positive
        clusters, liberal version.
        A cluster containing one or more true positive voxel is
        counted as a true positive.
        '''
        results = OrderedDict()
        numClusters = numLabels - 1 # do not count background
        numTruePosClusters = 0
        for clusterIndex in range(1,numLabels):
            numTruePos = int(np.sum(np.logical_and(seedRegionVol>0,labelVol==clusterIndex)))
            if numTruePos >= 1:
                numTruePosClusters += 1
        results['numClusters'] = numClusters
        if numClusters > 0:
            results['truePositiveClusters'] = float(numTruePosClusters) / numClusters
        else:
            results['truePositiveClusters'] = 0.
        return results

    def clusterTrueFalse50(self,seedRegionVol,binaryVol,labelVol,numLabels,maskVol):
        '''
        Return dict containing proportion of true positive
        clusters, 50% requirement version.
        A cluster containing 50% or more true positive voxels is
        counted as a true positive, otherwise false positive.
        '''
        results = OrderedDict()
        numClusters = numLabels - 1 # do not count background
        numTruePosClusters = 0
        for clusterIndex in range(1,numLabels):
            numTruePos = int(np.sum(np.logical_and(seedRegionVol>0,labelVol==clusterIndex)))
            numVoxelsInCluster = int(np.sum(labelVol==clusterIndex))
            if numTruePos >= numVoxelsInCluster / 2.:
                numTruePosClusters += 1
        results['numClusters'] = numClusters
        if numClusters > 0:
            results['truePositiveClusters'] = float(numTruePosClusters) / numClusters
        else:
            results['truePositiveClusters'] = 0.
        return results

    def clusterFWELiberal(self,seedRegionVol,binaryVol,labelVol,numLabels,maskVol):
        '''
        Returns dictionary with a Boolean: binary Vol contains one all
        true positive clusters, liberal version.
        A cluster containing one or more true positive voxel is
        counted as a true positive.
        '''
        results = OrderedDict()
        numClusters = numLabels - 1 # do not count background
        numTruePosClusters = 0
        for clusterIndex in range(1,numLabels):
            if not np.any(labelVol==clusterIndex):
                raise Exception('Invalid clusterIndex {0}'.format(clusterIndex))
            numTruePos = int(np.sum(np.logical_and(seedRegionVol>0,labelVol==clusterIndex)))
            if numTruePos >= 1:
                numTruePosClusters += 1
        results['numClusters'] = numClusters
        results['allClustersTruePositive'] = True if numTruePosClusters == numClusters else False
        return results

    def clusterFWE50(self,seedRegionVol,binaryVol,labelVol,numLabels,maskVol):
        '''
        Returns dictionary with a Boolean: binary Vol contains all 
        true positive clusters, 50% requirement version.
        Cluster-level family-wise error (FWE), liberal version.
        A cluster containing 50% or more true positive voxels is
        counted as a true positive, otherwise false positive.
        '''
        results = OrderedDict()
        numClusters = numLabels - 1 # do not count background
        numTruePosClusters = 0
        for clusterIndex in range(1,numLabels):
            numTruePos = int(np.sum(np.logical_and(seedRegionVol>0,labelVol==clusterIndex)))
            numVoxelsInCluster = int(np.sum(labelVol==clusterIndex))
            if numTruePos >= float(numVoxelsInCluster) / 2.:
                numTruePosClusters += 1
        results['numClusters'] = numClusters
        results['allClustersTruePositive'] = True if numTruePosClusters == numClusters else False
        return results

# ------------------------------
class ShowResults(object):
    '''
    Visual presentation of results from clusterResultsMan.
    '''


    def __init__(self,resultsDir,trialWeightsList=None):
        self.resultsDir       = resultsDir
        self.trialWeightsList = trialWeightsList
        self.resultsDict      = self.loadResultsFiles()
        self.plotFunctionList = [self.plot_confusionMatrix, self.plot_clusterTrueFalseLiberal, self.plot_clusterTrueFalse50,
                                 self.plot_clusterFWELiberal, self.plot_clusterFWE50]

    def loadResultsFiles(self):
        resultsDict = OrderedDict()
        (filesDict,localPThresholdList,measureFunctionList,trialWeightsIndexList) = self.getOutputFilesDict(self.resultsDir)
        self.localPThresholdList   = localPThresholdList
        self.measureFunctionList   = measureFunctionList
        self.trialWeightsIndexList = trialWeightsIndexList

        for localPThreshold in self.localPThresholdList:
            for funcName in self.measureFunctionList:
                for trialWeightsIndex in self.trialWeightsIndexList:
                    key = ClusterResultsKey(localPThreshold,funcName,trialWeightsIndex)
                    filePath = filesDict[key]
                    with open(filePath,'r') as f:
                        results = pickle.load(f)
                    resultsDict[key] = results
        return resultsDict

    def getOutputFilesDict(self,resultsDir):
        '''
        Returns (filesDict,localPThresholdList,measureFunctionList,trialWeightsIndexList).
        '''
        if not os.path.exists(resultsDir):
            raise IOError('resultsDir does not exist: {0}'.format(resultsDir))
        filesDict             = OrderedDict()
        localPThresholdList   = []
        measureFunctionList   = []
        trialWeightsIndexList = []

        for (dirPath,dirName,fileNameList) in os.walk(resultsDir):
            for fileName in fileNameList:
                if os.path.splitext(fileName)[1] != '.sav':
                    continue
                print(fileName)
                parts = fileName.split('.')[4].split('_')
                localPThreshold     = float(parts[1].replace('0p','0.'))
                measureFunction = parts[2]
                trialWeightsIndex   = int(parts[3].strip('twi'))
                key = ClusterResultsKey(localPThreshold,measureFunction,trialWeightsIndex)
                filesDict[key] = os.path.join(dirPath,fileName)

                localPThresholdList += [localPThreshold]
                measureFunctionList += [measureFunction]
                trialWeightsIndexList += [trialWeightsIndex]
        localPThresholdList   = list(set(localPThresholdList))
        localPThresholdList   = sorted(localPThresholdList,cmp=lambda x,y:int(float(x)-float(y)))
        measureFunctionList   = list(set(measureFunctionList))
        trialWeightsIndexList = list(set(trialWeightsIndexList))
        return (filesDict,localPThresholdList,measureFunctionList,trialWeightsIndexList)


    def getSortedLocalPThresholdArray(self):
        '''
        Returns localPThresholdArray
        '''
        localPThresholdArray = np.array(self.localPThresholdList)
        sortIndexArray = np.argsort(localPThresholdArray)
        localPThresholdArray = localPThresholdArray[sortIndexArray]
        return localPThresholdArray

    def plot_helper_1(self,measureFunction,resultsKey,yLabel,func=None,funcArgs=None,fig=1):
        if fig is not None:
            plt.figure(fig)
            plt.clf()
        localPThresholdArray = self.getSortedLocalPThresholdArray()
        curveList = []
        legendList = []
        for trialWeightsIndex in self.trialWeightsIndexList:
            curve = []
            for localPThreshold in localPThresholdArray:
                key = ClusterResultsKey(localPThreshold,measureFunction,trialWeightsIndex)
                if func is None:
                    curve += [1. - np.mean(self.resultsDict[key][resultsKey])]
                else:
                    curve += [func(self.resultsDict,key,funcArgs)]
            curve = np.array(curve)
            curveList += [curve]
            if self.trialWeightsList is not None:
                legendList += ['weight {0}'.format(self.trialWeightsList[trialWeightsIndex])] 
            else:
                legendList += ['index {0}'.format(trialWeightsIndex)] 

        ln = len(self.trialWeightsIndexList)
        colourList = ['#{0}00{1}'.format(self.getHex(255*i/ln),self.getHex(255*(ln-1-i)/ln)) for i in range(ln)]
        title      = measureFunction
        xLabel     = 'Local P Threshold'
        self.specPlot(localPThresholdArray,curveList,colourList,title,xLabel,yLabel,legendList,fig)

    def getHex(self,x):
        if x < 16:
            return '0' + hex(x)[-1]
        else:
            return hex(x)[2:]

    def specPlot(self,localPThresholdArray,curveList,colourList,title,xLabel,yLabel,legendList=[],fig=1):
        '''
        Plotting function.
        Flips localPThresholdArray around backward and plots evenly-spaced.
        '''
        for curve in curveList:
            print(curve)
        xArray = np.arange(1,len(localPThresholdArray)+1)
        for (curve,colour) in zip(curveList,colourList):
            plt.plot(xArray,curve[::-1],color=colour)
        for (curve,colour) in zip(curveList,colourList):
            plt.plot(xArray,curve[::-1],color=colour,marker='o')
        plt.xticks(xArray,[p for p in localPThresholdArray[::-1]])

        plt.title(title)
        plt.xlabel(xLabel)
        plt.ylabel(yLabel)
        if len(legendList)>0:
            plt.legend(legendList,loc='best')
        plt.show()

    def plot_confusionMatrix(self,fig=1):
        measureFunction = 'confusionMatrix'
        if fig is not None:
            plt.figure(fig)
            plt.clf()
        localPThresholdArray = self.getSortedLocalPThresholdArray()
        for trialWeightsIndex in self.trialWeightsIndexList:
            fpCurve = []
            fnCurve = []
            snCurve = []
            psCurve = []
            for localPThreshold in localPThresholdArray:
                key = ClusterResultsKey(localPThreshold,measureFunction,trialWeightsIndex)
                fpCurve += [np.mean(np.array(self.resultsDict[key]['falsePositive']) / 
                                    (np.array(self.resultsDict[key]['falsePositive']) +
                                     np.array(self.resultsDict[key]['truePositive'])))]
                fnCurve += [np.mean(np.array(self.resultsDict[key]['falseNegative']) /
                                    (np.array(self.resultsDict[key]['falseNegative']) + 
                                     np.array(self.resultsDict[key]['trueNegative'])))]
                snCurve += [np.mean(np.array(self.resultsDict[key]['truePositive']) / 
                                    (np.array(self.resultsDict[key]['falseNegative']) +
                                     np.array(self.resultsDict[key]['truePositive'])))]
                psCurve += [np.mean(np.array(self.resultsDict[key]['falsePositive']) +
                                    np.array(self.resultsDict[key]['truePositive']))]
            fpCurve = np.array(fpCurve)
            fnCurve = np.array(fnCurve)
            snCurve = np.array(snCurve)
            psCurve = np.array(psCurve)

            curveList  = [fpCurve,fnCurve,snCurve,psCurve]
            colourList = ['r','b','g','k']
            legendList = ['False discovery rate','False rejection rate','Sensitivity','Proportion positive'] 
            title      = measureFunction
            xLabel     = 'Local P Threshold'
            yLabel     = 'Proportion'
            self.specPlot(localPThresholdArray,curveList,colourList,title,xLabel,yLabel,legendList,fig)

    def plot_voxelSensitivity(self,fig=1):
        measureFunction = 'confusionMatrix'
        yLabel          = 'Voxelwise sensitivity'
        def func(resultsDict,key,*args,**kwArgs):
            return np.mean(np.array(resultsDict[key]['truePositive']) / 
                           (np.array(resultsDict[key]['falseNegative']) +
                            np.array(resultsDict[key]['truePositive'])))
        self.plot_helper_1(measureFunction,None,yLabel,func,None,fig)

    def plot_clusterTrueFalseLiberal(self,fig=1):
        measureFunction = 'clusterTrueFalseLiberal'
        resultsKey      = 'truePositiveClusters'
        yLabel          = 'Proportion False Positive Clusters'
        self.plot_helper_1(measureFunction,resultsKey,yLabel,fig)

    def plot_clusterTrueFalse50(self,fig=1):
        measureFunction = 'clusterTrueFalse50'
        resultsKey      = 'truePositiveClusters'
        yLabel          = 'Proportion False Positive Clusters'
        self.plot_helper_1(measureFunction,resultsKey,yLabel,fig)

    def plot_clusterFWELiberal(self,fig=1):
        measureFunction = 'clusterFWELiberal'
        resultsKey      = 'allClustersTruePositive'
        yLabel          = 'Proportion Analyses with >= 1 False Positive Cluster'
        self.plot_helper_1(measureFunction,resultsKey,yLabel,fig)

    def plot_clusterFWE50(self,fig=1):
        measureFunction = 'clusterFWE50'
        resultsKey      = 'allClustersTruePositive'
        yLabel          = 'Proportion Analyses with >= 1 False Positive Cluster'
        self.plot_helper_1(measureFunction,resultsKey,yLabel,fig)

    def plotAll(self,fig=1):
        plt.figure(fig)
        plt.clf()
        for (ind,func) in enumerate(self.plotFunctionList):
            plt.subplot(1,len(self.plotFunctionList),ind+1)
            func(None)

# ------------------------------
def getIndexRealCoordinates(nii):
    '''
    nii = nifti object from nibabel
    returns (iarray,jarray,karray,xarray,yarray,zarray)
    i,j,k arrays are indiced in the data volume
    x,y,z arrays are real coordinates
    Uses s-form matrix for coordinate transform.
    '''
    (ni,nj,nk) = nii.get_header().get_data_shape()
    [iarray,jarray,karray] = np.mgrid[0:ni,0:nj,0:nk]
    smat = nii.get_sform()
    xarray = np.zeros(iarray.shape)
    yarray = np.zeros(jarray.shape)
    zarray = np.zeros(karray.shape)
    for i,j,k in zip(iarray.ravel(),jarray.ravel(),karray.ravel()):
        [x,y,z,_] = np.dot(smat,[i,j,k,1])
        xarray[i,j,k] = x
        yarray[i,j,k] = y
        zarray[i,j,k] = z
    return (iarray,jarray,karray,xarray,yarray,zarray)

def getDistances(xarray,yarray,zarray,coord):
    return np.sqrt( (xarray-coord[0])**2 + (yarray-coord[1])**2 + (zarray-coord[2])**2)

# ------------------------------
def muldivSearch(target,func,lowx=0.,highx=100.,tolerance=1e-10,verbose=False):
    '''
    Finds argmin abs(func(x) - target)
    func must be monotonic increasing.
    func takes one scalar argument.
    Use lambda / closures to pass in extra args to func.

    eg: func = lambda x : scipy.stats.t.cdf(x,19)
    '''
    x = (lowx+highx)/2.
    val = func(x)
    count = 0
    maxCount = 1000
    if func(lowx) > target or func(highx) < target:
        raise Exception('Bracket range does not include target, change lowx and/or highx.')
    while abs(val-target) > tolerance and count < maxCount:
        count += 1
        if val > target:
            highx = x
            x = (x+lowx) / 2.
        elif val < target:
            lowx = x
            x = (x+highx) / 2.
        val = func(x)
    if verbose:
        print('Final count={0}, x={1}, value={2}'.format(count,x,val))
    if abs(val-target) <= tolerance:
        return x
    else:
        return None
    
# ------------------------------
def getPaths():
    dataPath = None
    scratchPath = None
    if 'uname' in dir(os) and os.uname()[0]=='Linux':
        dataPath    = '/data/2015_clusterthresholdsimulation'
        scratchPath = '/scratch/analysis/2015_clusterthresholdsimulation'
    elif 'uname' in dir(os) and os.uname()[0]=='Darwin' and 'lens' in os.uname()[1]:
        dataPath    = '/locdata/2015_clusterthresholdsimulation'
        scratchPath = '/locscratch/analysis/2015_clusterthresholdsimulation'
    elif 'uname' in dir(os) and os.uname()[0]=='Darwin':
        dataPath    = '/locdata/2015_clusterthresholdsimulation'
        scratchPath = '/locscratch/analysis/2015_clusterthresholdsimulation'
    return (dataPath,scratchPath)

def main(simulation_metaparameter):
    (dataPath,scratchPath)   = getPaths()
    simMan                   = SimulationMan(simulation_metaparameter,scratchPath,dataPath)
    return(simMan)

def main2(tag):
    trialWeightsList = None
    if tag == 'exp2':
        #resultsDir = '/locscratch/analysis/2015_clusterthresholdsimulation/clusterresults/clusterresults.ctsim_ClusterResultsMan.woo01.1189567758'
        #resultsDir = '/locscratch/analysis/2015_clusterthresholdsimulation/clusterresults/clusterresults.ctsim_ClusterResultsMan.woo01.1883507058'
        resultsDir = '/locscratch/analysis/2015_clusterthresholdsimulation/clusterresults/clusterresults.ctsim_ClusterResultsMan.woo01.3148951116'
        if os.path.split(resultsDir)[1] == 'clusterresults.ctsim_ClusterResultsMan.woo01.3148951116':
            trialWeightsList = [ 0.0, 0.01, 0.05, 0.1, 0.3, 0.6, 0.9, 1.2, 1.5]
    elif tag == 'exp3':
        resultsDir = '/locscratch/analysis/2015_clusterthresholdsimulation/clusterresults/clusterresults.ctsim_ClusterResultsMan.emogonogo01.2510431068'
        if os.path.split(resultsDir)[1] == 'clusterresults.ctsim_ClusterResultsMan.emogonogo01.2510431068':
            trialWeightsList = [ 0.0, 0.01, 0.05, 0.1, 0.3, 0.6, 0.9, 1.2, 1.5]
    else:
        raise ValueError('Invalid tag: {0}'.format(tag))
    showResults = ShowResults(resultsDir,trialWeightsList)
    return showResults

if __name__ == '__main__':
    #simman = main('exp2')
    #simman = main('exp3')
    #showResults = main2('exp2')
    showResults = main2('exp3')

