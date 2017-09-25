import os
import sys
import numpy as np
import scipy.ndimage
import nibabel
import nibabel as nib

#niftiFilePath='/locscratch/analysis/2015_clusterthresholdsimulation/simsubjectanalysis/simsubjectanalysis.ctsim_SubjectAnalysisMan.sa001.0019160939/simsubjectanalysis.ctsim_SubjectAnalysisMan.sa001.0136397921.spm_localizer.nii'

niftiFilePath='/locscratch/analysis/2015_clusterthresholdsimulation/simbetweensubjectanalysis/simbetweensubjectanalysis.ctsim_BetweenSubjectAnalysisMan.bsa001.4001868979/simbetweensubjectanalysis.ctsim_BetweenSubjectAnalysisMan.bsa001.4001868979.fl_localizer.sl_spm_localizer.nii'

def thresholdNifti(niftiFilePath,localTThreshold,clusterSizeThreshold):
    '''
    localTThreshold voxelwise t-threshold (two-tailed)
    clusterSizeThreshold in number voxels
    '''
    nii = nib.load(niftiFilePath)

    (dirPath,fileName) = os.path.split(niftiFilePath)
    (name,ext) = os.path.splitext(fileName)
    newFileName = '{0}_tth{1}_clth{2}{3}'.format(name,localTThreshold,clusterSizeThreshold,ext)
    newFilePath = os.path.join(dirPath,newFileName)

    spmData = nii.get_data()
    (binaryVol,_,_) = thresholdSPM(spmData,localTThreshold,clusterSizeThreshold)
    thresholdedSPMData = spmData.copy()
    thresholdedSPMData[binaryVol == 0] = 0

    if type(nii) == nibabel.nifti1.Nifti1Image:
        newNii = nib.nifti1.Nifti1Image(thresholdedSPMData,nii.get_affine(),header=nii.header)
    elif type(nii) == nibabel.nifti2.Nifti2Image:
        newNii = nib.nifti1.Nifti2Image(thresholdedSPMData,nii.get_affine(),header=nii.header)
    else:
        raise TypeError('Invalid type(nii) {0}'.format(type(nii)))
    print('Saving {0}'.format(newFilePath))
    nib.save(newNii,newFilePath)

def thresholdSPM(spmData,localTThreshold,clusterSizeThreshold):
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

def main():
    thresholdNifti(sys.argv[1],sys.argv[2],sys.argv[3])


localTThresholdList=[1.7291328113060445, 2.093024054192938, 2.539483190048486, 2.8609346132725477, 3.883405774831772, 4.897461831569672, 5.94935417175293]
clusterSizeThresholdList=[648, 289, 144, 95, 29, 9, 3]
for tth,cth in zip(localTThresholdList,clusterSizeThresholdList):
    thresholdNifti(niftiFilePath,tth,cth)
