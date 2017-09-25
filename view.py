import sys, os
sys.path.append(os.path.expanduser('~/code/python/fmri/2015_clusterthresholdsimulation'))
sys.path.append(os.path.expanduser('~/git/btfmri/python/'))
import easyfmri
import eftesting
from eftesting import *

if 'uname' in dir(os) and os.uname()[0]=='Linux':
    base_data_path    = '/data'
    base_scratch_path = '/scratch/analysis'
elif 'uname' in dir(os) and os.uname()[0]=='Darwin' and 'lens' in os.uname()[1]:
    base_data_path    = '/locdata'
    base_scratch_path = '/locscratch/analysis'
filePathList = None

filePathList   = []
#filePathList  += ['2015_clusterthresholdsimulation/residuals/residuals.MakeResiduals_emogonogo.era20_20150415a.679233458/residuals.MakeResiduals_emogonogo.era20_20150415a.679233458.subj01_run1_20100128_emogonogo.nii'] 
#filePathList += ['2015_clusterthresholdsimulation/residuals/residuals.MakeResiduals_emogonogo.era20_20150415a.679233458/mask_thresholded_average_epi.nii'] 
#filePathList += ['2015_clusterthresholdsimulation/residuals/residuals.MakeResiduals_emogonogo.era20_20150415a.679233458/residuals.MakeResiduals_emogonogo.era20_20150415a.679233458.subj04_run1_20100615_emogonogo.nii'] 
#filePathList  += ['2015_clusterthresholdsimulation/residuals/residuals.MakeResiduals_emogonogo.era20_20150415b.360470451/residuals.MakeResiduals_emogonogo.era20_20150415b.360470451.subj01_run1_20100128_emogonogo.nii'] 
#filePathList  += ['2015_clusterthresholdsimulation/residuals/residuals.MakeResiduals_emogonogo.era20_20150415c.3261238954/residuals.MakeResiduals_emogonogo.era20_20150415c.3261238954.subj01_run1_20100128_emogonogo.nii']
#filePathList  += ['2015_clusterthresholdsimulation/residuals/residuals.MakeResiduals_emogonogo.era20_20150415c.3261238954/residuals.MakeResiduals_emogonogo.era20_20150415c.3261238954.subj09_run4_20100716_emogonogo.nii']
#filePathList  += ['2015_clusterthresholdsimulation/residuals/residuals.MakeResiduals_emogonogo.era20_20150415c.3261238954/residuals.MakeResiduals_emogonogo.era20_20150415c.3261238954.subj03_run1_20100308_emogonogo_fixed.nii']
#filePathList  += ['2015_clusterthresholdsimulation/residuals/residuals.MakeResiduals_emogonogo.era20_20150415c.3261238954/residuals.MakeResiduals_emogonogo.era20_20150415c.3261238954.subj15_run1_20110304_emogonogo.nii']
#filePathList += ['atlases/xjview_atlases/TD_lobe.nii']
#filePathList += ['atlases/spm8_templates/EPI_mask_thresholded50.nii']
#filePathList += ['2015_clusterthresholdsimulation/residuals/residuals.MakeResiduals_emogonogo.era20_20150415a.679233458/mask_thresholded_average_epi.nii'] 
#filePathList += ['atlases/fsl-mni152-templates/MNI152_T1_2mm_brain_mask.nii.gz']
#filePathList += ['atlases/fsl-mni152-templates/MNI152_T1_2mm_VentricleMask.nii.gz']
#filePathList += ['atlases/mb_custom/boxes.nii']
#filePathList += ['atlases/mb_custom/ventricles_from_FSL_MNI152_T1_2mm_brain_mask.nii']
#filePathList += ['2015_clusterthresholdsimulation/mask/mask.ctsim_MaskMan.era20_combo1.0681379267/mask_era20_combo1.nii']

#filePathList = [os.path.join(base_data_path,x) for x in filePathList]

#filePathList += ['/locdata/2015_clusterthresholdsimulation/residuals/residuals.MakeResiduals_emogonogo.era20_20150415c.3261238954/residuals.MakeResiduals_emogonogo.era20_20150415c.3261238954.subj03_run1_20100308_emogonogo_fixed.nii']
#filePathList += ['/locdata/2015_clusterthresholdsimulation/residuals/other/residuals.MakeResiduals_emogonogo.era20_20150415c.3261238954.subj03_run1_20100308_emogonogo_fixed.nii']

#filePathList += ['/locdata/2015_clusterthresholdsimulation/mask/mask.ctsim_MaskMan.era20_combo1.0681379267/mask_era20_combo1.nii']
#filePathList += ['/locdata/2015_clusterthresholdsimulation/mask/mask_combo1_other/mask_era20_combo1.nii']

filePathList += ['/locscratch/analysis/2015_clusterthresholdsimulation/simsubjectdata/simsubjectdata.ctsim_SimSubjectDataMan.sd001.0014351923/simsubjectdata.ctsim_SimSubjectDataMan.sd001.0014351923.nii']
filePathList += ['/locdata/2015_clusterthresholdsimulation/residuals/residuals.MakeResiduals_emogonogo.era20_20150415c.3261238954/residuals.MakeResiduals_emogonogo.era20_20150415c.3261238954.subj03_run1_20100308_emogonogo_fixed.nii']
filePathList += ['/locscratch/analysis/2015_clusterthresholdsimulation/preprocseedregion/preprocseedregion.ctsim_PreprocSeedRegionSeriesMan.preproc_20131206.2354911930/preprocseedregion.ctsim_PreprocSeedRegionSeriesMan.preproc_20131206.2354911930.regionset_0000.nii']

#filePathList += ['/scratch/analysis/2015_clusterthresholdsimulation/seedregion/seedregion.ctsim_SeedRegionSeriesMan.woo2014_s001_18568.0433901185/seedregion.ctsim_SeedRegionSeriesMan.woo2014_s001_18568.0433901185.regionset_0000.nii']
#filePathList += ['/scratch/analysis/2015_clusterthresholdsimulation/preprocseedregion/preprocseedregion.ctsim_PreprocSeedRegionSeriesMan.preproc_20131206.2354911930/preprocseedregion.ctsim_PreprocSeedRegionSeriesMan.preproc_20131206.2354911930.regionset_0000.nii']

#filePathList += ['/scratch/analysis/2015_clusterthresholdsimulation/seedregion/seedregion.ctsim_SeedRegionSeriesMan.woo2014_s001_18568.0433901185/seedregion.ctsim_SeedRegionSeriesMan.woo2014_s001_18568.0433901185.regionset_0001.nii']
#filePathList += ['/scratch/analysis/2015_clusterthresholdsimulation/seedregion/seedregion.ctsim_SeedRegionSeriesMan.woo2014_s001_18568.0433901185/seedregion.ctsim_SeedRegionSeriesMan.woo2014_s001_18568.0433901185.regionset_0002.nii']

print(filePathList)

fileSeriesDirList = None

def fixWindowLevel(toplevel):
    w = toplevel.mriWidgetAxial
    for (uuid,filePath) in w.filePathDict.viewitems():
        print(filePath)
        if 'subj' in filePath:
            window = 5.
            level = 0.
            toplevel.dispatchFunction('applyToDerivedUUID','getUUIDForFilePath',filePath,'setWindow',window)
            toplevel.dispatchFunction('applyToDerivedUUID','getUUIDForFilePath',filePath,'setLevel',level)
        if 'mb_custom' in filePath:
            window = 1.
            level = 0.5
            toplevel.dispatchFunction('applyToDerivedUUID','getUUIDForFilePath',filePath,'setWindow',window)
            toplevel.dispatchFunction('applyToDerivedUUID','getUUIDForFilePath',filePath,'setLevel',level)
        if 'seed' in filePath:
            window = .0001
            level = 0. 
            toplevel.dispatchFunction('applyToDerivedUUID','getUUIDForFilePath',filePath,'setWindow',window)
            toplevel.dispatchFunction('applyToDerivedUUID','getUUIDForFilePath',filePath,'setLevel',level)

mriProvider = easyfmri.MRImageProvider(filePathList=filePathList, fileSeriesDirList=fileSeriesDirList)
toplevel = easyfmri.ThreeSliceAndTimecourseToplevel(root)
toplevel.setMRImageProvider(mriProvider)
#fixWindowLevel(toplevel)

if __name__ == '__main__':
    root.mainloop()
