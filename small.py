
import nibabel as nib
import numpy as np


def MakeBoxesMask():
    infilepath = '/data/atlases/fsl-mni152-templates/MNI152_T1_2mm_brain_mask.nii.gz'
    newfilepath = '/data/atlases/mb_custom/boxes.nii'
    nii = nib.load(infilepath)
    d = nii.get_data()
    d[:] = 0
    smat = nii.get_sform()
    ismat = np.linalg.inv(smat)
    # params = centre x,y,z and radius, all in mm in real brain space
    param_list = [[  0. ,  8.   ,  13. , 13.],
                  [  0. ,  -47. ,   4. , 13.],
                  [-28. ,  -47. ,   4. ,  9.],
                  [ 28. ,  -47. ,   4. ,  9.],
                  [  0. ,  -44. , -33. ,  5.],
                  ]
    for p in param_list:
        c = np.array(np.round(np.dot(ismat,p[:3] + [1.])[:3]),dtype=np.uint16)
        r = int(p[3])
        d[c[0]-r:c[0]+r+1, c[1]-r:c[1]+r+1, c[2]-r:c[2]+r+1] = 1

    newnii = nib.nifti1.Nifti1Image(d,smat,nii.header)
    newnii.header['descrip'] = np.array('MB custom mask')
    nib.save(newnii,newfilepath)
    print('Saved {}'.format(newfilepath))


def MakeVentriclesMask():
    infilepath1 = '/data/atlases/fsl-mni152-templates/MNI152_T1_2mm_brain_mask.nii.gz'
    infilepath2 = '/data/atlases/mb_custom/boxes.nii'
    newfilepath = '/data/atlases/mb_custom/ventricles_from_FSL_MNI152_T1_2mm_brain_mask.nii'
    nii1 = nib.load(infilepath1)
    nii2 = nib.load(infilepath2)
    d = np.array(np.logical_and(nii1.get_data()<.5,nii2.get_data()>.5),dtype=np.uint8)
    newnii = nib.nifti1.Nifti1Image(d,nii1.get_affine(),nii1.header)
    newnii.header['descrip'] = np.array('MB custom ventricles mask')
    nib.save(newnii,newfilepath)
    print('Saved {}'.format(newfilepath))

MakeBoxesMask()
MakeVentriclesMask()
