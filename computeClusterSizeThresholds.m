function thresholdsize_list = computeClusterSizeThresholds()
%
% Computes cluster size threshold (numvoxels) for cluster size threshold simulations
%
% Matthew Brown
% July 2015

	addpath /home/mbrown/code/matlab/fmri/

	volumesize    = [57, 67, 50];
	localp_list   = [0.1, 0.05, 0.02, 0.01, 0.001, 0.0001, 0.00001];
	%localp_list   = [ 0.02];
	numiter_list  = [5000, 5000, 5000, 10000, 10000, 10000, 10000];

	oneortwotails = 2;
	globalp       = 0.05;
	fwhm          = 12.1331 / 3;
	maskFilePath  = '/data/2015_clusterthresholdsimulation/mask/mask.ctsim_MaskMan.era20_combo1.1360341165/mask_era20_combo1.nii';
	maskNii       = load_untouch_nii(maskFilePath);
	mask          = maskNii.img;
	volumeormass  = 'numvoxels';

	thresholdsize_list = [];
	for ind = 1:length(localp_list)
		localp        = localp_list(ind);
		numiter       = numiter_list(ind);
		thresholdsize = fMRIMonteCluster(volumesize,localp,oneortwotails,globalp,fwhm,mask,volumeormass,numiter);
		thresholdsize_list(ind) = thresholdsize;
		fprintf('**********\n')
		fprintf('localp %f, cluster size threshold %f\n',localp,thresholdsize)
		fprintf('**********\n\n')
	end
	
