# DCN-Net
Deep Cerebellar Nuclei Segmentation via Semi-Supervised Deep Context-Aware Learning from 7T Diffusion MRI, IEEE Access, 2020

# How to run the code

- set a source path and your root path in run_dnn.sh
- designate target as dcn
- run sequentially multiple subjects
- set gpu_id=’0’ for GPU processing
- set is_reg=’1’ for registration based ROI localization
- set is_reg=’0’ and roi_pos (start_x_pos, end_x_pos, start_y_pos, end_y_pos, start_z_pos, end_z_pos measured from visualization tool (e.g., itk-snap)) for manual ROI localization (if registration fails)
- For now it requires all the training data (in dataset_path) for testing due to ROI localization and data normalization (later I will update the code to process without training data during the test phase)
- It is possible to train a model in current code by changing ‘mode’ if needed (mode 0; n-fold cross-validation , mode 1; training and testing with designated training set and test set (in config.py), mode 2: only testing with a trained model)
- please don’t change other python parameters

$ bash ./run_dnn.sh dcn patient_id#1, patient_id#2,...
