from __future__ import print_function

import os
import subprocess
from shutil import copyfile

import nibabel as nib
import numpy as np
from sklearn import preprocessing
import scipy.ndimage
from scipy.ndimage import zoom
from dicom2nifti import settings

from utils.readITKtransform import readITKtransform
from utils.image_preprocessing import translate_to_best_init_position
# from utils.image_registration import *
from utils.file_utils import get_or_create_dir, get_file_size
from utils.path import strip_full_ext #, basename_no_ext


class BrainImage(object):
    """
    A class for loading images and operating on them.
    """
    def __init__(self, path_to_image, sequence):
        """
        
        :param path_to_image: a valid nifti file. An exception will be raised if the file doesn't exist.
        :param sequence: valid Sequence object or None. The BrainImage would be useful with a null sequence for 
        applying transformations but not for registrations since these require knowlegde of the sequence.
        """
        self._path = path_to_image
        if not os.path.exists(self.path_to_image):
            print('File Not Found: ' + self._path)
        self._sequence = sequence
        self._nii_file = None
        self._nii_data = None

    @classmethod
    def by_dsn(dsn, sequence=None):
        return BrainImage(dsn, sequence=sequence)

    def _load_nii_file(self):
        self._nii_file = nib.load(self.path_to_image)

    def _load_image_data(self):
        self._nii_data = self.nii_file.get_data()

    @property
    def path_to_image(self):
        return self._path

    @property
    def sequence(self):
        return self._sequence

    @property
    def nii_file(self):
        if self._nii_file is None:
            self._load_nii_file()
        return self._nii_file

    @property
    def nii_data(self):
        if self._nii_data is None:
            self._load_image_data()
        return self._nii_data

    def nii_data_normalized(self, bits=None, feature_range=(0, 1)):
        nifti_1d_array = self.nii_data.flatten()
        if bits is not None:
            feature_range = (0, 2 ** bits)
        nifti_normalize = preprocessing.minmax_scale(np.double(nifti_1d_array), feature_range=feature_range)
        return np.reshape(nifti_normalize, self.nii_data.shape)

    def rescale(self, ratio):
        return zoom(self.nii_data, ratio)

    def mask_image(self, mask_image_path=None):
        """
        Create a binary (0 and 1) mask of the image and save it to the same location with the
        name *_mask.nii.gz
        """
        input_file = self.path_to_image
        if not mask_image_path:
            mask_image_path = strip_full_ext(input_file) + '_mask.nii.gz'
        if not os.path.exists(mask_image_path):
            input_img = nib.load(input_file)
            input_data = input_img.get_data()
            mask_data = input_data.copy()
            mask_data[input_data < 1] = 0.0
            mask_data[input_data >= 1] = 1.0
            mask_img = nib.Nifti1Image(mask_data, affine=input_img.get_affine(), header=input_img.get_header())
            mask_img.set_data_dtype(np.float(16))
            mask_img.to_filename(mask_image_path)

        return mask_image_path

    def ResampleTo(self, out_file, new_affine=None, new_shape=None, new_voxel_size=None, is_ant_resample=True):
        """
        
        :param res_x: 
        :param res_y: 
        :param res_z: 
        :param out_file:        The resampled file name
        :param flow_step: 
        :return: 
        """
        img_path = self.path_to_image
        if is_ant_resample:

            res_str = '%sx%sx%s' % (str(new_voxel_size[0]), str(new_voxel_size[1]), str(new_voxel_size[2]))
            get_or_create_dir(os.path.dirname(out_file))

            cmd = "Resample_img.sh %s %s %s" % (img_path, out_file, res_str)
            subprocess.check_call(['bash ./scripts/' + cmd], shell=True)

        else:
            # using dicom2nifti library
            # https://github.com/icometrix/dicom2nifti/blob/master/dicom2nifti/resample.py#L108
            input_image = nib.load(img_path)
            output_image = self.resample_nifti_images([input_image], new_affine, new_shape, new_voxel_size)
            output_image.to_filename(out_file)

        return out_file

    def resample_nifti_images(self, nifti_images, new_affine= None, new_shape=None, voxel_size=None):
        """
        In this function we will create an orthogonal image and resample the original images to this space
        In this calculation we work in 3 spaces / coordinate systems
        - original image coordinates
        - world coordinates
        - "projected" coordinates
        This last one is a new rotated "orthogonal" coordinates system in mm where
        x and y are perpendicular with the x and y or the image
        We do the following steps
        - calculate a new "projection" coordinate system
        - calculate the world coordinates of all corners of the image in world coordinates
        - project the world coordinates of the corners on the projection coordinate system
        - calculate the min and max corners to get the orthogonal bounding box of the image in projected space
        - translate the origin back to world coordinages
        We now have the new xyz axis, origin and size and can create the new affine used for resampling
        """

        # get the smallest voxelsize and use that
        if voxel_size is None:
            voxel_size = nifti_images[0].header.get_zooms()
            for nifti_image in nifti_images[1:]:
                voxel_size = np.minimum(voxel_size, nifti_image.header.get_zooms())

        x_axis_world = np.transpose(np.dot(nifti_images[0].affine, [[1], [0], [0], [0]]))[0, :3]
        y_axis_world = np.transpose(np.dot(nifti_images[0].affine, [[0], [1], [0], [0]]))[0, :3]
        x_axis_world /= np.linalg.norm(x_axis_world)  # normalization
        y_axis_world /= np.linalg.norm(y_axis_world)  # normalization
        z_axis_world = np.cross(y_axis_world, x_axis_world)
        z_axis_world /= np.linalg.norm(z_axis_world)  # calculate new z
        y_axis_world = np.cross(x_axis_world, z_axis_world)  # recalculate y in case x and y where not perpendicular
        y_axis_world /= np.linalg.norm(y_axis_world)

        points_world = []

        for nifti_image in nifti_images:
            original_size = nifti_image.shape

            points_image = [[0, 0, 0],
                            [original_size[0] - 1, 0, 0],
                            [0, original_size[1] - 1, 0],
                            [original_size[0] - 1, original_size[1] - 1, 0],
                            [0, 0, original_size[2] - 1],
                            [original_size[0] - 1, 0, original_size[2] - 1],
                            [0, original_size[1] - 1, original_size[2] - 1],
                            [original_size[0] - 1, original_size[1] - 1, original_size[2] - 1]]

            for point in points_image:
                points_world.append(np.transpose(np.dot(nifti_image.affine,
                                                              [[point[0]], [point[1]], [point[2]], [1]]))[0, :3])

        projections = []
        for point in points_world:
            projection = [np.dot(point, x_axis_world),
                          np.dot(point, y_axis_world),
                          np.dot(point, z_axis_world)]
            projections.append(projection)

        projections = np.array(projections)

        min_projected = np.amin(projections, axis=0)
        max_projected = np.amax(projections, axis=0)
        new_size_mm = max_projected - min_projected

        origin = min_projected[0] * x_axis_world + \
                 min_projected[1] * y_axis_world + \
                 min_projected[2] * z_axis_world

        new_voxelsize = voxel_size
        if new_shape is None:
            new_shape = np.ceil(new_size_mm / new_voxelsize).astype(np.int16) + 1

        if new_affine is None:
            new_affine = self._create_affine(x_axis_world, y_axis_world, z_axis_world, origin, voxel_size)

        # Resample each image
        nifti_data_init = nifti_images[0].get_data()
        if np.size(np.shape(nifti_data_init)) == 4:
            nifti_data_init = nifti_data_init[:, :, :, 0]
        combined_image_data = np.full(new_shape, settings.resample_padding, dtype=nifti_data_init.dtype)
        for nifti_image in nifti_images:
            image_affine = nifti_image.affine
            combined_affine = np.linalg.inv(new_affine).dot(image_affine)
            matrix, offset = nib.affines.to_matvec(np.linalg.inv(combined_affine))
            nifti_data = nifti_image.get_data()
            if np.size(np.shape(nifti_data)) == 4:
                nifti_data = nifti_data[:, :, :, 0]
            resampled_image = scipy.ndimage.affine_transform(nifti_data,
                                                             matrix=matrix,
                                                             offset=offset,
                                                             output_shape=new_shape,
                                                             output=nifti_data.dtype,
                                                             order=settings.resample_spline_interpolation_order,
                                                             mode='constant',
                                                             cval=settings.resample_padding,
                                                             prefilter=False)
            combined_image_data[combined_image_data == settings.resample_padding] = \
                resampled_image[combined_image_data == settings.resample_padding]

        return nib.Nifti1Image(combined_image_data, new_affine)

    def _create_affine(self, x_axis, y_axis, z_axis, image_pos, voxel_sizes):
        """
        Function to generate the affine matrix for a dicom series
        This method was based on (http://nipy.org/nibabel/dicom/dicom_orientation.html)
        :param sorted_dicoms: list with sorted dicom files
        """

        # Create affine matrix (http://nipy.sourceforge.net/nibabel/dicom/dicom_orientation.html#dicom-slice-affine)

        affine = np.array(
            [[x_axis[0] * voxel_sizes[0], y_axis[0] * voxel_sizes[1], z_axis[0] * voxel_sizes[2], image_pos[0]],
             [x_axis[1] * voxel_sizes[0], y_axis[1] * voxel_sizes[1], z_axis[1] * voxel_sizes[2], image_pos[1]],
             [x_axis[2] * voxel_sizes[0], y_axis[2] * voxel_sizes[1], z_axis[2] * voxel_sizes[2], image_pos[2]],
             [0, 0, 0, 1]])
        return affine

    def RigidRegisterBrainImageTo(self, ref_brain_image, out_dir, affine_iter_num):
        """
        Produce rigid registration of the image to a reference using ANTS algorithm
        It does so by submitting Rigid_Reg.sh script to the cluster
        Input:
            ref_BrainImage - BrainImage which is the reference to the registration
            out_dir - String path to the directory the output files will be saved. By default it will
                      be saved to the same directory of the Image
        Output:
            out_dir - String path to registered file (image - out_file.nii.gz, transform - out_file_Affine.txt)
        """
        similarity_metrics = 'MI'

        ref_path = ref_brain_image.path_to_image
        moving_path = self.path_to_image
        out_file = strip_full_ext(moving_path) + '_Regto_%s_%s' % (basename_no_ext(ref_path), similarity_metrics)
        out_file = os.path.join(out_dir, os.path.basename(out_file))
        return rigid_register_bi_to(moving_path, ref_path, out_file, affine_iter_num)

    def AffineRegisterBrainImageTo(self, ref_brain_image, out_dir, affine_iter_num, is_ants_init_trans=True):
        """
        Produce affine registration of the image to a reference using ANTS algorithm
        It does so buy submitting Affine_Reg.sh script to the SGE using qsub
        Input:
            ref_BrainImage - BrainImage which is the reference to the registration
            out_dir - String path to the directory the output files will be saved. By default it will
                      be saved to the same directory of the Image
        Output:
            out_file - String path to registered file (image - out_file.nii.gz, transform - out_file_Affine.txt)
        """
        similarity_metrics = 'MI'
        ref_path = ref_brain_image.path_to_image
        img_path = self.path_to_image
        out_file = strip_full_ext(img_path) + '_Regto_%s_%s' % (basename_no_ext(ref_path), similarity_metrics)
        out_file = os.path.join(out_dir, os.path.basename(out_file))
        return affine_register_bi_to(img_path, ref_path, out_file, affine_iter_num, is_ants_init_trans)

    def nonlinear_reg_to(self, ref_brain_image, out_file, number_of_affine_iter,
                         number_of_nl_iter, similarity_metrics='CC', mask='',
                         regularize_grad=1, regularize_deformation=2, step=1, radius=''):

        """
        Produce non-linear registration of the BrainImage to a reference using ANTS algorithm
        
        :param ref_brain_image: BrainImage which is the reference to the registration
        :param out_file: path to the base output file
        :param flow_step:
        :param number_of_affine_iter: Number of iteration to use for the affine registration (done before 
                                      the non-linear)
        :param number_of_nl_iter:
        :param similarity_metrics: The cost function used for the registration (MI, CC). Default is CC.
        :param mask: Mask in the reference image coordinate where the cost function should be calculated.
                     Default is the mask of the reference image itself.
        :param regularize_grad:
        :param regularize_deformation:
        :param step:
        :param radius:
        """

        NL_script_file = 'Patient_NL_Reg.sh'

        img_path = self.path_to_image
        ref_path = ref_brain_image.path_to_image
        if not mask:
            mask = ref_brain_image.mask_image()

        reg_files = (out_file + '.nii.gz',
                     out_file + '_Affine.txt',
                     out_file + '_Warp.nii.gz',
                     out_file + '_InverseWarp.nii.gz')

        # Define the parameters for the cost function calculation
        weight = 1
        if not radius:
            if similarity_metrics == 'MI':
                radius = 64
            elif similarity_metrics == 'CC':
                radius = 4

        # check if the output of the process exists. This is NOT the exact name of the out_file - need to add a suffix
        # from the Patient_NL_Reg.sh script
        if get_file_size(out_file + '_Affine.txt') > 0:
            print("Registration was already done")
            pass
        else:
            cmd = "%s %s %s %s %s %s %s %s %s %s %s %s %s" % (NL_script_file, img_path, ref_path, out_file,
                                                              mask, similarity_metrics, number_of_affine_iter,
                                                              str(weight), str(radius), number_of_nl_iter,
                                                              str(regularize_grad),
                                                              str(regularize_deformation), str(step))
            subprocess.check_call(['bash ./scripts/' + cmd], shell=True)

        return reg_files

    def apply_transform_to(self, ref_brain_image, out_file, list_of_transforms=None):
        assert isinstance(ref_brain_image, BrainImage)
        return self.apply_transform_to_2(ref_brain_image.path_to_image, list_of_transforms, out_file)

    def apply_transform_to_2(self, ref_image_path, list_of_transforms, output_image_path):
        if list_of_transforms is None:
            list_of_transforms = ['--Id']

        cmd = "Apply_Transform.sh %s %s %s '%s'" % (self.path_to_image, ref_image_path,
                                                    output_image_path, " ".join(list_of_transforms))
        try:
            subprocess.check_call(['bash ./scripts/' + cmd], shell=True)
        except subprocess.CalledProcessError as e:
            print(e.returncode)
            print(e.output)

        return output_image_path

    def CompositeTransforms(self, ref_brain_image, out_dir, list_of_transforms=()):

        ref_path = ref_brain_image.path_to_image
        img_path = self.path_to_image
        out_file_name = '%s_Regto_%s.txt' % (basename_no_ext(img_path), basename_no_ext(ref_path))
        get_or_create_dir(out_dir)
        out_file = os.path.join(out_dir, out_file_name)

        return self.composite_transforms(list_of_transforms, out_file)

    def composite_transforms(self, list_of_transforms, composite_transform_path):
        """

        :param list_of_transforms:
        :param composite_transform_path: filename for composite transform output
        :param flow_step:
        :return: composite_transform_path
        """
        if len(list_of_transforms) == 0:
            list_of_transforms = ['--Id']

        cmd = "Composite_Transforms.sh %s '%s'" % (composite_transform_path, " ".join(list_of_transforms))
        try:
            subprocess.check_call(['bash ./scripts/' + cmd], shell=True)
        except subprocess.CalledProcessError as e:
            print(e.returncode)
            print(e.output)

        return composite_transform_path

    def ApplyTemplateTransformToMaskOf(self, atlas_transform_path, atlas_mask_path, patient_atlas_mask_path,
                                       patient_atlas_roi_image_path):
        """
        Creating a Mask of the image based on and ROI which was defined in the template space (e.g. mid-brain)
        It uses a script which is submitted to the SGE and apply the pre-calculated transform on the mask using
        Warpmultitranform and fslmaths to create the ROI
        
        :param atlas_transform_path: Transform from the mask space to the image. Usually this would be the transform
                                     calculated from template space to the image
        :param atlas_mask_path: Path to the mask. Usually it is a mask defined in the template space.
        :param patient_atlas_mask_path: Path where output mask will be saved.
        :param patient_atlas_roi_image_path: Path where output ROI will be saved.
        :param flow_step:
        :return: (patient_atlas_mask_path, patient_atlas_roi_image_path)
        """
        if not os.path.exists(atlas_transform_path):
            sis_log_and_exit(SLM.PREDICTION, SLC.WORKFILES_NOT_FOUND, "Atlas affine transform missing: %s",
                             atlas_transform_path)

        # Run script that apply the transform found on the atlas ROI
        cmd = 'register_Atlas_to_Image_qsub.sh "%s" "%s" "%s" "%s" "%s"' % \
              (atlas_mask_path, self.path_to_image, patient_atlas_mask_path, patient_atlas_roi_image_path,
               _remove_ending(atlas_transform_path, '_Affine.txt'))

        output_files = patient_atlas_mask_path, patient_atlas_roi_image_path
        subprocess.check_call(['bash ./scripts/' + cmd], shell=True)

        return output_files

    def MeasureSimilarityTo(self, brain_path, roi_mask, similarity_file_path, similarity_roi_image_path):
        """
        Measure the similarity of the image to another image. Uses a script with ANTS command MeasureSimilarity
        to calculate MI which is currently the similarity measure.
        
        :param brain_path: The brain image to which similarity should be measured.
        :param roi_mask: A mask which is applied to the moving image (self) for measuring the similarity in that ROI. 
                         Usually this should be the mask calculated using the atlas.
        :param similarity_file_path: File name where similarity results should be written.
        :param similarity_roi_image_path: File name where the ROI image should be created.
        :param flow_step:
        :return: (similarity_file_path, similarity_roi_image_path)
        """

        cmd = "Measure_Similarity_ROI.sh '%s' '%s' '%s' '%s' '%s'" % (self.path_to_image, brain_path, roi_mask,
                                                                      similarity_file_path, similarity_roi_image_path)
        output_files = (similarity_file_path, similarity_roi_image_path)
        subprocess.check_call(['bash ./scripts/' + cmd], shell=True)

        return output_files

    def composite_rigid_affine_registration_t1_to_ct(self, context, schema, ):

        """
        Produce composite rigid affine registration of T1 image to a reference CT using ANTS algorithms

        Input:
        :param context:
        :param schema:
        :type: schema: CompositeRigidAffinePreOpToPostOpRegistrationSchema
        :param flow_step:
        :returns: None

        For detailed description see:
        https://surgicalis.atlassian.net/wiki/pages/viewpage.action?spaceKey=SIS&title=T1+to+CT+registration%3A+algorithm+refinement+following+MDT+experiment
        """

        # 1) Initialization #############################################################################
        iter_num = '1000x1000x1000x10000'

        fixed_image_path = context.get_files_path(schema.fixed_image_path())

        # it is the user's responsibility ot make sure that this image is T1
        moving_image_path = self.path_to_image

        # Make a copy of T1 image, to be used as a moving copy towards the CT
        moving_image_copy = context.get_files_path(schema.working_moving_image_path())
        copyfile(moving_image_path, moving_image_copy)

        # 2) Compute CT mask #############################################################################
        # compute CT mask to put emphasis on skull matching in the first phase of registration

        print("Creating CT mask to prepare for registration")

        preliminary_mask_image_path = context.get_files_path(schema.preliminary_mask_image_path())
        number_of_thresholds = 3
        cmd = "Compute_CT_Mask.sh '%s' '%s' '%d'" % (fixed_image_path, preliminary_mask_image_path,
                                                     number_of_thresholds)
        subprocess.check_call(['bash ./scripts/' + cmd], shell=True)

        # 3) Compute rigid registration between T1 and CT mask ##########################################

        print("## Performing rough registration")

        registration_method = 'Rigid_Reg_T1_to_CT_optimized'
        computed_rigid_transform = context.get_files_path(schema.registration_transform_path(registration_method))

        cmd = "'%s.sh' '%s' '%s' '%s' '%s'" % (registration_method, preliminary_mask_image_path, moving_image_copy,
                                               _remove_ending(computed_rigid_transform, '_Affine.txt'), iter_num)
        subprocess.check_call(['bash ./scripts/' + cmd], shell=True)

        # applying the transformation on T1 image
        print('## Apply rigid registration')
        self.apply_transform_to_2(fixed_image_path, [computed_rigid_transform], moving_image_copy)

        # 4) compute mask for fine affine registration #################################################
        print("## Creating a mask for fine registration")
        mask_image_path = context.get_files_path(schema.mask_image_path())
        self.compute_mask_t1_ct(fixed_image_path, moving_image_copy, mask_image_path)

        # 5) compute affine registration between T1 and CT ##########################################

        registration_method = 'Affine_Reg_T1_to_CT_optimized'
        computed_affine_transform = context.get_files_path(schema.registration_transform_path(registration_method))

        print('## Affine registration')
        cmd = "'%s.sh' '%s' '%s' '%s' '%s' '%s'" % (registration_method, fixed_image_path, moving_image_copy,
                                                    _remove_ending(computed_affine_transform, '_Affine.txt'),
                                                    iter_num, mask_image_path)
        subprocess.check_call(['bash ./scripts/' + cmd], shell=True)

        # 6) composite the transforms, save them and and apply them on the T1 image ################

        affine_scale = self.compute_scaling_factor(computed_affine_transform)

        if affine_scale < 10:  # if scale is less than 10% add the affine transform
            computed_transforms_list = [computed_affine_transform, computed_rigid_transform]
        else:                  # if scaling is too high - ignore the affine part
            computed_transforms_list = [computed_rigid_transform]

        composite_transform_path = context.get_files_path(schema.composite_registration_transform_path())
        self.composite_transforms(computed_transforms_list, composite_transform_path)

        composite_registration_image_path = context.get_files_path(schema.composite_registration_image_path())
        self.apply_transform_to_2(fixed_image_path, computed_transforms_list, composite_registration_image_path)

    def compute_mask_t1_ct(self, fixed_image_path, moving_image_path, mask_image_path):

        cmd = "compute_affine_registration_mask_T1_to_CT.sh '%s' '%s' '%s'" % (fixed_image_path, moving_image_path,
                                                                               mask_image_path)
        subprocess.check_call(['bash ./scripts/' + cmd], shell=True)

        # color with black the lower part of the mask to avoid stretching of neck area
        lower_part_width = 20
        mask_image_obj = nib.load(mask_image_path)
        mask_image = mask_image_obj.get_data()
        non_zero_indexes = np.transpose(np.nonzero(mask_image))
        lowest_non_zero_row = np.amin(non_zero_indexes, axis=0)
        threshold_val = min([lowest_non_zero_row[2] + lower_part_width, int(mask_image_obj.shape[2])])
        zeros_block = np.zeros((int(mask_image_obj.shape[0]), int(mask_image_obj.shape[1]), lower_part_width))
        mask_image[0:zeros_block.shape[0],
                   0:zeros_block.shape[1],
                   range(lowest_non_zero_row[2], threshold_val)] = zeros_block
        mask_image_obj.get_data()[:] = mask_image
        nib.save(mask_image_obj, mask_image_path)

        return mask_image_path

    def composite_rigid_affine_registration_v2(self, init_reg_dir, image_name_pattern, sel_trn_patient_id, test_patient_id,
                                               ref_image, sel_trn_reg2_tst_image_name, do_translation_first=False):

        # 0) Initialization #############################################################################

        iter_num = '1000x1000x1000x2000'

        moving_image_path = self.path_to_image
        ref_image_path = ref_image.path_to_image

        sel_trn_image_name = sel_trn_patient_id + '_' + 'moving' + '_' + image_name_pattern

        # Make a copy of training T1 image, to be used as a moving copy towards the test T1
        moving_image_copy = os.path.join(init_reg_dir, sel_trn_image_name)
        copyfile(moving_image_path, moving_image_copy)

        computed_translations = []
        if do_translation_first:
            print('translate first!')
            # 1) Compute the translation between T2 and T1
            translated_moving_image_name = sel_trn_patient_id + '_' + 'translated_moving' + '_' + image_name_pattern
            translate_moving_image_path = os.path.join(init_reg_dir,
                                                       translated_moving_image_name)
            translated, _ = translate_to_best_init_position(ref_image_path, moving_image_copy,
                                                            strip_full_ext(translate_moving_image_path))

            # the moving image
            if translated:
                translate_transform_name = sel_trn_reg2_tst_image_name + '_translation.txt'
                computed_translations = os.path.join(init_reg_dir,
                                                           translate_transform_name)
                copyfile(translate_moving_image_path, moving_image_copy)

        # 2) Compute rigid registration ##########################################
        rigid_registration_method = 'Rigid_Reg_T2_to_T1_optimized'
        #rigid_registration_method = 'Rigid_Reg_optimized'

        transform_name = sel_trn_reg2_tst_image_name + '_' + rigid_registration_method + '_Affine.txt'
        computed_rigid_transform_path = os.path.join(init_reg_dir, transform_name)

        print('## rigid registration ###################################')
        cmd = "%s.sh %s %s %s %s" % (rigid_registration_method, ref_image_path, moving_image_copy,
                                     _remove_ending(computed_rigid_transform_path, '_Affine.txt'), iter_num)
        try:
            subprocess.check_call(['bash ./scripts/' + cmd], shell=True)
        except subprocess.CalledProcessError as e:
            print(e.returncode)
            print(e.output)

        # apply the transformation on the moving image
        print('## apply rigid registration ###################################')
        cmd = "Apply_Transform.sh %s %s %s '%s'" % (moving_image_path, ref_image_path, moving_image_copy,
                                                  " ".join([computed_rigid_transform_path] + computed_translations))
        try:
            subprocess.check_call(['bash ./scripts/' + cmd], shell=True)
        except subprocess.CalledProcessError as e:
            print(e.returncode)
            print(e.output)

        # 3) compute mask for fine affine registration #################################################
        print("Creating a mask for fine registration")

        mask_image_path = os.path.join(init_reg_dir, 'mask_fine_reg.nii.gz')
        self.compute_mask(ref_image_path, moving_image_copy, mask_image_path)

        # 4) compute affine registration between T2 and T1 ##########################################
        affine_registration_method = 'Affine_Reg_T2_to_T1_optimized'
        #affine_registration_method = 'Affine_Reg_optimized'

        transform_name = sel_trn_reg2_tst_image_name + '_' + affine_registration_method + '_Affine.txt'
        computed_affine_transform_path = os.path.join(init_reg_dir, transform_name)

        cmd = "%s.sh %s %s %s %s %s" % (affine_registration_method, ref_image_path, moving_image_copy,
                                        _remove_ending(computed_affine_transform_path, '_Affine.txt'), iter_num,
                                        mask_image_path)
        try:
            subprocess.check_call(['bash ./scripts/' + cmd], shell=True)
        except subprocess.CalledProcessError as e:
            print(e.returncode)
            print(e.output)

        # 5) composite the transforms, save them and and apply them on the T2 image ################
        affine_scale = self.compute_scaling_factor(computed_affine_transform_path)
        print('affine_scale: %s' % str(affine_scale))

        computed_transforms_list = [computed_affine_transform_path] if affine_scale < 10 else []
        computed_transforms_list.append(computed_rigid_transform_path)
        computed_transforms_list.extend(computed_translations)
        print(computed_transforms_list)

        transform_name = sel_trn_reg2_tst_image_name + '_' + 'composite_rigid_affine.txt'
        composite_transform_path = os.path.join(init_reg_dir, transform_name)
        self.composite_transforms(computed_transforms_list, composite_transform_path)

        # 6) Apply the composite transform to the moving image ######################
        composite_reg_img_name = sel_trn_reg2_tst_image_name + '_' + 'composite_rigid_affine.nii.gz'
        composite_registration_image_path = os.path.join(init_reg_dir,
                                                         composite_reg_img_name)
        self.apply_transform_to_2(ref_image_path, computed_transforms_list, composite_registration_image_path)


    def composite_rigid_affine_registration(self, context, schema, do_translation_first=False):

        """
        The moving image is self. Currently we support only T2 and T1 images
        The reference image is bi_reference.
        The result is a transformation from moving to the reference images.
        Produce composite rigid affine registration of T2 image to a reference T1 using ANTS algorithms

        :type context: Context
        :type schema: CompositeRigidAffineRegistrationSchema
        :type flow_step: FlowStep
        :type do_translation_first: bool
        :return: None
        """

        # 0) Initialization #############################################################################
        assert schema.fixed_image_sequence in [SEQ_T1, SEQ_T2]

        iter_num = '1000x1000x1000x2000'

        reference_image_path = context.get_files_path(schema.fixed_image_path())

        # it is the user's responsibility to make sure that this image is T2
        moving_image_path = self.path_to_image

        # Make a copy of T2 image, to be used as a moving copy towards the T1
        moving_image_copy = context.get_files_path(schema.working_moving_image_path())
        copyfile(moving_image_path, moving_image_copy)

        computed_translations = []

        if do_translation_first:
            # 1) Compute the translation between T2 and T1
            translate_moving_image_path = context.get_files_path(schema.translated_moving_image_path())

            translated, _ = translate_to_best_init_position(reference_image_path, moving_image_copy,
                                                            strip_full_ext(translate_moving_image_path))

            # the moving image
            if translated:
                computed_translations = [context.get_files_path(schema.translate_transform_path())]
                copyfile(translate_moving_image_path, moving_image_copy)

        # 2) Compute rigid registration ##########################################
        if schema.moving_image_sequence == SEQ_T2 and schema.fixed_image_sequence == SEQ_T1:
            registration_method = 'Rigid_Reg_T2_to_T1_optimized'
        else:
            registration_method = 'Rigid_Reg_optimized'

        computed_rigid_transform = context.get_files_path(schema.registration_transform_path(registration_method))

        print('## rigid registration ###################################')
        cmd = "%s.sh %s %s %s %s" % (registration_method, reference_image_path, moving_image_copy,
                                     _remove_ending(computed_rigid_transform, '_Affine.txt'), iter_num)
        subprocess.check_call(['bash ./scripts/' + cmd], shell=True)

        # apply the transformation on the moving image
        print('## apply rigid registration ###################################')
        cmd = "Apply_Transform.sh %s %s %s '%s'" % (moving_image_path, reference_image_path, moving_image_copy,
                                                    ' '.join([computed_rigid_transform] + computed_translations))
        subprocess.check_call(['bash ./scripts/' + cmd], shell=True)

        # 3) compute mask for fine affine registration #################################################
        print("Creating a mask for fine registration")
        mask_image_path = context.get_files_path(schema.mask_image_path())
        self.compute_mask(reference_image_path, moving_image_copy, mask_image_path)

        # 4) compute affine registration between T2 and T1 ##########################################
        if schema.moving_image_sequence == SEQ_T2 and schema.fixed_image_sequence == SEQ_T1:
            registration_method = 'Affine_Reg_T2_to_T1_optimized'
        else:
            registration_method = 'Affine_Reg_optimized'

        computed_affine_transform = context.get_files_path(schema.registration_transform_path(registration_method))

        sis_log_simple('## affine registration %s ##', computed_affine_transform)
        cmd = "%s.sh %s %s %s %s %s" % (registration_method, reference_image_path, moving_image_copy,
                                        _remove_ending(computed_affine_transform, '_Affine.txt'),
                                        iter_num, mask_image_path)
        subprocess.check_call(['bash ./scripts/' + cmd], shell=True)

        # 5) composite the transforms, save them and and apply them on the T2 image ################
        affine_scale = self.compute_scaling_factor(computed_affine_transform)

        computed_transforms_list = [computed_affine_transform] if affine_scale < 10 else []
        computed_transforms_list.append(computed_rigid_transform)
        computed_transforms_list.extend(computed_translations)

        composite_transform_path = context.get_files_path(schema.composite_registration_transform_path())
        self.composite_transforms(computed_transforms_list, composite_transform_path)

        # 6) Apply the composite transform to the moving image #######################3
        composite_registration_image_path = context.get_files_path(schema.composite_registration_image_path())
        self.apply_transform_to_2(reference_image_path, computed_transforms_list, composite_registration_image_path)

    def composite_rigid_Affine_registration_db_to_patient_t2(self, context, schema):
        """
        Produce DB T2 - patient T2 registration
        using composite rigid - affine registration of T2 image to a reference T1 using ANTS algorithms

        For detailed description see:
        https://surgicalis.atlassian.net/wiki/x/AoC1

        :type context: Context
        :type schema: CompositeRigidAffineRegistrationDbToPatientSchema
        """

        # 1) Initialization #############################################################################
        iter_num = '1000x1000x1000x2000'

        patient_T2_brain_path = context.get_files_path(schema.fixed_image_path())

        # it is the user's responsibility ot make sure that this image is T2
        db_T2_brain_path = self.path_to_image

        db_t2_moving_image_path = context.get_files_path(schema.working_moving_image_path())
        copyfile(db_T2_brain_path, db_t2_moving_image_path)

        # 2) Compute rigid registration between DB and patient  ##########################################
        registration_method = 'Rigid_Reg_optimized'
        rigid_transform_path = context.get_files_path(schema.registration_transform_path(registration_method))

        print('## rigid registration ###################################')
        cmd = "%s.sh %s %s %s %s" % (registration_method, patient_T2_brain_path, db_t2_moving_image_path,
                                     _remove_ending(rigid_transform_path, '_Affine.txt'), iter_num)
        subprocess.check_call(['bash ./scripts/' + cmd], shell=True)

        # apply the transformation on T2 image
        print('## apply rigid registration ###################################')
        cmd = "Apply_Transform.sh %s %s %s %s" % (db_T2_brain_path, patient_T2_brain_path, db_t2_moving_image_path,
                                                  rigid_transform_path)
        subprocess.check_call(['bash ./scripts/' + cmd], shell=True)

        # 3) compute mask for fine affine registration #################################################
        print("Creating a mask for fine registration")
        mask_image_path = context.get_files_path(schema.mask_image_path())
        self.compute_mask(patient_T2_brain_path, db_t2_moving_image_path, mask_image_path)

        # if a mask is empty or does not exist, do affine registration without it
        if not os.path.exists(mask_image_path) or self._voxel_count(mask_image_path) < 1000:
            mask_image_path = None

        # 4) compute affine registration between T2 and T1 ##########################################
        registration_method = 'Affine_Reg_optimized'
        affine_transform_path = context.get_files_path(schema.registration_transform_path(registration_method))

        print('## affine registration ###################################')
        cmd = "%s.sh %s %s %s %s %s" % (registration_method, patient_T2_brain_path, db_t2_moving_image_path,
                                        _remove_ending(affine_transform_path, '_Affine.txt'), iter_num,
                                        mask_image_path or '')
        subprocess.check_call(['bash ./scripts/' + cmd], shell=True)

        # 5) composite the transforms
        print('## composite transforms and apply registration ###################################')
        affine_scale = self.compute_scaling_factor(affine_transform_path)

        if affine_scale < 10:  # if scale is less than 10% add the affine transform
            computed_transforms_list = [affine_transform_path, rigid_transform_path]
        else:                  # if scaling is too high - ignore the affine part
            computed_transforms_list = [rigid_transform_path]

        patient_T2_brain_image = BrainImage(patient_T2_brain_path, SEQ_T2)

        ref_path = patient_T2_brain_image.path_to_image
        composite_transform_path = context.get_files_path(schema.composite_registration_transform_path())
        if len(computed_transforms_list) == 0:
            computed_transforms_list = ['--Id']
        cmd = "Composite_Transforms.sh %s '%s'" % (composite_transform_path, " ".join(computed_transforms_list))
        subprocess.check_call(['bash ./scripts/' + cmd], shell=True)

        # 6) Apply the composite transform to the moving image #######################3
        composite_registration_image_path = context.get_files_path(schema.composite_registration_image_path())
        if len(computed_transforms_list) == 0:
            computed_transforms_list = ['--Id']
        cmd = "Apply_Transform.sh %s %s %s '%s'" % (db_T2_brain_path, ref_path,
                                                    strip_full_ext(composite_registration_image_path),
                                                    " ".join(computed_transforms_list))
        subprocess.check_call(['bash ./scripts/' + cmd], shell=True)

    def compute_mask(self, fixed_image_path, moving_image_path, mask_image_path):
        """
        computes the mask and places the result in a directory relative to the input moving (self) brain image.

        :param moving_image_path:
        :param fixed_image_path:
        :param mask_image_path:
        :return: 
        """
        print('compute_mask[%s, %s] => %s' % (fixed_image_path, moving_image_path, mask_image_path))
        cmd = "%s %s %s %s" % ('compute_affine_registration_mask.sh', fixed_image_path,
                               moving_image_path, mask_image_path)
        subprocess.check_call(['bash ./scripts/' + cmd], shell=True)

    # return the scale factor in % units
    def compute_scaling_factor(self, transform_filename):
        transform = readITKtransform(transform_filename)
        # noinspection PyTypeChecker
        return abs(1 - np.linalg.det(transform)) * 100

    def _voxel_count(self, mask_filename):
        mask_file = nib.load(mask_filename)
        mask_image = mask_file.get_data()
        return np.sum(mask_image)


def _remove_ending(s, ending):
    if not s.endswith(ending):
        raise ValueError('{} does not end with {}'.format(repr(s), repr(ending)))

    return s[:len(s) - len(ending)]
