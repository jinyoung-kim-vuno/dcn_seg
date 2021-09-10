import numpy as np
import stl
from stl import mesh
import vtk
import math

__all__ = ['combine_stl_files', 'combine_stl_objects']


def combine_stl_objects(stl_objects, center_mesh=True):
    combined_stl = mesh.Mesh(np.concatenate(stl_objects))

    if center_mesh:
        center_of_mass = np.zeros([1, 3])
        for index in range(0, combined_stl.vectors.shape[0]):
            for index2 in range(0, 3):
                center_of_mass = center_of_mass + combined_stl.vectors[index, index2, :]

        center_of_mass /= 3 * combined_stl.vectors.shape[0]

        for index in range(0, combined_stl.vectors.shape[0]):
            for index2 in range(0, 3):
                combined_stl.vectors[index, index2, :] = -center_of_mass + combined_stl.vectors[index, index2, :]

    return combined_stl


def combine_stl_files(list_of_stl_files, combined_stl_file, center_mesh=True, mode=stl.Mode.ASCII):
    # Read all of the input files
    stl_obj_lst = [mesh.Mesh.from_file(stl_file).data for stl_file in list_of_stl_files]

    # Create a single object from the component objects.
    combined_stl = combine_stl_objects(stl_obj_lst, center_mesh=center_mesh)

    # Save the result to the desired output file.
    combined_stl.save(combined_stl_file, mode=mode)


def decimate_and_smooth(target_decimation_ratio, number_of_iterations, vtk_surface_output_port):

    decimate = vtk.vtkDecimatePro()
    decimate.SetInputConnection(vtk_surface_output_port)
    decimate.SetTargetReduction(target_decimation_ratio)
    decimate.PreserveTopologyOn()
    decimate.SplittingOff()
    decimate.Update()

    smoother = vtk.vtkSmoothPolyDataFilter()
    smoother.SetInputConnection(decimate.GetOutputPort())
    smoother.SetNumberOfIterations(number_of_iterations)
    smoother.BoundarySmoothingOff()
    smoother.FeatureEdgeSmoothingOff()
    smoother.Update()

    return smoother


def convert_vtkPolyData_to_vtkImageData(smooth_output, inval, outval):
    white_image = vtk.vtkImageData()

    bounds = smooth_output.GetOutput().GetBounds()
    bounds_array = [0, 0, 0, 0, 0, 0]

    # convert bounds to normal list
    bounds_array[0] = bounds[0] - 1
    bounds_array[1] = bounds[1] + 1
    bounds_array[2] = bounds[2] - 1
    bounds_array[3] = bounds[3] + 1
    bounds_array[4] = bounds[4] - 1
    bounds_array[5] = bounds[5] + 1

    spacing = [0.2, 0.2, 0.2]

    white_image.SetSpacing(spacing)

    dim = []
    for index in range(0, 3):
        dim.append(int(math.ceil((bounds_array[index * 2 + 1] - bounds_array[index * 2]) / spacing[index])))

    white_image.SetDimensions(dim)
    white_image.SetExtent(0, dim[0] - 1, 0, dim[1] - 1, 0, dim[2] - 1)

    origin = [0, 0, 0]
    origin[0] = bounds_array[0] + spacing[0] / 2
    origin[1] = bounds_array[2] + spacing[1] / 2
    origin[2] = bounds_array[4] + spacing[2] / 2
    white_image.SetOrigin(origin)

    if vtk.VTK_MAJOR_VERSION <= 5:
        white_image.AllocateScalars()
    else:
        white_image.AllocateScalars(vtk.VTK_FLOAT, 1)

    # fill the image with foreground voxels:

    count = white_image.GetNumberOfPoints()

    for index in range(0, count):
        white_image.GetPointData().GetScalars().SetTuple1(index, inval)

    # polygonal data --> image stencil:
    poly_data_to_image_data = vtk.vtkPolyDataToImageStencil()
    poly_data_to_image_data.SetInputConnection(smooth_output.GetOutputPort())
    poly_data_to_image_data.SetOutputOrigin(origin)
    poly_data_to_image_data.SetOutputSpacing(spacing)
    poly_data_to_image_data.SetOutputWholeExtent(white_image.GetExtent())
    poly_data_to_image_data.Update()

    # cut the corresponding white image and set the background:
    imgstenc = vtk.vtkImageStencil()
    if vtk.VTK_MAJOR_VERSION <= 5:
        imgstenc.SetInput(white_image)
        imgstenc.SetStencil(poly_data_to_image_data.GetOutput())
    else:
        imgstenc.SetInputData(white_image)
        imgstenc.SetStencilConnection(poly_data_to_image_data.GetOutputPort())
    imgstenc.ReverseStencilOff()
    imgstenc.SetBackgroundValue(outval)
    imgstenc.Update()

    return imgstenc, spacing


def erode_and_dilate(image_stencil, inval, outval, kernel_size):

    erode = vtk.vtkImageDilateErode3D()
    if vtk.VTK_MAJOR_VERSION <= 5:
        erode.SetInput(image_stencil.GetOutput())
    else:
        erode.SetInputConnection(image_stencil.GetOutputPort())

    erode.SetErodeValue(inval)
    erode.SetDilateValue(outval)
    erode.SetKernelSize(kernel_size, kernel_size, kernel_size)
    erode.ReleaseDataFlagOff()
    erode.Update()

    dilate = vtk.vtkImageDilateErode3D()
    if vtk.VTK_MAJOR_VERSION <= 5:
        dilate.SetInput(erode.GetOutput())
    else:
        dilate.SetInputConnection(erode.GetOutputPort())

    dilate.SetDilateValue(inval)
    dilate.SetErodeValue(outval)
    dilate.SetKernelSize(kernel_size, kernel_size, kernel_size)
    dilate.ReleaseDataFlagOff()
    dilate.Update()

    return dilate


def measure_spd_stl(struct1, struct2):
    """
    :param struct1: stl filename
    :param struct2: stl filename
    :return: Averaged Euclidan distance between the structures surface points (in voxels)
    """
    mesh1 = mesh.Mesh.from_file(struct1)
    mesh2 = mesh.Mesh.from_file(struct2)
    verts1 = np.concatenate((mesh1.v0, mesh1.v1, mesh1.v2), axis=0)
    verts2 = np.concatenate((mesh2.v0, mesh2.v1, mesh2.v2), axis=0)
    min_s_dist_array = np.zeros(verts1.shape[0])
    for ind, surface_point in enumerate(verts1):
        min_s_dist_array[ind] = min(np.linalg.norm((verts2 - surface_point), axis=1))

    mean_surface_dist = np.mean(min_s_dist_array)
    return mean_surface_dist