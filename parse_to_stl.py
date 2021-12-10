import numpy as np
import SimpleITK as sitk
import vtk

from vtk.util import numpy_support
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkFiltersCore import vtkWindowedSincPolyDataFilter
from vtkmodules.vtkFiltersGeneral import vtkDiscreteMarchingCubes
from vtkmodules.vtkIOGeometry import vtkSTLWriter
from vtkmodules.vtkCommonCore import vtkLookupTable
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkPolyDataMapper,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkRenderer
)

import paths as p
import utils as u

def sitk2vtk(img, debugOn=False):
    """
    Code from: https://github.com/dave3d/dicom2stl/blob/main/utils/sitk2vtk.py
    """
    size = list(img.GetSize())
    origin = list(img.GetOrigin())
    spacing = list(img.GetSpacing())
    ncomp = img.GetNumberOfComponentsPerPixel()
    direction = img.GetDirection()
    i2 = sitk.GetArrayFromImage(img)
    if debugOn:
        i2_string = i2.tostring()
        print("data string address inside sitk2vtk", hex(id(i2_string)))
    vtk_image = vtk.vtkImageData()
    if len(size) == 2:
        size.append(1)
    if len(origin) == 2:
        origin.append(0.0)
    if len(spacing) == 2:
        spacing.append(spacing[0])
    if len(direction) == 4:
        direction = [ direction[0], direction[1], 0.0,
                      direction[2], direction[3], 0.0,
                               0.0,          0.0, 1.0 ]

    vtk_image.SetDimensions(size)
    vtk_image.SetSpacing(spacing)
    vtk_image.SetOrigin(origin)
    vtk_image.SetExtent(0, size[0] - 1, 0, size[1] - 1, 0, size[2] - 1)

    if vtk.vtkVersion.GetVTKMajorVersion()<9:
        print("Warning: VTK version <9.  No direction matrix.")
    else:
        vtk_image.SetDirectionMatrix(direction)

    depth_array = numpy_support.numpy_to_vtk(i2.ravel())
    depth_array.SetNumberOfComponents(ncomp)
    vtk_image.GetPointData().SetScalars(depth_array)

    vtk_image.Modified()
    if debugOn:
        print("Volume object inside sitk2vtk")
        print(vtk_image)
        print("num components = ", ncomp)
        print(size)
        print(origin)
        print(spacing)
        print(vtk_image.GetScalarComponentAsFloat(0, 0, 0, 0))
    return vtk_image

def parse_to_stl(input_path, output_path, show=False, name=None, smoothing_iterations=15, pass_band=0.001, feature_angle=120.0):
    image = sitk.ReadImage(input_path)
    vtk_obj = sitk2vtk(image, debugOn=False)

    discrete = vtkDiscreteMarchingCubes()
    discrete.SetInputData(vtk_obj)

    smoothing_iterations = smoothing_iterations
    pass_band = pass_band
    feature_angle = feature_angle

    smoother = vtkWindowedSincPolyDataFilter()
    smoother.SetInputConnection(discrete.GetOutputPort())
    smoother.SetNumberOfIterations(smoothing_iterations)
    smoother.BoundarySmoothingOff()
    smoother.FeatureEdgeSmoothingOff()
    smoother.SetFeatureAngle(feature_angle)
    smoother.SetPassBand(pass_band)
    smoother.NonManifoldSmoothingOn()
    smoother.NormalizeCoordinatesOn()
    smoother.Update()

    if output_path is not None:
        stlWriter = vtkSTLWriter()
        stlWriter.SetFileName(output_path)
        stlWriter.SetInputConnection(smoother.GetOutputPort())
        stlWriter.Write()

    if show:
        colors = vtkNamedColors()

        mapper = vtkPolyDataMapper()
        mapper.SetInputConnection(smoother.GetOutputPort())
        lut = vtkLookupTable()
        lut.SetNumberOfColors(2)
        lut.SetTableRange(0, 1)
        lut.SetScaleToLinear()
        lut.Build()
        lut.SetTableValue(0, 0.4, 0.4, 0.4, 1.0)
        lut.SetTableValue(1, 0.5, 0.8, 0.7, 1.0)
        mapper.SetLookupTable(lut)
        mapper.SetScalarRange(0, 1)

        ren = vtkRenderer()
        ren_win = vtkRenderWindow()
        ren_win.AddRenderer(ren)
        if name is None:
            ren_win.SetWindowName('STL Visualization')
        else:
            ren_win.SetWindowName(name)

        iren = vtkRenderWindowInteractor()
        iren.SetRenderWindow(ren_win)
        
        actor = vtkActor()
        actor.SetMapper(mapper)
        
        ren.AddActor(actor)
        ren.SetBackground(colors.GetColor3d('Black'))

        ren_win.Render()
        iren.Start()

def run():
    input_path = str(p.task_2_testing_path / str("sub07.nrrd"))
    output_path = str(p.stl_path / "sub07.stl")
    parse_to_stl(input_path, output_path)

    input_path = str(p.second_step_exp3_results_path / "Testing" / "Task2" / str("sub07.nrrd"))
    output_path = str(p.stl_path / "sub07_reconstruction.stl")
    parse_to_stl(input_path, output_path)

    input_path = str(p.implant_modeling_exp2_results_path / "Testing" / "Task2" / str("sub07.nrrd"))
    output_path = str(p.stl_path / "sub07_implant.stl")
    parse_to_stl(input_path, output_path)

if __name__ == "__main__":
    run()