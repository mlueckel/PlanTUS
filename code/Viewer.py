import sys
import nibabel as nib
import vtk
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
try:
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QVBoxLayout, QWidget, QRadioButton,QCheckBox,
        QToolBar, QFileDialog, QSlider, QLabel, QComboBox, QHBoxLayout,QPushButton,QSizePolicy
    )
    from PySide6.QtGui import QAction, QPainter, QImage
    from PySide6.QtCore import Qt
except:
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QVBoxLayout, QWidget, QRadioButton, QCheckBox,QAction,
        QToolBar, QFileDialog, QSlider, QLabel, QComboBox, QHBoxLayout, QPushButton, QSizePolicy
    )
    from PyQt5.QtGui import QPainter, QImage
    from PyQt5.QtCore import Qt
import numpy as np
from vtk.util import numpy_support
import trimesh


def _capture_vtk_widget_as_qimage(vtk_widget):
    """Capture a QVTKRenderWindowInteractor's actual rendered content as
    a QImage.

    VTK typically renders to a native OpenGL surface that bypasses
    Qt's normal paint/composite pipeline, so a plain QWidget.grab() on
    a widget containing one comes back blank for that region — this
    reads the real framebuffer content directly via VTK's own capture
    mechanism instead.
    """
    render_window = vtk_widget.GetRenderWindow()
    render_window.Render()  # ensure the back buffer holds the current frame

    w2i = vtk.vtkWindowToImageFilter()
    w2i.SetInput(render_window)
    w2i.SetInputBufferTypeToRGB()
    # Back buffer, not front: the front buffer reflects whatever's
    # actually composited on-screen at the moment of capture, which can
    # be wrong if another window is overlapping/occluding this one. The
    # back buffer holds the fully-rendered current frame regardless —
    # validated directly against a known scene before shipping this.
    w2i.ReadFrontBufferOff()
    w2i.Update()

    vtk_image = w2i.GetOutput()
    width, height, _ = vtk_image.GetDimensions()
    scalars = vtk_image.GetPointData().GetScalars()
    arr = numpy_support.vtk_to_numpy(scalars).reshape(height, width, 3)
    arr = np.ascontiguousarray(np.flipud(arr))  # VTK is bottom-up; Qt is top-down

    qimage = QImage(arr.data, width, height, arr.strides[0], QImage.Format_RGB888)
    return qimage.copy()  # copy so it owns its data once `arr` goes out of scope


def _make_transformed_glyph(source, color, opacity=1.0):
    """A vtk polydata source + a persistent 4x4 matrix/transform, so its
    orientation/scale/position can all be updated in place (via
    _set_glyph_pose / _set_cylinder_pose) without rebuilding the
    pipeline on every update. Shared by the transducer cylinder glyph
    (GiftiViewer) and the focus ellipsoid glyph (VolumeFocusViewer)."""
    matrix = vtk.vtkMatrix4x4()
    matrix.Identity()
    transform = vtk.vtkTransform()
    transform.SetMatrix(matrix)

    transform_filter = vtk.vtkTransformPolyDataFilter()
    transform_filter.SetTransform(transform)
    transform_filter.SetInputConnection(source.GetOutputPort())

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(transform_filter.GetOutputPort())
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(*color)
    actor.GetProperty().SetOpacity(opacity)
    return {'source': source, 'matrix': matrix, 'transform': transform,
            'transform_filter': transform_filter, 'mapper': mapper, 'actor': actor}


def _set_glyph_pose(glyph, center, u, v, n, a, b, c):
    """Position/orient/scale a glyph built on a source whose local X/Y/Z
    axes should map to world (u, v, n) with extents (a, b, c) — the
    convention that matches vtkSphereSource (isotropic, so any axis
    assignment is equally valid), used for the focus ellipsoid."""
    m = glyph['matrix']
    for r in range(3):
        m.SetElement(r, 0, a * u[r])
        m.SetElement(r, 1, b * v[r])
        m.SetElement(r, 2, c * n[r])
        m.SetElement(r, 3, center[r])
    glyph['transform'].SetMatrix(m)
    glyph['transform'].Modified()


def _set_cylinder_pose(glyph, center, u, v, n, radius, height):
    """Position/orient/scale a glyph built on vtkCylinderSource, whose
    axis runs along local Y (VTK's convention) rather than Z — mapped
    here to world direction n (the trajectory), with the circular
    cross-section (radius) in the u/v plane."""
    m = glyph['matrix']
    for r in range(3):
        m.SetElement(r, 0, radius * u[r])
        m.SetElement(r, 1, height * n[r])
        m.SetElement(r, 2, radius * v[r])
        m.SetElement(r, 3, center[r])
    glyph['transform'].SetMatrix(m)
    glyph['transform'].Modified()


def _trajectory_frame(normal):
    """Build an orthonormal (u, v, n) frame from a direction n, so u and
    v are each perpendicular to n and to each other."""
    n = normal / np.linalg.norm(normal)
    reference = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(reference, n)) > 0.95:
        reference = np.array([0.0, 1.0, 0.0])
    u = np.cross(reference, n)
    u = u / np.linalg.norm(u)
    v = np.cross(n, u)
    return u, v, n


class GiftiViewer(QWidget):
    def __init__(self, gifti_files,selectedFunc=0,shared_camera=None, parent=None, callbackSync=None,
                transducer_diameter=20.0, offset=0.0, additional_offset=0.0, roi_path=None):
        super().__init__(parent)
        self.transducer_diameter = transducer_diameter
        self.offset = offset
        self.additional_offset = additional_offset

        # Target centroid, for aiming the transducer glyph — matches
        # prepare_acoustic_simulation()'s vertex_vector, which always
        # aims at the target centroid now (previously used the raw skin
        # normal instead when there was a direct intersection, but that
        # isn't necessarily aimed at the centroid). Computed once here
        # rather than per-pick since it doesn't depend on the vertex.
        self.target_center = None
        if roi_path is not None:
            try:
                from PlanTUS import roi_center_of_gravity
                self.target_center = np.asarray(roi_center_of_gravity(roi_path), dtype=float)
            except Exception:
                self.target_center = None

        # --- Qt Layout ---
        layout = QVBoxLayout()
        self.setLayout(layout)

        # VTK Widget
        self.vtkWidget = QVTKRenderWindowInteractor(self)

        self.titleLabel = QComboBox(self)
        self.titleLabel.addItems([g[4] for g in gifti_files])
        self.titleLabel.setCurrentIndex(selectedFunc)
        self.titleLabel.currentIndexChanged.connect(self.select_function)

        # Apply stylesheet
        self.titleLabel.setStyleSheet("""
            QComboBox {
                font-size: 14px;       /* Change font size */
                color: green;            /* Change text color */
            }
        """)
        layout.addWidget(self.titleLabel,alignment=Qt.AlignCenter)
        layout.addWidget(self.vtkWidget)

        self.valueLabel = QLabel("Value: N/A")
        layout.addWidget(self.valueLabel)

        self.renderer = vtk.vtkRenderer()

        self.selectedFunc=selectedFunc

        self.Entries=[]

        self.func_data = []
        self.faces = []

        self.currentHeatmapVisibility = True

        for g in gifti_files:
            entry={}

        # --- Load GIFTI File ---
            gii = nib.load(g[0])
            coords = gii.darrays[0].data  # vertex coordinates (Nx3)
            faces = gii.darrays[1].data   # triangles (Mx3)

            entry['coords'] = coords
            coordsOrig = coords.copy()
            entry['coordsOrig'] = coordsOrig
            entry['faces'] = faces
            entry['title'] = g[4] if len(g) > 4 else ""
            entry['colormap'] = g[5] if len(g) > 5 else "jet"

            # Outward-pointing per-vertex normals (same convention PlanTUS.py's
            # compute_surface_metrics() uses: trimesh vertex_normals, with
            # "-normal" treated as inward elsewhere in the codebase).
            entry['vertex_normals'] = trimesh.Trimesh(
                vertices=coordsOrig, faces=faces, process=False
            ).vertex_normals


            func = nib.load(g[1])
            func_data = func.darrays[0].data

            entry['func_data'] = func_data

            thresh = nib.load(g[2])
            thresh_data = thresh.darrays[0].data
            coords_outside_mask = coords.copy()
            coords_outside_mask[thresh_data==1, :] = np.nan  # remove vertices below threshold
            coords[thresh_data==0, :] = np.nan  # remove vertices below threshold

            scalars = func_data
            # --- Convert to VTK PolyData ---
            
            for n,c in enumerate([coords, coords_outside_mask,coordsOrig]):
                points = vtk.vtkPoints()
                for x, y, z in c:
                    points.InsertNextPoint(x, y, z)

                polys = vtk.vtkCellArray()
                for tri in faces:
                    polys.InsertNextCell(3)
                    polys.InsertCellPoint(int(tri[0]))
                    polys.InsertCellPoint(int(tri[1]))
                    polys.InsertCellPoint(int(tri[2]))

                polydata = vtk.vtkPolyData()
                polydata.SetPoints(points)
                polydata.SetPolys(polys)

                # --- Mapper + Actor ---
                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputData(polydata)

                actor = vtk.vtkActor()
                actor.SetMapper(mapper)
            
                if  n in [0,2]:
                    vtk_scalars = vtk.vtkFloatArray()
                    vtk_scalars.SetName("Heatmap")
                    for val in scalars:
                        vtk_scalars.InsertNextValue(float(val))

                    polydata.GetPointData().SetScalars(vtk_scalars)
                    if g[3] is None:
                        mapper.SetScalarRange(vtk_scalars.GetRange())
                    else:
                        mapper.SetScalarRange(g[3])
                    self.renderer.AddActor(actor)
                    if n==0:
                        entry['mapperMasked'] = mapper
                        entry['actorHeatMapMasked'] = actor
                    else:
                        entry['mapperUnmasked'] = mapper
                        entry['actorHeatMapUnmasked'] = actor
                else:
                    # If no scalars or the rest of the scalp, just give the actor a solid color
                    actor.GetProperty().SetColor(0.8, 0.8, 0.8)  # light gray
                    self.renderer.AddActor(actor)
                    entry['actorSkinMasked'] = actor

                actor.SetVisibility(False)

            self.Entries.append(entry)


        # --- Renderer ---
        self.select_function(self.selectedFunc)
        self.renderer.SetBackground(0.1, 0.1, 0.1)

        if shared_camera:
            self.renderer.SetActiveCamera(shared_camera)

        # --- Render Window ---
        self.vtkWidget.GetRenderWindow().AddRenderer(self.renderer)
        self.interactor = self.vtkWidget.GetRenderWindow().GetInteractor()

        # --- Better Interaction (like 3D Slicer) ---
        style = vtkInteractorStyleTrackballCamera()
        self.interactor.SetInteractorStyle(style)

        # Right-click picks a transducer placement — registered as a
        # plain interactor observer, the same proven mechanism the old
        # left-click-in-"Selection Mode" picking used (AddObserver, not
        # a subclassed interactor style; that alternative couldn't be
        # confirmed to actually fire from VTK's internal event dispatch
        # without a live display to test against, so this sticks to the
        # approach already known to work in this codebase). Right-click
        # still also triggers the style's own default zoom/dolly
        # gesture, but for a click without a drag that produces no
        # visible movement, so this shouldn't be noticeable in practice.
        self.interactor.AddObserver("RightButtonPressEvent", lambda obj, event: self.on_right_click())

        def zoom_callback(obj, event):
            camera = self.renderer.GetActiveCamera()
            if event == "MouseWheelForwardEvent":
                camera.Dolly(1.05)  # zoom in
            elif event == "MouseWheelBackwardEvent":
                camera.Dolly(0.95)  # zoom out
            self.renderer.ResetCameraClippingRange()
            if callbackSync:
                callbackSync(self, event)
            else:
                self.vtkWidget.GetRenderWindow().Render()

        self.interactor.AddObserver("MouseWheelForwardEvent", zoom_callback)
        self.interactor.AddObserver("MouseWheelBackwardEvent", zoom_callback)

        def keypress_callback(obj, event):
            key = obj.GetKeySym()
            camera = self.renderer.GetActiveCamera()
            if key == "plus" or key == "equal":  # "+" key
                camera.Dolly(1.1)
            elif key == "minus":
                camera.Dolly(0.9)
            self.renderer.ResetCameraClippingRange()
            if callbackSync:
                callbackSync(self, event)
            else:
                self.vtkWidget.GetRenderWindow().Render()

        self.interactor.AddObserver("KeyPressEvent", keypress_callback)

        self.interactor.Initialize()
        self.interactor.Start()

        # --- Picker ---
        self.picker = vtk.vtkCellPicker()
        self.picker.SetTolerance(0.001)
        # Restrict picking to the skin-surface actors only — otherwise a
        # right-click could hit a previously-placed transducer glyph (or
        # its rod, or an earlier placement marker) sitting in front of
        # the surface and pick garbage cell/vertex data from the wrong
        # mesh entirely.
        self.picker.InitializePickList()
        for entry in self.Entries:
            self.picker.AddPickList(entry['actorHeatMapMasked'])
            self.picker.AddPickList(entry['actorHeatMapUnmasked'])
            self.picker.AddPickList(entry['actorSkinMasked'])
        self.picker.PickFromListOn()

        # Placement markers — one new sphere per pick, kept visible
        # (not moved/reused) so earlier placements stay marked on the
        # surface when a new one is picked. Cleared explicitly via
        # clear_selection() (the "Remove Transducer" button), not
        # automatically on each new pick.
        self.placement_marker_actors = []

        # Transducer placement glyph — a generic cylinder approximating
        # the transducer body, plus a thin "rod" indicating the
        # handle/cable direction (see show_transducer() docstring for
        # exactly what this does and doesn't represent).
        cylinder_source = vtk.vtkCylinderSource()
        cylinder_source.SetRadius(1.0)
        cylinder_source.SetHeight(1.0)
        cylinder_source.SetResolution(32)
        self.transducer_glyph = _make_transformed_glyph(cylinder_source, (0.6, 0.6, 0.6), opacity=0.45)
        self.renderer.AddActor(self.transducer_glyph['actor'])
        self.transducer_glyph['actor'].SetVisibility(False)

        rod_source = vtk.vtkCylinderSource()
        rod_source.SetRadius(1.0)
        rod_source.SetHeight(1.0)
        rod_source.SetResolution(16)
        self.transducer_rod_glyph = _make_transformed_glyph(rod_source, (0.35, 0.35, 0.35), opacity=0.6)
        self.renderer.AddActor(self.transducer_rod_glyph['actor'])
        self.transducer_rod_glyph['actor'].SetVisibility(False)

        # Callback for broadcasting selection
        self.selection_callback = None

        axes = vtk.vtkAxesActor()
        axes.SetTotalLength(50, 50, 50)   # size of arrows
        axes.AxisLabelsOff()               # show X/Y/Z labels
        axes.SetCylinderRadius(0.1)
        axes.SetShaftTypeToCylinder()

        self.orientation_widget = vtk.vtkOrientationMarkerWidget()
        self.orientation_widget.SetOrientationMarker(axes)
        self.orientation_widget.SetInteractor(self.interactor)
        self.orientation_widget.SetViewport(0.0, 0.0, 0.2, 0.2)  # bottom-left corner
        self.orientation_widget.SetEnabled(1)
        self.orientation_widget.InteractiveOff()

        for entry in self.Entries:
            self.set_colormap(entry['colormap'], entry=entry)

        # --- Scalar bar ---
        # Configured directly on the actor (not via vtkScalarBarWidget's
        # representation) — the actor has its own default position
        # (0.82, 0.1) / size (0.17, 0.8), i.e. right side, tall and
        # narrow, which is what actually gets rendered; setting position
        # only on the widget's representation left the actor's own
        # default in place, which was the bug last time. No
        # interactivity (dragging) is needed here, so the widget is
        # dropped entirely rather than trying to keep both in sync.
        scalar_bar = vtk.vtkScalarBarActor()
        self.renderer.AddActor(scalar_bar)
        scalar_bar.SetLookupTable(self.current_ActorsEntry['mapperMasked'].GetLookupTable())
        scalar_bar.SetNumberOfLabels(5)
        scalar_bar.SetOrientationToHorizontal()
        scalar_bar.SetPosition(0.1, 0.02)   # lower-left corner, normalized viewport coords
        scalar_bar.SetPosition2(0.8, 0.08)  # width/height, normalized viewport coords

    def select_function(self,selection):
        #we first hide the current function's actors
        self.current_ActorsEntry['actorHeatMapMasked'].SetVisibility(False)
        self.current_ActorsEntry['actorHeatMapUnmasked'].SetVisibility(False)
        self.current_ActorsEntry['actorSkinMasked'].SetVisibility(False)
        self.selectedFunc=selection
        self.set_heatmap_visibility(self.currentHeatmapVisibility) #this will honor the current selection


    @property
    def current_ActorsEntry(self):
        return self.Entries[self.selectedFunc]

    def on_right_click(self):
        x, y = self.interactor.GetEventPosition()
        if self.picker.Pick(x, y, 0, self.renderer):
            cell_id = self.picker.GetCellId()
            if cell_id >= 0 and cell_id < len(self.current_ActorsEntry['faces']):
                # Get triangle vertices
                tri = self.current_ActorsEntry['faces'][cell_id]
                vtx_coords = self.current_ActorsEntry['coordsOrig'][tri]
                if np.any(np.isnan(vtx_coords)):
                    return  # skip if any vertex is NaN
                
                # Place sphere at picked point
                pick_pos = self.picker.GetPickPosition()
                vertex_normal = self.current_ActorsEntry['vertex_normals'][tri[0]]
                if self.selection_callback:
                    # Notify main window about selection
                    self.selection_callback(tri[0], cell_id, pick_pos, vertex_normal)

        return
    
    def highlight_triangle(self, cell_id,pick_pos):
        """Highlight a triangle by ID and add a persistent marker at the
        pick position — a new sphere each time, not a reused/repositioned
        one, so earlier placements stay visible on the surface."""

        tri = self.current_ActorsEntry['faces'][cell_id]
        values = self.current_ActorsEntry['func_data'][tri]
        value = np.mean(values)
        if np.isnan(value):
            return  # skip if value is NaN
        self.valueLabel.setText(f"Value: {value:.2f}")

        marker_source = vtk.vtkSphereSource()
        marker_source.SetRadius(2.0)
        marker_source.SetCenter(*pick_pos)
        marker_mapper = vtk.vtkPolyDataMapper()
        marker_mapper.SetInputConnection(marker_source.GetOutputPort())
        marker_actor = vtk.vtkActor()
        marker_actor.SetMapper(marker_mapper)
        marker_actor.GetProperty().SetColor(0.8, 0.8, 0.8)
        self.renderer.AddActor(marker_actor)
        self.placement_marker_actors.append(marker_actor)

        self.vtkWidget.GetRenderWindow().Render()

    def show_transducer(self, pick_pos, vertex_normal, roll_degrees=0.0, orientation_mode="target_centered"):
        """Show a generic cylinder + handle-rod glyph at the picked placement.

        Approximates any real/custom transducer_surface_model — uses
        native VTK primitives rather than loading/transforming the
        actual GIFTI model on each pick, so it stays responsive.

        orientation_mode mirrors PlanTUS.prepare_acoustic_simulation()'s
        own parameter of the same name exactly, so the live preview
        matches whichever mode will actually be used at generation time:
        - "target_centered" (default): beam axis aimed at the target
          ROI's centroid.
        - "vertex_normal": beam axis aimed along the local skin surface
          normal instead — not necessarily at the target.

        - the transducer's CENTER sits `additional_offset + offset`
          (additional_offset + plane_offset) mm from the skin surface,
          outward along the normal — matching
          prepare_acoustic_simulation()'s own `transducer_center_coordinates`
          formula exactly, so the preview's anchor and the actual
          generated placement agree.
        - the cylinder's height is `offset + additional_offset` too,
          matching the actual generic transducer model PlanTUS now
          always generates and exports (create_surface_transducer_model()
          is called with the same height) — so the live preview's size
          reflects the real exported model's dimensions, not an
          arbitrary fixed visual size.
        - the perpendicular (x, y) frame at the center, and how
          roll_degrees rotates it, comes from
          PlanTUS.deterministic_perpendicular_frame() applied to the
          *inward* direction — the exact same function and input
          PlanTUS.prepare_acoustic_simulation() uses to build the real
          position matrices, so a given roll angle here means the same
          physical orientation as what actually gets generated.
        - the thin "rod" glyph extends from the center along local +x,
          representing the handle/cable direction.
        """
        pick_pos = np.asarray(pick_pos, dtype=float)
        normal = np.asarray(vertex_normal, dtype=float)
        norm_len = np.linalg.norm(normal)
        if norm_len == 0:
            return
        normal = normal / norm_len

        if orientation_mode == "vertex_normal":
            # Aimed along the local skin normal — not necessarily at
            # the target centroid. Matches
            # prepare_acoustic_simulation()'s "vertex_normal" mode.
            inward = -normal
        elif self.target_center is not None:
            # "target_centered" (default): aim at the target centroid.
            # Matches prepare_acoustic_simulation()'s vertex_vector for
            # that mode. This affects both the beam axis AND the
            # standoff position below (the real formula computes
            # transducer_center_coordinates using this same vector, not
            # the raw skin normal), so both need to follow it, not just
            # orientation. Falls back to the skin normal if the target
            # centroid isn't available for any reason.
            to_target = self.target_center - pick_pos
            to_target_norm = np.linalg.norm(to_target)
            inward = to_target / to_target_norm if to_target_norm > 0 else -normal
        else:
            inward = -normal
        outward = -inward

        center = pick_pos - inward * (self.additional_offset + self.offset)

        try:
            from PlanTUS import deterministic_perpendicular_frame
            x_axis, y_axis = deterministic_perpendicular_frame(inward, roll_degrees)
        except Exception:
            x_axis, y_axis, _ = _trajectory_frame(outward)

        height = self.offset + self.additional_offset  # matches create_surface_transducer_model()'s height exactly
        _set_cylinder_pose(self.transducer_glyph, center, x_axis, y_axis, outward,
                           radius=self.transducer_diameter / 2.0, height=height)
        self.transducer_glyph['actor'].SetVisibility(True)

        rod_length = self.transducer_diameter / 2.0 + 15.0
        rod_radius = 1.5
        rod_center = center + x_axis * (rod_length / 2.0)
        _set_cylinder_pose(self.transducer_rod_glyph, rod_center, outward, y_axis, x_axis,
                           radius=rod_radius, height=rod_length)
        self.transducer_rod_glyph['actor'].SetVisibility(True)

        self.vtkWidget.GetRenderWindow().Render()

    def hide_transducer(self):
        self.transducer_glyph['actor'].SetVisibility(False)
        self.transducer_rod_glyph['actor'].SetVisibility(False)
        self.vtkWidget.GetRenderWindow().Render()

    def clear_selection(self):
        """Hide the transducer glyph only — accumulated placement
        markers on the head surface are left in place."""
        self.hide_transducer()

    def clear_placement_markers(self):
        """Remove all accumulated placement-marker spheres from the
        head surface (does not affect the transducer glyph)."""
        for actor in self.placement_marker_actors:
            self.renderer.RemoveActor(actor)
        self.placement_marker_actors = []
        self.vtkWidget.GetRenderWindow().Render()

    def reset_camera(self):
        self.renderer.ResetCamera()
        self.vtkWidget.GetRenderWindow().Render()

    def set_colormap(self, name, entry=None):
        entry = entry if entry is not None else self.current_ActorsEntry

        from matplotlib.pyplot import get_cmap

        cmap = get_cmap(name)  # get all colors
        lut = vtk.vtkLookupTable()
        lut.SetNumberOfTableValues(256)
        for i in range(256):
            val = cmap(i)
            lut.SetTableValue(i, val[0], val[1], val[2], 1.0)

        entry['mapperMasked'].SetLookupTable(lut)
        entry['mapperUnmasked'].SetLookupTable(lut)

        self.vtkWidget.GetRenderWindow().Render()

    def set_heatmap_visibility(self, visible):
        self.currentHeatmapVisibility = visible
        self.current_ActorsEntry['actorHeatMapMasked'].SetVisibility(visible)
        self.current_ActorsEntry['actorSkinMasked'].SetVisibility(visible)
        self.current_ActorsEntry['actorHeatMapUnmasked'].SetVisibility(not visible)
        self.vtkWidget.GetRenderWindow().Render()

class MultiGiftiViewerWidget(QWidget):
    def __init__(self, gifti_files, MaxViews=4, parent=None, callBackAfterGenTrajectory=None,
                t1_path=None, distances_metric_path=None,
                additional_offset=0.0, min_distance=0.0, max_distance=100.0,
                roi_path=None, roi_stl_path=None, focal_distance_list=None, flhm_list=None,
                transducer_diameter=20.0, offset=0.0, initial_vertex=None):
        super().__init__(parent)
        self.viewers = []
        self.MaxViews = MaxViews
        self.callBackAfterGenTrajectory = callBackAfterGenTrajectory

        layout = QVBoxLayout(self)
        self.setLayout(layout)

        # Toolbar
        self.toolbar = QToolBar("Controls", self)
        self.toolbar.setStyleSheet("""
            QToolButton {
                background-color: #f0f0f0;
                border: 1px solid #888888;
                border-radius: 4px;
                padding: 4px 10px;
                margin: 2px;
            }
            QToolButton:hover {
                background-color: #e0e0e0;
                border: 1px solid #555555;
            }
            QToolButton:pressed {
                background-color: #c8c8c8;
            }
        """)
        layout.addWidget(self.toolbar)

        self.markersToolbar = QToolBar("Markers", self)
        self.markersToolbar.setStyleSheet(self.toolbar.styleSheet())
        layout.addWidget(self.markersToolbar)

        instructions_label = QLabel("Right-click on the head surface to select a transducer placement.")
        instructions_label.setStyleSheet("QLabel { font-size: 13px; font-weight: bold; color: #2a7a2a; padding: 4px; }")
        instructions_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(instructions_label)

        # Horizontal layout for the viewers
        viewers_layout = QHBoxLayout()
        layout.addLayout(viewers_layout)

        # Shared camera
        shared_camera = vtk.vtkCamera()

        # --- Synchronize Rendering ---
        def sync_cameras(caller=None, event=None):
            for v in self.viewers:
                v.vtkWidget.GetRenderWindow().Render()

        # Create viewers
        for n in range(self.MaxViews):
            v = GiftiViewer(gifti_files,
                            shared_camera=shared_camera,
                            callbackSync=sync_cameras,
                            parent=self,
                            selectedFunc=n,
                            transducer_diameter=transducer_diameter,
                            offset=offset,
                            additional_offset=additional_offset,
                            roi_path=roi_path)
            viewers_layout.addWidget(v)
            self.viewers.append(v)

        # Rightmost panel: oblique volume view aligned to the picked
        # vertex's normal, with the estimated intracranial focus marked.
        self.volume_focus_viewer = None
        if t1_path is not None:
            self.volume_focus_viewer = VolumeFocusViewer(
                t1_path,
                distances_metric_path=distances_metric_path,
                additional_offset=additional_offset,
                min_distance=min_distance,
                max_distance=max_distance,
                roi_path=roi_path,
                roi_stl_path=roi_stl_path,
                focal_distance_list=focal_distance_list,
                flhm_list=flhm_list,
                parent=self,
            )
            viewers_layout.addWidget(self.volume_focus_viewer)

        # Attach observer to all interactors
        for v in self.viewers:
            v.interactor.AddObserver("InteractionEvent", sync_cameras)

        # Toolbar buttons
        remove_transducer_action = QAction("Remove Transducer Model", self)
        remove_transducer_action.triggered.connect(self.clear_selection)
        self.toolbar.addAction(remove_transducer_action)

        remove_markers_action = QAction("Remove Placement Markers", self)
        remove_markers_action.triggered.connect(self.clear_placement_markers)
        self.markersToolbar.addAction(remove_markers_action)

        # Match the two buttons' widths by widening the shorter one
        # ("Remove Transducer Model") to the longer one's natural width
        # — not shrinking "Remove Placement Markers" down to a fixed
        # guess, which is what happened last time.
        markers_button = self.markersToolbar.widgetForAction(remove_markers_action)
        target_width = markers_button.sizeHint().width()
        self.toolbar.widgetForAction(remove_transducer_action).setFixedWidth(target_width)

        self.toolbar.addWidget(QLabel("  Transducer rotation:"))
        self.transducer_roll_degrees = 0.0
        self.rollSlider = QSlider(Qt.Horizontal)
        self.rollSlider.setMinimum(0)
        self.rollSlider.setMaximum(359)
        self.rollSlider.setValue(0)
        self.rollSlider.setFixedWidth(150)
        self.rollSlider.valueChanged.connect(self.set_transducer_roll)
        self.toolbar.addWidget(self.rollSlider)
        self.rollLabel = QLabel("0°")
        self.toolbar.addWidget(self.rollLabel)

        # Orientation mode: how the beam axis is aimed at the picked
        # vertex — mirrors PlanTUS.prepare_acoustic_simulation()'s own
        # orientation_mode parameter exactly (same two string values),
        # so this choice determines both what's shown live here and
        # what actually gets generated on "Save Placement".
        self.toolbar.addWidget(QLabel("     Orientation:"))
        self.orientation_mode = "target_centered"  # default, per case 1
        self.orientationComboBox = QComboBox()
        self.orientationComboBox.addItem("Towards target center", "target_centered")
        self.orientationComboBox.addItem("Along surface normal", "vertex_normal")
        self.orientationComboBox.currentIndexChanged.connect(self.set_orientation_mode)
        self.toolbar.addWidget(self.orientationComboBox)

        self.toolbar.addWidget(QLabel("     "))
        self.heatmap_checkbox = QCheckBox("Show masked maps")
        self.heatmap_checkbox.setChecked(True)  # default ON
        self.heatmap_checkbox.toggled.connect(self.toggle_heatmap)
        self.toolbar.addWidget(self.heatmap_checkbox)

        self.toolbar.addWidget(QLabel("     Views:"))

        axial_action = QAction("Top", self)
        axial_action.triggered.connect(lambda: self.set_preset_view("top"))
        self.toolbar.addAction(axial_action)

        coronal_action = QAction("Front", self)
        coronal_action.triggered.connect(lambda: self.set_preset_view("front"))
        self.toolbar.addAction(coronal_action)

        lateral_right_action = QAction("Lateral Right", self)
        lateral_right_action.triggered.connect(lambda: self.set_preset_view("lateral_right"))
        self.toolbar.addAction(lateral_right_action)

        lateral_left_action = QAction("Lateral Left", self)
        lateral_left_action.triggered.connect(lambda: self.set_preset_view("lateral_left"))
        self.toolbar.addAction(lateral_left_action)

        oblique_right_action = QAction("Oblique Right", self)
        oblique_right_action.triggered.connect(lambda: self.set_preset_view("oblique_right"))
        self.toolbar.addAction(oblique_right_action)

        oblique_left_action = QAction("Oblique Left", self)
        oblique_left_action.triggered.connect(lambda: self.set_preset_view("oblique_left"))
        self.toolbar.addAction(oblique_left_action)

        # Align the Screenshot button's right edge with row 1's right
        # edge (which now ends at "Oblique Left", the last item added
        # above) — estimated from sizeHint()s, the same mechanism
        # already used above to match the two "Remove ..." buttons'
        # widths, since nothing is actually shown/laid out yet at
        # __init__ time to measure real pixel positions from.
        # markers_leading_width is measured BEFORE adding the
        # Screenshot action/spacer, so it only reflects "Remove
        # Placement Markers" (already width-matched above).
        row1_width = self.toolbar.sizeHint().width()
        markers_leading_width = self.markersToolbar.sizeHint().width()

        screenshot_action = QAction("Screenshot", self)
        screenshot_action.triggered.connect(self.save_screenshot)
        self.markersToolbar.addAction(screenshot_action)
        screenshot_width = self.markersToolbar.widgetForAction(screenshot_action).sizeHint().width()

        needed_spacer_width = max(0, row1_width - markers_leading_width - screenshot_width)
        spacer = QWidget()
        spacer.setFixedWidth(int(needed_spacer_width))
        self.markersToolbar.insertWidget(screenshot_action, spacer)

        # Hook up selection synchronization
        for v in self.viewers:
            v.selection_callback = self.broadcast_selection

        # Initial camera reset and preset
        if self.viewers:
            self.viewers[0].renderer.ResetCamera()
            for v in self.viewers:
                v.vtkWidget.GetRenderWindow().Render()

        self.set_preset_view("oblique_right")

        self.select_vertex = None
        self.current_pick_pos = None
        self.current_vertex_normal = None

        button = QPushButton("Save Placement", self)
        button.clicked.connect(self.GenerateTrajectory)
        button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)  # only as wide as needed
        button.setStyleSheet("""
            QPushButton {
                font-size: 16px;
                color: gray;
                background-color: #eeeeee;
                border: 1px solid #aaaaaa;
                border-radius: 5px;
                padding: 8px 36px;
            }
        """)
        button.setEnabled(False)  
        self.generateTrajectoryPushButton = button
        layout.addWidget(button, alignment=Qt.AlignHCenter) 

        citation_label = QLabel(
            'Please cite: Lueckel, M., Vijayakumar, S., &amp; Bergmann, T. O. (2025). '
            '<b>PlanTUS: A heuristic tool for prospective planning of transcranial '
            'ultrasound transducer placements</b>. Brain Stimulation: Basic, Translational, '
            'and Clinical Research in Neuromodulation, 18(5), 1563–1565. '
            '<a href="https://doi.org/10.1016/j.brs.2025.08.013">https://doi.org/10.1016/j.brs.2025.08.013</a>'
        )
        citation_label.setWordWrap(True)
        citation_label.setOpenExternalLinks(True)
        citation_label.setAlignment(Qt.AlignCenter)
        citation_label.setStyleSheet("QLabel { font-size: 11px; color: #555555; padding: 6px; }")
        layout.addWidget(citation_label)

        # Suggest an initial placement (e.g. the vertex with the best
        # composite metric score) by simulating the same selection a
        # click would produce — the user can still pick a different
        # vertex afterward exactly as normal.
        if initial_vertex is not None and self.viewers:
            entry = self.viewers[0].Entries[0]
            faces = entry['faces']
            containing = np.where((faces == initial_vertex).any(axis=1))[0]
            if len(containing) > 0:
                cell_id = int(containing[0])
                pick_pos = entry['coordsOrig'][initial_vertex]
                vertex_normal = entry['vertex_normals'][initial_vertex]
                self.broadcast_selection(initial_vertex, cell_id, pick_pos, vertex_normal)

    def GenerateTrajectory(self):
        print("Generating trajectory... for vertex id",self.select_vertex, "roll:", self.transducer_roll_degrees,
              "orientation_mode:", self.orientation_mode)
        if self.callBackAfterGenTrajectory:
            self.callBackAfterGenTrajectory(self.select_vertex, self.transducer_roll_degrees, self.orientation_mode)

    # --------------------------
    # Methods from MainWindow
    # --------------------------
    def toggle_heatmap(self, checked):
        for v in self.viewers:
            v.set_heatmap_visibility(checked)

    def clear_selection(self):
        """Remove the current transducer placement so another vertex can
        be picked — hides the transducer glyph in all four panels
        (accumulated placement markers on the head surface are left in
        place), clears the volume-view markers, resets the rotation
        slider, and disables "Save Placement" again until a new
        vertex is picked."""
        self.select_vertex = None
        self.current_pick_pos = None
        self.current_vertex_normal = None
        for v in self.viewers:
            v.clear_selection()
        if self.volume_focus_viewer is not None:
            self.volume_focus_viewer.clear()
        self.transducer_roll_degrees = 0.0
        self.rollSlider.blockSignals(True)
        self.rollSlider.setValue(0)
        self.rollSlider.blockSignals(False)
        self.rollLabel.setText("0°")
        self.orientation_mode = "target_centered"
        self.orientationComboBox.blockSignals(True)
        self.orientationComboBox.setCurrentIndex(0)
        self.orientationComboBox.blockSignals(False)
        self.generateTrajectoryPushButton.setEnabled(False)
        self.generateTrajectoryPushButton.setStyleSheet("""
            QPushButton {
                font-size: 16px;
                color: gray;
                background-color: #eeeeee;
                border: 1px solid #aaaaaa;
                border-radius: 5px;
                padding: 8px 36px;
            }
        """)

    def clear_placement_markers(self):
        """Remove all accumulated placement-marker dots from the head
        surface in all four panels. Does not affect the transducer
        glyph or the current selection/placement state."""
        for v in self.viewers:
            v.clear_placement_markers()

    def set_transducer_roll(self, value):
        """Called when the rotation slider moves — re-renders the
        transducer glyph at the current placement with the new roll, if
        a vertex is currently selected."""
        self.transducer_roll_degrees = float(value)
        self.rollLabel.setText(f"{value}°")
        if self.current_pick_pos is not None:
            for v in self.viewers:
                v.show_transducer(self.current_pick_pos, self.current_vertex_normal,
                                  self.transducer_roll_degrees, self.orientation_mode)

    def set_orientation_mode(self, index):
        """Called when the Orientation dropdown changes — re-renders the
        current placement (transducer glyph and volume/focus view) with
        the new orientation mode, if a vertex is currently selected."""
        self.orientation_mode = self.orientationComboBox.itemData(index)
        if self.current_pick_pos is not None:
            for v in self.viewers:
                v.show_transducer(self.current_pick_pos, self.current_vertex_normal,
                                  self.transducer_roll_degrees, self.orientation_mode)
            if self.volume_focus_viewer is not None:
                self.volume_focus_viewer.update_view(self.select_vertex, self.current_pick_pos,
                                                     self.current_vertex_normal, self.orientation_mode)

    def broadcast_selection(self, vertex, cell_id, pick_pos, vertex_normal):
        """Called when one viewer selects a triangle."""
        self.select_vertex = vertex
        self.current_pick_pos = pick_pos
        self.current_vertex_normal = vertex_normal
        for v in self.viewers:
            v.highlight_triangle(cell_id, pick_pos)
            v.show_transducer(pick_pos, vertex_normal, self.transducer_roll_degrees, self.orientation_mode)
        if self.volume_focus_viewer is not None:
            self.volume_focus_viewer.update_view(vertex, pick_pos, vertex_normal, self.orientation_mode)
        #once a valid triangle is selected, enable the button
        self.generateTrajectoryPushButton.setEnabled(True)
        self.generateTrajectoryPushButton.setStyleSheet("""
            QPushButton {
                font-size: 16px;
                color: blue;
                font-weight: bold;
                background-color: #eaf0ff;
                border: 1px solid #3355cc;
                border-radius: 5px;
                padding: 8px 36px;
            }
            QPushButton:hover {
                background-color: #d5e0ff;
            }
            QPushButton:pressed {
                background-color: #b8c8ff;
            }
        """)

    def save_screenshot(self):
        """Save a screenshot of the entire viewer window — all four
        panels, the volume-focus view, and the toolbar — as a single
        PNG.

        A plain QWidget.grab() alone isn't enough: VTK's render windows
        typically draw to a native OpenGL surface that bypasses Qt's
        paint/composite pipeline, so grab() comes back blank wherever a
        VTK view sits (confirmed — this is exactly what showed up in
        testing). So this grabs the native Qt chrome (toolbar, labels,
        dropdowns — all correctly captured by grab()) as the base image,
        then separately captures each VTK render window's actual
        content via _capture_vtk_widget_as_qimage() and draws it in at
        the right position on top.
        """
        filename, _ = QFileDialog.getSaveFileName(self, "Save Screenshot", "screenshot.png", "PNG Files (*.png)")
        if not filename:
            return

        base_pixmap = self.grab()

        vtk_widgets = [v.vtkWidget for v in self.viewers]
        if self.volume_focus_viewer is not None:
            vtk_widgets += [view['vtkWidget'] for view in self.volume_focus_viewer.views]

        painter = QPainter(base_pixmap)
        for widget in vtk_widgets:
            qimage = _capture_vtk_widget_as_qimage(widget)
            top_left = widget.mapTo(self, widget.rect().topLeft())
            painter.drawImage(top_left, qimage)
        painter.end()

        base_pixmap.save(filename, "PNG")

    def set_preset_view(self, preset):
        if not self.viewers:
            return

        camera = self.viewers[0].renderer.GetActiveCamera()

        bounds = self.viewers[0].current_ActorsEntry['actorHeatMapMasked'].GetBounds()
        center = [(bounds[0] + bounds[1]) / 2,
                  (bounds[2] + bounds[3]) / 2,
                  (bounds[4] + bounds[5]) / 2]

        camera.SetFocalPoint(center)

        if preset == "top":
            camera.SetPosition(center[0], center[1], center[2] +800)
            camera.SetViewUp(0, 1, 0)
        elif preset == "front":
            camera.SetPosition(center[0], center[1] + 800, center[2])
            camera.SetViewUp(0, 0, 1)
        elif preset == "lateral_right":
            camera.SetPosition(center[0] + 800, center[1], center[2])
            camera.SetViewUp(0, 0, 1)
        elif preset == "lateral_left":
            camera.SetPosition(center[0] - 800, center[1], center[2])
            camera.SetViewUp(0, 0, 1)
        elif preset == "oblique_right":
            camera.SetPosition(center[0] + 400,
                               center[1] + 400,
                               center[2] + 400)
            camera.SetViewUp(0, 0, 1)
        elif preset == "oblique_left":
            camera.SetPosition(center[0] - 400,
                               center[1] + 400,
                               center[2] + 400)
            camera.SetViewUp(0, 0, 1)

        # Recompute clipping range for the NEW camera position on every
        # panel's own renderer (they share one camera object, but each
        # has its own renderer). Previously this called ResetCamera()
        # BEFORE repositioning the camera, which set a clipping range
        # for the OLD position/distance — then threw away everything
        # ResetCamera() computed except that now-stale clipping range,
        # since position/focal-point/view-up were immediately
        # overwritten afterward. If the new preset's camera distance
        # differed enough from the old one, the object could end up
        # entirely outside the (stale) clipping range and simply not
        # render — reproduced and confirmed headlessly before this fix.
        for v in self.viewers:
            v.renderer.ResetCameraClippingRange()
            v.vtkWidget.GetRenderWindow().Render()


class FinalResultViewer(QWidget):
    def __init__(self, gifti_files, parent=None, callbackSync=None):
        super().__init__(parent)

        # --- Qt Layout ---
        layout = QVBoxLayout()
        self.setLayout(layout)

        # VTK Widget
        self.vtkWidget = QVTKRenderWindowInteractor(self)

        layout.addWidget(self.vtkWidget)


        self.renderer = vtk.vtkRenderer()

        self.Entries=[]

        self.func_data = []
        self.faces = []

        self.currentHeatmapVisibility = True

        # --- Load GIFTI File ---
        gii = nib.load(gifti_files[0])
        coordsHead = gii.darrays[0].data  # vertex coordinates (Nx3)
        facesHead = gii.darrays[1].data   # triangles (Mx3)

        gii = nib.load(gifti_files[1])
        coordsTx = gii.darrays[0].data  # vertex coordinates (Nx3)
        facesTx = gii.darrays[1].data   # triangles (Mx3)

        objects=[[coordsHead,facesHead],
                    [coordsTx,facesTx]]

        # --- Convert to VTK PolyData ---

        for n,e in enumerate(objects):
            coords, faces = e
            points = vtk.vtkPoints()
            for x, y, z in coords:
                points.InsertNextPoint(x, y, z)

            polys = vtk.vtkCellArray()
            for tri in faces:
                polys.InsertNextCell(3)
                polys.InsertCellPoint(int(tri[0]))
                polys.InsertCellPoint(int(tri[1]))
                polys.InsertCellPoint(int(tri[2]))

            polydata = vtk.vtkPolyData()
            polydata.SetPoints(points)
            polydata.SetPolys(polys)

            # --- Mapper + Actor ---
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(polydata)

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
        
        
            # If no scalars or the rest of the scalp, just give the actor a solid color
            if n==0:
                actor.GetProperty().SetColor(0.8, 0.8, 0.8)  # light gray
                self.ActorHead = actor
            else:
                actor.GetProperty().SetColor(0.4, 0.4, 1.0)  # blue
                self.ActorTx = actor
            self.renderer.AddActor(actor)

            actor.SetVisibility(True)

        # --- Renderer ---
        self.renderer.SetBackground(0.1, 0.1, 0.1)

     
        # --- Render Window ---
        self.vtkWidget.GetRenderWindow().AddRenderer(self.renderer)
        self.interactor = self.vtkWidget.GetRenderWindow().GetInteractor()

        # --- Better Interaction (like 3D Slicer) ---
        style = vtkInteractorStyleTrackballCamera()
        self.interactor.SetInteractorStyle(style)

        def zoom_callback(obj, event):
            camera = self.renderer.GetActiveCamera()
            if event == "MouseWheelForwardEvent":
                camera.Dolly(1.05)  # zoom in
            elif event == "MouseWheelBackwardEvent":
                camera.Dolly(0.95)  # zoom out
            self.renderer.ResetCameraClippingRange()
            if callbackSync:
                callbackSync(self, event)
            else:
                self.vtkWidget.GetRenderWindow().Render()

        self.interactor.AddObserver("MouseWheelForwardEvent", zoom_callback)
        self.interactor.AddObserver("MouseWheelBackwardEvent", zoom_callback)

        def keypress_callback(obj, event):
            key = obj.GetKeySym()
            camera = self.renderer.GetActiveCamera()
            if key == "plus" or key == "equal":  # "+" key
                camera.Dolly(1.1)
            elif key == "minus":
                camera.Dolly(0.9)
            self.renderer.ResetCameraClippingRange()
            if callbackSync:
                callbackSync(self, event)
            else:
                self.vtkWidget.GetRenderWindow().Render()

        self.interactor.AddObserver("KeyPressEvent", keypress_callback)

        self.interactor.Initialize()
        self.interactor.Start()
        self.reset_camera()




    def reset_camera(self):
        self.renderer.ResetCamera()
        self.vtkWidget.GetRenderWindow().Render()

        camera = self.renderer.GetActiveCamera()

        bounds = self.ActorHead.GetBounds()
        center = [(bounds[0] + bounds[1]) / 2,
                  (bounds[2] + bounds[3]) / 2,
                  (bounds[4] + bounds[5]) / 2]


        camera.SetPosition(center[0] + 400,
                            center[1] + 400,
                            center[2] + 400)
        camera.SetViewUp(0, 0, 1)


class OrthoSliceViewer(QWidget):
    """
    Three linked orthogonal views (Axial/Coronal/Sagittal) for a NIfTI volume.
    - Click in any view to move the crosshair: only the in-plane indices change.
    - Crosshairs update in all views.
    - Uses vtkNIFTIImageReader to preserve orientation (origin/spacing/direction).
    """
    def __init__(self, nifti_path: str, size, parent=None):
        super().__init__(parent)
        

        # ----- Read NIfTI (respects qform/sform) -----
        self.reader = vtk.vtkNIFTIImageReader()
        self.reader.SetFileName(nifti_path)
        self.reader.Update()
        self.image = self.reader.GetOutput()

        # Convenience: extent/origin/spacing/direction
        self.extent = self.image.GetExtent()        # (iMin,iMax, jMin,jMax, kMin,kMax)
        self.origin = np.array(self.image.GetOrigin(), dtype=float)
        self.spacing = np.array(self.image.GetSpacing(), dtype=float)

        # direction is a 3x3
        dir_mat = self.image.GetDirectionMatrix()
        self.direction = np.array([[dir_mat.GetElement(r, c) for c in range(3)] for r in range(3)], dtype=float)
        self.dir_inv = np.linalg.inv(self.direction)

        # Crosshair starts at volume center (indices)
        self.crosshair = [
            (self.extent[0] + self.extent[1]) // 2,  # i
            (self.extent[2] + self.extent[3]) // 2,  # j
            (self.extent[4] + self.extent[5]) // 2,  # k
        ]

        # ----- Build UI layout -----
        layout = QHBoxLayout(self)
        self.setLayout(layout)

        # Keep per-view data
        self.views = []  # list of dicts: {name, widget, renderer, mapper, actor, lines, orientation}

        # Create the three orthogonal views
        self._add_view(layout, name="Axial",    orientation="Z")  # k fixed, (i,j) vary
        self._add_view(layout, name="Coronal",  orientation="Y")  # j fixed, (i,k) vary
        self._add_view(layout, name="Sagittal", orientation="X")  # i fixed, (j,k) vary

        # Initial draw
        self._update_all()

    # -----------------------------
    # Helpers: IJK <-> World
    # -----------------------------
    def ijk_to_world(self, i, j, k):
        """Convert voxel indices (i,j,k) to world coordinates using origin/spacing/direction."""
        ijk_mm = np.array([i * self.spacing[0], j * self.spacing[1], k * self.spacing[2]], dtype=float)
        return (self.direction @ ijk_mm) + self.origin

    def world_to_ijk(self, x, y, z):
        """Convert world to voxel indices (floating)."""
        xyz = np.array([x, y, z], dtype=float)
        ijk_mm = self.dir_inv @ (xyz - self.origin)
        ijk = ijk_mm / self.spacing
        return ijk  # float indices

    # -----------------------------
    # View construction
    # -----------------------------
    def _add_view(self, parent_layout, name, orientation):
        # Mapper oriented to X/Y/Z
        mapper = vtk.vtkImageSliceMapper()
        mapper.SetInputConnection(self.reader.GetOutputPort())
        if orientation == "Z":
            mapper.SetOrientationToZ()
            slice_index = self.crosshair[2]
        elif orientation == "Y":
            mapper.SetOrientationToY()
            slice_index = self.crosshair[1]
        elif orientation == "X":
            mapper.SetOrientationToX()
            slice_index = self.crosshair[0]
        else:
            raise ValueError("orientation must be 'X','Y', or 'Z'.")

        mapper.SetSliceNumber(int(slice_index))

        # Image actor + contrast defaults
        actor = vtk.vtkImageSlice()
        actor.SetMapper(mapper)
        prop = actor.GetProperty()
        # Window/level from data range for a reasonable default
        lo, hi = self.image.GetScalarRange()
        prop.SetColorWindow(max(hi - lo, 1e-3))
        prop.SetColorLevel((hi + lo) * 0.5)

        # Renderer
        ren = vtk.vtkRenderer()
        ren.AddViewProp(actor)
        ren.SetBackground(0, 0, 0)

        # Crosshair (two lines)
        line_color = (1.0, 0.0, 0.0)
        lines = []
        for _ in range(2):
            line = vtk.vtkLineSource()
            mapper_line = vtk.vtkPolyDataMapper()
            mapper_line.SetInputConnection(line.GetOutputPort())
            actor_line = vtk.vtkActor()
            actor_line.SetMapper(mapper_line)
            actor_line.GetProperty().SetColor(*line_color)
            actor_line.GetProperty().SetLineWidth(1.5)
            ren.AddActor(actor_line)
            lines.append((line, actor_line))

        # Widget + interactor
        w = QVTKRenderWindowInteractor()
        w.GetRenderWindow().AddRenderer(ren)
        iren = w.GetRenderWindow().GetInteractor()
        style = vtk.vtkInteractorStyleImage()
        iren.SetInteractorStyle(style)

        # Picker (pick only this slice actor to get correct world coords)
        picker = vtk.vtkPropPicker()
        picker.PickFromListOn()
        picker.AddPickList(actor)

        # Click handler: update in-plane indices only
        def on_left_click(obj, evt, view_name=name, orient=orientation, pick=picker, renderer=ren):
            x, y = iren.GetEventPosition()
            if pick.Pick(x, y, 0, renderer):
                wx, wy, wz = pick.GetPickPosition()
                fi, fj, fk = self.world_to_ijk(wx, wy, wz)
                # Clamp to extent & round only the in-plane axes
                i, j, k = self.crosshair
                if orient == "Z":      # axial: (i,j) from click, keep k
                    i = int(np.clip(round(fi), self.extent[0], self.extent[1]))
                    j = int(np.clip(round(fj), self.extent[2], self.extent[3]))
                elif orient == "Y":    # coronal: (i,k) from click, keep j
                    i = int(np.clip(round(fi), self.extent[0], self.extent[1]))
                    k = int(np.clip(round(fk), self.extent[4], self.extent[5]))
                elif orient == "X":    # sagittal: (j,k) from click, keep i
                    j = int(np.clip(round(fj), self.extent[2], self.extent[3]))
                    k = int(np.clip(round(fk), self.extent[4], self.extent[5]))
                self.crosshair = [i, j, k]
                self._update_all()
            # obj.OnLeftButtonDown()  # preserve default pan/zoom behavior

        iren.AddObserver("LeftButtonPressEvent", on_left_click)
        iren.Initialize()

        if orientation == "Z":
            # View from superior (looking down -Z)
            ren.GetActiveCamera().SetFocalPoint(0, 0, 0)
            ren.GetActiveCamera().SetPosition(0, 0, 1)
            ren.GetActiveCamera().SetViewUp(0, 1, 0)

        elif orientation == "Y":
            # View from front (looking along -Y)
            ren.GetActiveCamera().SetFocalPoint(0, 0, 0)
            ren.GetActiveCamera().SetPosition(0, -1, 0)
            ren.GetActiveCamera().SetViewUp(0, 0, 1)

        elif orientation == "X":
            # View from left (looking along -X)
            ren.GetActiveCamera().SetFocalPoint(0, 0, 0)
            ren.GetActiveCamera().SetPosition(-1, 0, 0)
            ren.GetActiveCamera().SetViewUp(0, 0, 1)

        ren.ResetCamera()

        mainQ = QWidget()
        tlayout = QVBoxLayout()
        mainQ.setLayout(tlayout)
        secondQ = QWidget()
        slayout = QHBoxLayout()
        secondQ.setLayout(slayout)

        if orientation == "Z":
            al=QLabel("A")
            al.setStyleSheet("""
            QLabel {
                font-size: 18px;       /* Change font size */
                color: green;            /* Change text color */
                font-weight: bold;       /* Change font weight */
            }
            """)
            tlayout.addWidget(al,alignment=Qt.AlignCenter)
            ll=QLabel("L")
            ll.setStyleSheet("""
            QLabel {
                font-size: 18px;       /* Change font size */
                color: blue;            /* Change text color */
                font-weight: bold;       /* Change font weight */
            }
            """)
            slayout.addWidget(ll)
        elif orientation == "Y":
            sl=QLabel("S")
            sl.setStyleSheet("""
            QLabel {
                font-size: 18px;       /* Change font size */
                color: red;            /* Change text color */
                font-weight: bold;       /* Change font weight */
            }
            """)
            tlayout.addWidget(sl,alignment=Qt.AlignCenter)
            ll=QLabel("L")
            ll.setStyleSheet("""
            QLabel {
                font-size: 18px;       /* Change font size */
                color: blue;            /* Change text color */
                font-weight: bold;       /* Change font weight */
            }
            """)
            slayout.addWidget(ll)
        else:
            sl=QLabel("S")
            sl.setStyleSheet("""
            QLabel {
                font-size: 18px;       /* Change font size */
                color: red;            /* Change text color */
                font-weight: bold;       /* Change font weight */
            }
            """)
            tlayout.addWidget(sl,alignment=Qt.AlignCenter)
            al=QLabel("A")
            al.setStyleSheet("""
            QLabel {
                font-size: 18px;       /* Change font size */
                color: green;            /* Change text color */
                font-weight: bold;       /* Change font weight */
            }
            """)
            slayout.addWidget(al)

        tlayout.addWidget(w)
        slayout.addWidget(mainQ)
        parent_layout.addWidget(secondQ)
        self.views.append(dict(
            name=name, orientation=orientation,
            widget=w, renderer=ren, mapper=mapper, actor=actor, lines=lines
        ))

        # w.GetRenderWindow().Render()

        # def add_orientation_labels(vtk_widget,orientation):
        # # Helper to create a text actor
        #     def make_label(text, x, y):
        #         actor = vtk.vtkTextActor()
        #         actor.SetInput(text)
        #         actor.GetTextProperty().SetFontSize(24)
        #         actor.GetTextProperty().SetColor(1, 1, 1)  # white text
        #         actor.GetTextProperty().SetBold(True)
        #         actor.SetDisplayPosition(x, y)
        #         return actor

        #     # Get render window size
        #     iren.Initialize()
        #     size = vtk_widget.GetRenderWindow().GetSize()
        #     dpr = vtk_widget.devicePixelRatioF()
        #     w = int(vtk_widget.width()*dpr)
        #     h = int(vtk_widget.height()*dpr)
        #     # w = int(size[0] )
        #     # h = int(size[1] )


        #     if orientation == "Z":
        #         # Axial: L/R (X), P/A (Y)
        #         ren.AddActor2D(make_label("L", int(w/100*5), h//2))
        #         ren.AddActor2D(make_label("R", w-int(w/10)*4, h//2))
        #         ren.AddActor2D(make_label("P", w//2, 10))
        #         ren.AddActor2D(make_label("A", w//2, h-10))

        #     elif orientation == "Y":
        #         # Coronal: L/R (X), S/I (Z)
        #         ren.AddActor2D(make_label("L", 10, h//2))
        #         ren.AddActor2D(make_label("R", w-100, h//2))
        #         ren.AddActor2D(make_label("S", w//2, h-10))
        #         ren.AddActor2D(make_label("I", w//2, 10))

        #     elif orientation == "X":
        #         # Sagittal: P/A (Y), S/I (Z)
        #         ren.AddActor2D(make_label("P", 10, h//2))
        #         ren.AddActor2D(make_label("A", w-10, h//2))
        #         ren.AddActor2D(make_label("S", w//2, h-10))
        #         ren.AddActor2D(make_label("I", w//2, 10))

        #     vtk_widget.GetRenderWindow().Render()
        # add_orientation_labels(w, orientation)


    # -----------------------------
    # Update all view slices + crosshairs
    # -----------------------------
    def _update_all(self):
        i, j, k = self.crosshair
        ei0, ei1, ej0, ej1, ek0, ek1 = self.extent

        for view in self.views:
            orient = view["orientation"]
            mapper = view["mapper"]
            lines = view["lines"]

            # Set the slice number appropriate for this view
            if orient == "Z":
                mapper.SetSliceNumber(int(k))
            elif orient == "Y":
                mapper.SetSliceNumber(int(j))
            elif orient == "X":
                mapper.SetSliceNumber(int(i))

            # Update crosshair lines (world coords via ijk_to_world)
            # Each view plane has two in-plane axes; draw lines across full extent.

            if orient == "Z":
                # Plane: k fixed; in-plane axes: i, j
                p1 = self.ijk_to_world(ei0, j,  k)
                p2 = self.ijk_to_world(ei1, j,  k)
                p3 = self.ijk_to_world(i,  ej0, k)
                p4 = self.ijk_to_world(i,  ej1, k)

            elif orient == "Y":
                # Plane: j fixed; in-plane axes: i, k
                p1 = self.ijk_to_world(ei0, j,  k)
                p2 = self.ijk_to_world(ei1, j,  k)
                p3 = self.ijk_to_world(i,  j,  ek0)
                p4 = self.ijk_to_world(i,  j,  ek1)

            elif orient == "X":
                # Plane: i fixed; in-plane axes: j, k
                p1 = self.ijk_to_world(i,  ej0, k)
                p2 = self.ijk_to_world(i,  ej1, k)
                p3 = self.ijk_to_world(i,  j,  ek0)
                p4 = self.ijk_to_world(i,  j,  ek1)

            # Apply to the two lines
            lineA, _ = lines[0]
            lineB, _ = lines[1]
            lineA.SetPoint1(*p1); lineA.SetPoint2(*p2)
            lineB.SetPoint1(*p3); lineB.SetPoint2(*p4)

            # Render this view
            view["widget"].GetRenderWindow().Render()

class VolumeFocusViewer(QWidget):
    """
    Two linked oblique volume views through a picked surface vertex,
    aligned to its normal — the two planes contain the trajectory axis,
    at 90 degrees to each other, so depth along it is visible from two
    independent angles. Both planes pass through the picked point and
    share the estimated intracranial focus marker.

    Camera framing is independent of the pick: each view is always
    centered on the volume's own center and scaled to show the entire
    volume (fixed once at load), so picking a vertex changes which
    oblique slice is shown, not how the view is zoomed/panned.

    The focus estimate reuses the same focal-distance formula
    PlanTUS.prepare_acoustic_simulation() applies at trajectory-generation
    time (per-vertex target distance + additional_offset, clamped to
    [min_distance, max_distance]), including both branches: if
    roi_stl_path is given, a real ray/ROI-mesh intersection is cast live
    for the picked vertex (via PlanTUS.compute_vector_mesh_intersections,
    the exact same function prepare_acoustic_simulation() uses) to check
    for a direct line of sight to the target, matching generation exactly
    rather than only ever approximating it. Falls back to the
    distance-to-centroid estimate (as before) if roi_stl_path isn't
    given or the intersection test fails for any reason.
    """

    # (name, role) — "inline" planes contain the trajectory axis, at 90
    # degrees to each other, so depth along it is visible from two angles.
    _VIEW_SPECS = [
        ("In-line View 1", "inline"),
        ("In-line View 2", "inline"),
    ]

    def __init__(self, t1_path, distances_metric_path=None,
                additional_offset=0.0, min_distance=0.0, max_distance=100.0,
                roi_path=None, roi_stl_path=None, focal_distance_list=None, flhm_list=None,
                parent=None):
        super().__init__(parent)
        self.additional_offset = additional_offset
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.roi_stl_path = roi_stl_path
        self.focal_distance_list = focal_distance_list
        self.flhm_list = flhm_list

        # Target centroid, for aiming the oblique planes/focus ellipsoid
        # — matches prepare_acoustic_simulation()'s vertex_vector, which
        # always aims at the target centroid now (previously used the
        # raw skin normal instead when there was a direct intersection).
        self.target_center = None
        if roi_path is not None:
            try:
                from PlanTUS import roi_center_of_gravity
                self.target_center = np.asarray(roi_center_of_gravity(roi_path), dtype=float)
            except Exception:
                self.target_center = None

        self.distances = None
        if distances_metric_path:
            gii = nib.load(distances_metric_path)
            self.distances = np.asarray(gii.darrays[0].data, dtype=float)

        layout = QVBoxLayout(self)
        self.setLayout(layout)

        # --- Load volume (shared reader across all views) ---
        self.reader = vtk.vtkNIFTIImageReader()
        self.reader.SetFileName(t1_path)
        self.reader.Update()
        image = self.reader.GetOutput()
        self.scalar_range = image.GetScalarRange()

        # Fixed framing target: the volume's own center and a radius
        # (half the bounding-box diagonal) large enough to always show
        # the whole volume under parallel projection, regardless of
        # where the reslice plane cuts through it. zoom_factor scales
        # this down for a tighter default view — under parallel
        # projection, apparent zoom is governed entirely by
        # ParallelScale, not camera distance, so only this needs to
        # change (cam_distance below just needs to stay far enough away
        # to avoid clipping, independent of zoom level).
        xmin, xmax, ymin, ymax, zmin, zmax = image.GetBounds()
        self.volume_center = np.array([(xmin + xmax) / 2.0, (ymin + ymax) / 2.0, (zmin + zmax) / 2.0])
        self.volume_half_diagonal = 0.5 * np.linalg.norm(
            [xmax - xmin, ymax - ymin, zmax - zmin])
        self.zoom_factor = 0.4

        # --- Target ROI overlay (opaque green wherever the mask is set,
        # fully transparent elsewhere, via a 2-entry lookup table) ---
        self.roi_reader = None
        self.roi_lut = None
        if roi_path:
            self.roi_reader = vtk.vtkNIFTIImageReader()
            self.roi_reader.SetFileName(roi_path)
            self.roi_reader.Update()

            self.roi_lut = vtk.vtkLookupTable()
            self.roi_lut.SetNumberOfTableValues(2)
            self.roi_lut.SetTableRange(0, 1)
            self.roi_lut.SetTableValue(0, 0.0, 0.0, 0.0, 0.0)  # background: transparent
            self.roi_lut.SetTableValue(1, 0.0, 1.0, 0.0, 1.0)  # ROI: opaque green
            self.roi_lut.Build()

        self.views = []  # one dict per plane, built by _add_view
        for name, role in self._VIEW_SPECS:
            self._add_view(layout, name, role)

        # Placed below the views, matching GiftiViewer's own
        # title-above/value-below layout convention for its panels.
        self.infoLabel = QLabel("Pick a vertex to preview the focus.")
        layout.addWidget(self.infoLabel)

    def _add_view(self, parent_layout, name, role):
        """Build one reformatted-plane row: label + oblique reslice view."""
        label = QLabel(name)
        label.setStyleSheet("QLabel { font-size: 13px; color: green; }")
        parent_layout.addWidget(label, alignment=Qt.AlignCenter)

        vtkWidget = QVTKRenderWindowInteractor(self)
        parent_layout.addWidget(vtkWidget)

        # --- Oblique reslice plane (point + normal), via the purpose-built
        # vtkImageResliceMapper/vtkPlane pair rather than a hand-built
        # reslice-axes matrix ---
        plane = vtk.vtkPlane()
        plane.SetOrigin(0, 0, 0)
        plane.SetNormal(0, 0, 1)

        reslice_mapper = vtk.vtkImageResliceMapper()
        reslice_mapper.SetInputConnection(self.reader.GetOutputPort())
        reslice_mapper.SetSlicePlane(plane)
        reslice_mapper.SetSliceFacesCamera(False)
        reslice_mapper.SetSliceAtFocalPoint(False)

        image_slice = vtk.vtkImageSlice()
        image_slice.SetMapper(reslice_mapper)
        prop = image_slice.GetProperty()
        lo, hi = self.scalar_range
        prop.SetColorWindow(max(hi - lo, 1e-3))
        prop.SetColorLevel((hi + lo) * 0.5)

        renderer = vtk.vtkRenderer()
        renderer.AddViewProp(image_slice)
        renderer.SetBackground(0, 0, 0)

        # --- Target ROI overlay: same plane as the T1, nearest-neighbor
        # resampling (so the binary mask isn't blurred into fractional
        # values a 2-entry LUT can't represent), opaque green via roi_lut.
        roi_image_slice = None
        if self.roi_reader is not None:
            roi_interpolator = vtk.vtkImageInterpolator()
            roi_interpolator.SetInterpolationModeToNearest()

            roi_reslice_mapper = vtk.vtkImageResliceMapper()
            roi_reslice_mapper.SetInputConnection(self.roi_reader.GetOutputPort())
            roi_reslice_mapper.SetSlicePlane(plane)  # same plane object as the T1 slice
            roi_reslice_mapper.SetSliceFacesCamera(False)
            roi_reslice_mapper.SetSliceAtFocalPoint(False)
            roi_reslice_mapper.SetInterpolator(roi_interpolator)

            roi_image_slice = vtk.vtkImageSlice()
            roi_image_slice.SetMapper(roi_reslice_mapper)
            roi_prop = roi_image_slice.GetProperty()
            roi_prop.SetLookupTable(self.roi_lut)
            roi_prop.SetUseLookupTableScalarRange(True)
            renderer.AddViewProp(roi_image_slice)

        # Picked skin point marker (yellow sphere) — separate actor
        # instances per view (a VTK actor can't be shared across renderers).
        pick_sphere = self._make_sphere((0.9, 0.9, 0.2), radius=2.0)
        renderer.AddActor(pick_sphere['actor'])
        pick_sphere['actor'].SetVisibility(False)

        # Estimated-focus marker: the actual ellipsoid glyph (same
        # a/b/c radii convention as PlanTUS.create_surface_ellipsoid),
        # oriented to the trajectory and positioned via a single 4x4
        # matrix updated in place on every pick.
        sphere_source = vtk.vtkSphereSource()
        sphere_source.SetRadius(1.0)
        sphere_source.SetThetaResolution(32)
        sphere_source.SetPhiResolution(32)
        focus_ellipsoid = _make_transformed_glyph(sphere_source, (1.0, 0.15, 0.15), opacity=0.6)
        renderer.AddActor(focus_ellipsoid['actor'])
        focus_ellipsoid['actor'].SetVisibility(False)

        vtkWidget.GetRenderWindow().AddRenderer(renderer)
        interactor = vtkWidget.GetRenderWindow().GetInteractor()
        style = vtkInteractorStyleTrackballCamera()
        interactor.SetInteractorStyle(style)
        interactor.Initialize()
        interactor.Start()

        self.views.append({
            'name': name, 'role': role,
            'vtkWidget': vtkWidget, 'renderer': renderer,
            'plane': plane, 'pick_sphere': pick_sphere, 'focus_ellipsoid': focus_ellipsoid,
        })

    @staticmethod
    def _make_sphere(color, radius):
        source = vtk.vtkSphereSource()
        source.SetRadius(radius)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(source.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*color)
        return {'source': source, 'mapper': mapper, 'actor': actor}

    def clear(self):
        """Hide the pick/focus markers and reset the info label — called
        when the transducer placement is removed so another vertex can
        be picked."""
        for view in self.views:
            view['pick_sphere']['actor'].SetVisibility(False)
            view['focus_ellipsoid']['actor'].SetVisibility(False)
            view['vtkWidget'].GetRenderWindow().Render()
        self.infoLabel.setText("Pick a vertex to preview the focus.")

    def estimate_focal_distance(self, vertex_index, pick_pos=None, vertex_normal=None):
        """Focal distance (mm) for a vertex — matches
        prepare_acoustic_simulation()'s own two-branch computation
        exactly when roi_stl_path/pick_pos/vertex_normal are available:
        casts the same ray-mesh intersection test live for this vertex;
        falls back to the distance-to-centroid estimate (using the
        precomputed distances_skin metric) if that's not possible or
        fails for any reason — see class docstring.

        Sets self._last_focal_distance_exact (bool) so callers can tell
        which branch was actually used, e.g. for a status label.
        """
        self._last_focal_distance_exact = False
        if self.distances is None or vertex_index is None or vertex_index >= len(self.distances):
            return None

        fallback = float(self.distances[vertex_index]) + self.additional_offset

        if self.roi_stl_path is not None and pick_pos is not None and vertex_normal is not None:
            try:
                from PlanTUS import compute_vector_mesh_intersections
                hits = compute_vector_mesh_intersections(
                    [np.asarray(pick_pos, dtype=float)],
                    [np.asarray(vertex_normal, dtype=float)],
                    self.roi_stl_path, 200)[0]
                if len(hits) > 1:
                    inter_center = (np.asarray(hits[0]) + np.asarray(hits[1])) / 2.0
                    raw = float(np.linalg.norm(np.asarray(pick_pos) - inter_center)) + self.additional_offset
                    self._last_focal_distance_exact = True
                    return float(np.clip(raw, self.min_distance, self.max_distance))
            except Exception:
                pass  # fall through to the distance-based estimate below

        return float(np.clip(fallback, self.min_distance, self.max_distance))

    def estimate_FLHM(self, focal_distance):
        """FLHM (mm, ellipsoid long-axis length) for a given focal
        distance, via the same calibration curve
        PlanTUS.compute_FLHM_for_focal_distance() uses at trajectory-
        generation time (same function, same inputs — not an
        approximation of it). Lazily imported (not at module load) to
        avoid a module-level circular import with PlanTUS.py, which
        itself lazily imports from this module inside a function for
        the same reason. Falls back to a fixed default if the lists
        aren't available or the import fails, so the live preview stays
        usable even without exact calibration data.

        Sets self._last_flhm_extrapolated (bool): True if
        focal_distance falls outside the range actually covered by
        focal_distance_list — compute_FLHM_for_focal_distance fits an
        unconstrained cubic through the calibration points, which can
        grow arbitrarily large outside the range it was fitted on, so
        a "too long" focus at generation time (not just in this
        preview — the real code has the same behavior) is often this,
        not a rendering bug.
        """
        self._last_flhm_extrapolated = False
        if self.focal_distance_list and self.flhm_list:
            if len(self.focal_distance_list) > 1:
                lo, hi = min(self.focal_distance_list), max(self.focal_distance_list)
                if focal_distance < lo or focal_distance > hi:
                    self._last_flhm_extrapolated = True
            try:
                from PlanTUS import compute_FLHM_for_focal_distance
                return float(compute_FLHM_for_focal_distance(
                    focal_distance, self.focal_distance_list, self.flhm_list))
            except Exception:
                pass
        return 10.0  # fallback default (mm) if calibration data isn't available

    def update_view(self, vertex_index, pick_pos, vertex_normal, orientation_mode="target_centered"):
        pick_pos = np.asarray(pick_pos, dtype=float)
        normal = np.asarray(vertex_normal, dtype=float)
        norm_len = np.linalg.norm(normal)
        if norm_len == 0:
            return
        normal = normal / norm_len

        # orientation_mode mirrors PlanTUS.prepare_acoustic_simulation()'s
        # own parameter of the same name — see GiftiViewer.show_transducer
        # for the full rationale on both modes.
        if orientation_mode == "vertex_normal":
            inward = -normal
        elif self.target_center is not None:
            to_target = self.target_center - pick_pos
            to_target_norm = np.linalg.norm(to_target)
            inward = to_target / to_target_norm if to_target_norm > 0 else -normal
        else:
            inward = -normal
        outward = -inward
        u, v, n = _trajectory_frame(outward)

        # Estimated intracranial focus, inward along the (now
        # target-aimed) direction above. Depth is
        # (focal_distance - additional_offset), not focal_distance
        # alone: the real generation path builds this position starting
        # from the transducer's own center (already offset +
        # additional_offset mm outward) and moves inward by
        # (offset + focal_distance) — the two "offset" terms cancel
        # algebraically there. Note estimate_focal_distance() below still
        # ray-casts along the raw skin normal (vertex_normal) for the
        # intersection test — that matches prepare_acoustic_simulation()
        # exactly too, which also always uses skin_normals (not
        # vertex_vector) for that specific ray-cast.
        focal_distance = self.estimate_focal_distance(vertex_index, pick_pos, normal)
        focus_pos = (pick_pos + inward * (focal_distance - self.additional_offset)
                    if focal_distance is not None else None)

        if focal_distance is not None:
            FLHM = self.estimate_FLHM(focal_distance)
            precision_note = "exact" if getattr(self, '_last_focal_distance_exact', False) else "approximate"
            flhm_note = " [FLHM EXTRAPOLATED beyond calibration range!]" if getattr(self, '_last_flhm_extrapolated', False) else ""
            self.infoLabel.setText(
                f"Focal distance: {focal_distance:.1f} mm, FLHM: {FLHM:.1f} mm ({precision_note}){flhm_note}"
            )
        else:
            FLHM = None
            self.infoLabel.setText("Estimated focal distance: N/A (no distance metric loaded)")

        # Ellipsoid radii: matches PlanTUS.prepare_acoustic_simulation's
        # create_surface_ellipsoid(FLHM, 5, ...) call — long axis (c)
        # along the trajectory is FLHM/2, short axes (a, b) are a fixed
        # 5mm-width/2 (not derived from the transducer diameter; this is
        # the same convention used for the final trajectory's own focus
        # ellipsoid, kept identical here for consistency).
        a = b = 5.0 / 2.0
        c = FLHM / 2.0 if FLHM is not None else None

        inline_normals = iter([u, v])

        for view in self.views:
            plane_normal = next(inline_normals)

            # Both planes pass through the picked point.
            view['plane'].SetOrigin(*pick_pos)
            view['plane'].SetNormal(*plane_normal)

            view['pick_sphere']['source'].SetCenter(*pick_pos)
            view['pick_sphere']['actor'].SetVisibility(True)

            if focus_pos is not None and c is not None:
                _set_glyph_pose(view['focus_ellipsoid'], focus_pos, u, v, n, a, b, c)
                view['focus_ellipsoid']['actor'].SetVisibility(True)
            else:
                view['focus_ellipsoid']['actor'].SetVisibility(False)

            # Camera looks straight down this view's own plane normal, but
            # is always centered on the volume's own center and scaled to
            # show the whole volume (fixed at load, in __init__) — picking
            # a vertex changes which oblique slice is shown, not the
            # framing. Parallel projection keeps that framing exact
            # regardless of viewing angle.
            camera = view['renderer'].GetActiveCamera()
            camera.SetParallelProjection(True)
            camera.SetParallelScale(self.volume_half_diagonal * self.zoom_factor)
            cam_distance = self.volume_half_diagonal * 2.0
            camera.SetFocalPoint(*self.volume_center)
            camera.SetPosition(*(self.volume_center + plane_normal * cam_distance))
            camera.SetViewUp(*n)
            view['renderer'].ResetCameraClippingRange()

            view['vtkWidget'].GetRenderWindow().Render()


def PrepareShowResults(skin_surf,distances_skin,distances_skin_thresholded,
                    target_intersection_skin,angles_skin,skin_skull_angles_skin,
                    CallBackGenerateTrajectory=None,
                    t1_path=None, additional_offset=0.0, min_distance=0.0, max_distance=100.0,
                    roi_path=None, roi_stl_path=None, focal_distance_list=None, flhm_list=None,
                    transducer_diameter=20.0, offset=0.0, initial_vertex=None):

    # Replace with path to your GIFTI file
    gifti_files = []
    gifti_files.append((skin_surf,
                        distances_skin,
                        distances_skin_thresholded,
                        None,
                        'Target Distance'))
    gifti_files.append((skin_surf,
                        target_intersection_skin,
                        distances_skin_thresholded,
                        None,
                        'Target Intersection'))
    gifti_files.append((skin_surf,
                        angles_skin,
                        distances_skin_thresholded,
                        [0,20],
                        'Transducer Tilt'))
    gifti_files.append((skin_surf,
                        skin_skull_angles_skin,
                        distances_skin_thresholded,
                        [0,20],
                        'Skin-Skull Angle'))

    widget = MultiGiftiViewerWidget(gifti_files,MaxViews=4,
                                    callBackAfterGenTrajectory=CallBackGenerateTrajectory,
                                    t1_path=t1_path,
                                    distances_metric_path=distances_skin,
                                    additional_offset=additional_offset,
                                    min_distance=min_distance,
                                    max_distance=max_distance,
                                    roi_path=roi_path,
                                    roi_stl_path=roi_stl_path,
                                    focal_distance_list=focal_distance_list,
                                    flhm_list=flhm_list,
                                    transducer_diameter=transducer_diameter,
                                    offset=offset,
                                    initial_vertex=initial_vertex)
    widget.resize(1700, 600)
    return widget
