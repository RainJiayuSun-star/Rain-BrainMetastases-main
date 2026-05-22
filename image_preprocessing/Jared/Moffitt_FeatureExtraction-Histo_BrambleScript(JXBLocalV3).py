"""
Whole-slide HistomicsTK feature extraction with fast settings.

NO patches.
NO tiles.
NO ROI subsets unless a GeoJSON ROI is paired with the slide.

Each .svs slide is loaded as one whole-slide image at 15x magnification.
One Excel row is produced per slide.
PatientID is repeated for each slide.

GeoJSON ROI behavior:
- If a slide at <patient>/<name>.svs has a sibling <patient>/<name>.geojson,
  only the bounding box of the GeoJSON polygons is loaded from disk and only
  pixels inside the polygons (minus any holes) are used for feature extraction.
- If no .geojson is present, the whole slide is loaded as before.
- GeoJSON coordinates are assumed to be in base (level-0) slide pixels
  (the QuPath default). They are rescaled to 15x using the slide's reported
  base magnification.

Tissue-edge cleanup:
- Segmentation runs first on the full Tissue_Mask, including poorly-behaved
  pixels along tissue/background borders.
- An intermediate segmentation overlay is saved with the "_Intermediate" suffix
  so you can visually compare what segmentation saw vs. what features were
  extracted from.
- The background mask (~Tissue_Mask) is morphologically dilated by
  Tissue_Edge_Buffer_Pixels (default 3). Any nucleus label that touches the
  dilated buffer is removed in full (whole-label, not pixel-clipped) so that
  feature extraction sees only the well-behaved interior nuclei.

Fast mode:
- 15x magnification
- No stain normalization
- Morphometry + intensity features only
- No FSD
- No Haralick
- No gradient features
- Saves segmentation overlay only (intermediate + final)
- Saves Excel after every completed slide (atomic write)
- Resumes from existing Excel: any (PatientID, Slide) already present is skipped
"""
###
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
###

import os
import json
import time
import numpy as np

###
from numpy.exceptions import RankWarning
warnings.filterwarnings("ignore", category=RankWarning)
###

import pandas as pd
import natsort as ns
import large_image
import scipy as sp
import matplotlib.pyplot as plt
import gc

import histomicstk as htk

#
# from skimage.segmentation import mark_boundaries
#

from histomicstk.preprocessing.color_deconvolution import color_deconvolution
from histomicstk.preprocessing.color_deconvolution import stain_color_map


# =============================================================================
# SETTINGS
# =============================================================================
Sides = [
    {
        "WSI_MainFolder_Directory": r"C:\Users\JXB242\VSCode_HistoPatho\0-Moffitt-CodedKey_PathSlides_wROIs\0-Moffitt-CodedKey_PathSlides\InsideSlides",
        "Output_MainFolder_Directory": r"C:\Users\JXB242\VSCode_HistoPatho\0-Moffitt-CodedKey_Features_wROIs_Inside",
        "Segmentation_Output_Directory": r"C:\Users\JXB242\VSCode_HistoPatho\0-Moffitt-CodedKey_PathSlides_Segs_wROIs_Inside",
        "Excel_Filename": "Histomics_InsideTumor.xlsx"
    }
    ,
    {
        "WSI_MainFolder_Directory": r"C:\Users\JXB242\VSCode_HistoPatho\0-Moffitt-CodedKey_PathSlides_wROIs\0-Moffitt-CodedKey_PathSlides\OutsideSlides",
        "Output_MainFolder_Directory": r"C:\Users\JXB242\VSCode_HistoPatho\0-Moffitt-CodedKey_Features_wROIs_Outside",
        "Segmentation_Output_Directory": r"C:\Users\JXB242\VSCode_HistoPatho\0-Moffitt-CodedKey_PathSlides_Segs_wROIs_Outside",
        "Excel_Filename": "Histomics_OutsideTumor.xlsx"
    }
]

Slide_Extensions = (".svs",)
GeoJSON_Extension = ".geojson"

Tissue_Min_Fraction = 0.01

Magnification = 15


#
# Foreground_Threshold = 80
# Min_Nucleus_Radius = 3
# Max_Nucleus_Radius = 8
#

Local_Max_Search_Radius = 6
Min_Nucleus_Area = 40

Tissue_Edge_Buffer_Pixels = 100

Visualization_Scale = 2

Save_Segmentation_Overlay = True


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def Find_SVS_Files(Folder_Path):
    return ns.natsorted([
        f for f in os.listdir(Folder_Path)
        if f.lower().endswith(Slide_Extensions)
    ])


def Find_GeoJSON_For_Slide(Slide_Path):
    # Sibling file with the same base name, .geojson extension.
    Base, _ = os.path.splitext(Slide_Path)
    Candidate = Base + GeoJSON_Extension
    if os.path.exists(Candidate):
        return Candidate
    return None


def Load_GeoJSON_Polygons(GeoJSON_Path):
    # Returns a list of (exterior_ring, [hole_rings...]) tuples, each ring a
    # list of [x, y] in base (level-0) slide pixel coordinates. Handles
    # FeatureCollection, Feature, and bare Polygon / MultiPolygon objects.
    with open(GeoJSON_Path, "r") as f:
        Data = json.load(f)

    Polygons = []

    def _emit(Geometry):
        if Geometry is None:
            return
        T = Geometry.get("type")
        if T == "Polygon":
            Rings = Geometry.get("coordinates", [])
            if len(Rings) == 0:
                return
            Polygons.append((Rings[0], Rings[1:]))
        elif T == "MultiPolygon":
            for Poly in Geometry.get("coordinates", []):
                if len(Poly) == 0:
                    continue
                Polygons.append((Poly[0], Poly[1:]))

    if isinstance(Data, dict):
        Top_Type = Data.get("type")
        if Top_Type == "FeatureCollection":
            for Feat in Data.get("features", []):
                _emit(Feat.get("geometry"))
        elif Top_Type == "Feature":
            _emit(Data.get("geometry"))
        elif Top_Type in ("Polygon", "MultiPolygon"):
            _emit(Data)
    elif isinstance(Data, list):
        # Some exporters dump a bare list of Features
        for Item in Data:
            if isinstance(Item, dict):
                if Item.get("type") == "Feature":
                    _emit(Item.get("geometry"))
                elif Item.get("type") in ("Polygon", "MultiPolygon"):
                    _emit(Item)

    return Polygons


def Compute_Polygon_BBox(Polygons):
    # (min_x, min_y, max_x, max_y) in base pixels covering all exterior rings.
    All_X = []
    All_Y = []
    for Exterior, _ in Polygons:
        for Pt in Exterior:
            All_X.append(Pt[0])
            All_Y.append(Pt[1])
    if not All_X:
        return None
    return (
        int(np.floor(min(All_X))),
        int(np.floor(min(All_Y))),
        int(np.ceil(max(All_X))),
        int(np.ceil(max(All_Y))),
    )


def Load_Slide_Region(Slide_Path, Magnification, Polygons=None):
    # Returns (RGB image, Metadata, Region_Offset_BasePixels, Scale_Factor).
    # If Polygons is given, loads only their (clipped) bounding box; otherwise
    # loads the whole slide. Scale_Factor maps base pixels -> returned pixels:
    #   returned_xy = (base_xy - Region_Offset) * Scale_Factor
    Tile_Source = large_image.getTileSource(Slide_Path)
    Metadata = Tile_Source.getMetadata()

    Base_Magnification = Metadata.get("magnification")
    if not Base_Magnification:
        # Slides missing the magnification field default to 40x (typical Aperio scan).
        Base_Magnification = 40.0

    Region_Kwargs = dict(units="base_pixels")
    Region_Offset = (0, 0)

    if Polygons:
        BBox = Compute_Polygon_BBox(Polygons)
        if BBox is not None:
            Min_X, Min_Y, Max_X, Max_Y = BBox
            Slide_W = int(Metadata.get("sizeX") or (Max_X + 1))
            Slide_H = int(Metadata.get("sizeY") or (Max_Y + 1))
            Min_X = max(0, Min_X)
            Min_Y = max(0, Min_Y)
            Max_X = min(Slide_W, Max_X)
            Max_Y = min(Slide_H, Max_Y)
            Width = max(1, Max_X - Min_X)
            Height = max(1, Max_Y - Min_Y)
            Region_Kwargs.update(dict(
                left=Min_X, top=Min_Y, width=Width, height=Height
            ))
            Region_Offset = (Min_X, Min_Y)

    Image = Tile_Source.getRegion(
        region=Region_Kwargs,
        scale=dict(magnification=Magnification),
        format=large_image.tilesource.TILE_FORMAT_NUMPY
    )[0]

    if Image.shape[-1] == 4:
        Image = Image[:, :, :3]

    Scale_Factor = float(Magnification) / float(Base_Magnification)

    return Image.astype(np.uint8), Metadata, Region_Offset, Scale_Factor


def Rasterize_Polygons_To_Mask(Polygons, Region_Offset, Scale_Factor, Image_Shape):
    # Boolean mask of polygon interiors (with holes subtracted) in returned-image
    # pixel coordinates. Image_Shape is (H, W, ...).
    from skimage.draw import polygon as sk_polygon

    H, W = Image_Shape[:2]
    Mask = np.zeros((H, W), dtype=bool)

    Off_X, Off_Y = Region_Offset

    for Exterior, Holes in Polygons:
        Ex = np.asarray(Exterior, dtype=float)
        if Ex.size == 0:
            continue
        Ex_X = (Ex[:, 0] - Off_X) * Scale_Factor
        Ex_Y = (Ex[:, 1] - Off_Y) * Scale_Factor
        rr, cc = sk_polygon(Ex_Y, Ex_X, shape=(H, W))
        Mask[rr, cc] = True
        for Hole in Holes:
            Hl = np.asarray(Hole, dtype=float)
            if Hl.size == 0:
                continue
            Hl_X = (Hl[:, 0] - Off_X) * Scale_Factor
            Hl_Y = (Hl[:, 1] - Off_Y) * Scale_Factor
            rr, cc = sk_polygon(Hl_Y, Hl_X, shape=(H, W))
            Mask[rr, cc] = False

    return Mask


def Generate_Tissue_Mask(Whole_Slide_RGB):
    Tissue_Mask = np.any(Whole_Slide_RGB < 220, axis=2)

    Tissue_Mask = sp.ndimage.binary_fill_holes(Tissue_Mask)
    Tissue_Mask = sp.ndimage.binary_opening(
        Tissue_Mask,
        structure=np.ones((5, 5))
    )

    return Tissue_Mask.astype(bool)


def Segment_Nuclei(Whole_Slide_RGB, Tissue_Mask):
    import skimage.morphology
    import skimage.measure
    import skimage.segmentation
    import skimage.feature
    import scipy.ndimage as ndi

    ###
    from skimage.filters import threshold_otsu
    ###

    Stain_Matrix = np.array([
        stain_color_map["hematoxylin"],
        stain_color_map["eosin"],
        stain_color_map["null"]
    ]).T

    Stains = color_deconvolution(
        Whole_Slide_RGB,
        Stain_Matrix
    ).Stains

    Hematoxylin = Stains[:, :, 0]

    #
    #     Nuclei_Foreground = np.logical_and(
    #         Hematoxylin < Foreground_Threshold,
    #         Tissue_Mask
    #     )
    #

    ###
    Tissue_Values = Hematoxylin[Tissue_Mask]
    Auto_Threshold = float(threshold_otsu(Tissue_Values))
    Nuclei_Foreground = np.logical_and(
        Hematoxylin < Auto_Threshold,
        Tissue_Mask
    )
    ###

    Nuclei_Foreground = skimage.morphology.remove_small_objects(
        Nuclei_Foreground.astype(bool),
        min_size=Min_Nucleus_Area
    )

    Nuclei_Foreground = skimage.morphology.remove_small_holes(
        Nuclei_Foreground,
        area_threshold=Min_Nucleus_Area
    )

    Distance = ndi.distance_transform_edt(Nuclei_Foreground)

    ###
    Distance = ndi.gaussian_filter(Distance, sigma=1.0)
    ###

    #
    #     Coordinates = skimage.feature.peak_local_max(
    #         Distance,
    #         min_distance=4,
    #         threshold_abs=2,
    #         labels=Nuclei_Foreground
    #     )
    #

    ###
    Coordinates = skimage.feature.peak_local_max(
        Distance,
        min_distance=Local_Max_Search_Radius,
        labels=Nuclei_Foreground
    )
    ###

    Markers = np.zeros(Distance.shape, dtype=np.int32)

    for i, coord in enumerate(Coordinates, start=1):
        Markers[coord[0], coord[1]] = i

    Nuclei_Label = skimage.segmentation.watershed(
        -Distance,
        markers=Markers,
        mask=Nuclei_Foreground
    )

    Nuclei_Label = htk.segmentation.label.area_open(
        Nuclei_Label.astype(np.int32),
        Min_Nucleus_Area
    ).astype(np.int32)

    return Nuclei_Label, Hematoxylin


def Count_Labels(Label_Image):
    Unique = np.unique(Label_Image)
    return int((Unique > 0).sum())


def Remove_Edge_Labels(Nuclei_Label, Tissue_Mask, Buffer_Pixels):
    # Drop any nucleus label that touches the (Buffer_Pixels-dilated)
    # tissue/background interface. Whole-label removal (not pixel-wise clipping)
    # so feature extraction never sees clipped sliver nuclei.
    # Returns (Cleaned_Label, Cleanup_Applied_Flag, Number_Of_Labels_Removed).
    Background_Mask = ~Tissue_Mask

    if not Background_Mask.any():
        return Nuclei_Label, False, 0

    Buffer_Mask = sp.ndimage.binary_dilation(
        Background_Mask,
        structure=np.ones((3, 3)),
        iterations=Buffer_Pixels
    )

    Bad_Labels = np.unique(Nuclei_Label[Buffer_Mask])
    Bad_Labels = Bad_Labels[Bad_Labels > 0]

    if Bad_Labels.size == 0:
        return Nuclei_Label, True, 0

    Cleaned_Label = Nuclei_Label.copy()
    Cleaned_Label[np.isin(Cleaned_Label, Bad_Labels)] = 0

    return Cleaned_Label.astype(np.int32), True, int(Bad_Labels.size)


def Extract_Nuclei_Features(Nuclei_Label, Hematoxylin):
    if np.max(Nuclei_Label) == 0:
        return pd.DataFrame()

    Feature_Table = htk.features.compute_nuclei_features(
        im_label=Nuclei_Label,
        im_nuclei=Hematoxylin,
        im_cytoplasm=None,
        morphometry_features_flag=True,
        fsd_features_flag=False,
        intensity_features_flag=True,
        gradient_features_flag=False,
        haralick_features_flag=False
    )

    return Feature_Table


def Aggregate_Slide_Features(Feature_Table):
    if Feature_Table.empty:
        return {}

    Numeric_Features = Feature_Table.select_dtypes(include=[np.number])

    Output = {}

    for Column in Numeric_Features.columns:
        Values = Numeric_Features[Column]
        Values = Values.replace([np.inf, -np.inf], np.nan).dropna()

        if len(Values) == 0:
            continue

        Output[f"{Column}_mean"] = Values.mean()
        Output[f"{Column}_median"] = Values.median()
        Output[f"{Column}_std"] = Values.std()
        Output[f"{Column}_min"] = Values.min()
        Output[f"{Column}_max"] = Values.max()

    Output["N_Nuclei"] = len(Feature_Table)

    return Output


def Save_Segmentation_Overlay_Image(
    PatientID,
    Slide_File,
    Whole_Slide_RGB,
    Nuclei_Label,
    Segmentation_Output_Directory,
    Suffix=""
):
    from skimage.segmentation import find_boundaries

    Patient_Output_Dir = os.path.join(
        Segmentation_Output_Directory,
        PatientID
    )

    os.makedirs(Patient_Output_Dir, exist_ok=True)

    Base_Name = os.path.splitext(Slide_File)[0]

    Visualization_Image = Whole_Slide_RGB[
        ::Visualization_Scale,
        ::Visualization_Scale
    ].copy()

    Visualization_Label = Nuclei_Label[
        ::Visualization_Scale,
        ::Visualization_Scale
    ]

    Visualization_Image = Visualization_Image.astype(np.uint8)

    Boundaries = find_boundaries(
        Visualization_Label,
        mode="inner"
    )

    Visualization_Image[Boundaries] = [0, 255, 0]

    Overlay_Path = os.path.join(
        Patient_Output_Dir,
        f"{Base_Name}_SegmentationOverlay{Suffix}.png"
    )

    plt.imsave(
        Overlay_Path,
        Visualization_Image
    )

    plt.close("all")

    print("Saved segmentation overlay:")
    print(Overlay_Path)


def Save_Global_Excel(Global_Results, Global_Output_Excel):
    # Atomic write: write to a temp file in the same directory, then os.replace.
    # os.replace is atomic on both POSIX and Windows, so a crash mid-write
    # cannot leave a half-written Excel at Global_Output_Excel.
    if not Global_Results:
        return

    # Tmp filename keeps the original extension so pandas/openpyxl can auto-
    # select the writer and pass extension validation. ".xlsx" -> ".tmp.xlsx".
    Base, Ext = os.path.splitext(Global_Output_Excel)
    Tmp_Path = Base + ".tmp" + Ext

    pd.DataFrame(Global_Results).to_excel(
        Tmp_Path,
        index=False
    )

    os.replace(Tmp_Path, Global_Output_Excel)

    print("Intermediate Excel saved:")
    print(Global_Output_Excel)


def Load_Existing_Results(Global_Output_Excel):
    # Resume support: load previously completed rows from the Excel (if any).
    # Returns (Global_Results list, set of (PatientID, Slide) already done).
    if not os.path.exists(Global_Output_Excel):
        return [], set()

    try:
        Existing_DF = pd.read_excel(Global_Output_Excel)
    except Exception as e:
        print(f"Could not read existing Excel ({e}); starting fresh.")
        return [], set()

    if Existing_DF.empty or "PatientID" not in Existing_DF.columns or "Slide" not in Existing_DF.columns:
        return [], set()

    Global_Results = Existing_DF.to_dict(orient="records")

    Done_Keys = set(
        (str(r["PatientID"]), str(r["Slide"]))
        for r in Global_Results
    )

    print(
        f"Resuming from existing Excel: "
        f"{len(Global_Results)} row(s) already present, "
        f"{len(Done_Keys)} (patient, slide) pair(s) will be skipped."
    )

    return Global_Results, Done_Keys


# =============================================================================
# MAIN LOOP OVER SIDES
# =============================================================================

for side in Sides:
    WSI_MainFolder_Directory = side["WSI_MainFolder_Directory"]
    Output_MainFolder_Directory = side["Output_MainFolder_Directory"]
    Segmentation_Output_Directory = side["Segmentation_Output_Directory"]
    Excel_Filename = side["Excel_Filename"]


    # =============================================================================
    # OUTPUT SETUP
    # =============================================================================

    os.makedirs(Output_MainFolder_Directory, exist_ok=True)
    os.makedirs(Segmentation_Output_Directory, exist_ok=True)

    Global_Output_Excel = os.path.join(
        Output_MainFolder_Directory,
        Excel_Filename
    )


    # =============================================================================
    # RESUME FROM EXISTING EXCEL
    # =============================================================================

    Global_Results, Done_Keys = Load_Existing_Results(Global_Output_Excel)


    # =============================================================================
    # PATIENT IDS
    # =============================================================================

    PatientIDs = ns.natsorted([
        f for f in os.listdir(WSI_MainFolder_Directory)
        if os.path.isdir(os.path.join(WSI_MainFolder_Directory, f))
    ])


    # =============================================================================
    # MAIN LOOP
    # =============================================================================

    Total_Start_Time = time.time()

    for PatientID in PatientIDs:

        print("\n======================================")
        print(f"Processing patient: {PatientID}")
        print("======================================")

        Patient_Dir = os.path.join(
            WSI_MainFolder_Directory,
            PatientID
        )

        if not os.path.isdir(Patient_Dir):
            print("Missing patient folder, skipping.")
            continue

        Slide_Files = Find_SVS_Files(Patient_Dir)

        if len(Slide_Files) == 0:
            print("No .svs slides found, skipping.")
            continue

        print(f"Found {len(Slide_Files)} slide(s).")

        for Slide_File in Slide_Files:

            if (str(PatientID), str(Slide_File)) in Done_Keys:
                print(f"Already processed, skipping: {PatientID} | {Slide_File}")
                continue

            Slide_Start_Time = time.time()

            Slide_Path = os.path.join(Patient_Dir, Slide_File)

            print("--------------------------------------")
            print(f"Processing slide: {Slide_File}")

            try:

                GeoJSON_Path = Find_GeoJSON_For_Slide(Slide_Path)
                Polygons = []
                Used_GeoJSON_ROI = False

                if GeoJSON_Path is not None:
                    print(f"Found GeoJSON ROI: {os.path.basename(GeoJSON_Path)}")
                    Polygons = Load_GeoJSON_Polygons(GeoJSON_Path)
                    if Polygons:
                        Used_GeoJSON_ROI = True
                        print(f"Loaded {len(Polygons)} polygon(s) from GeoJSON.")
                    else:
                        print("GeoJSON had no usable polygons; falling back to whole slide.")

                print("Loading slide region..." if Used_GeoJSON_ROI else "Loading whole slide...")
                Whole_Slide_RGB, Metadata, Region_Offset, Scale_Factor = Load_Slide_Region(
                    Slide_Path,
                    Magnification,
                    Polygons=Polygons if Used_GeoJSON_ROI else None
                )

                Slide_Height = Whole_Slide_RGB.shape[0]
                Slide_Width = Whole_Slide_RGB.shape[1]

                print(f"Region loaded: {Slide_Width} x {Slide_Height}")

                print("Generating tissue mask...")
                Tissue_Mask = Generate_Tissue_Mask(Whole_Slide_RGB)

                if Used_GeoJSON_ROI:
                    print("Applying GeoJSON ROI mask...")
                    ROI_Mask = Rasterize_Polygons_To_Mask(
                        Polygons,
                        Region_Offset,
                        Scale_Factor,
                        Whole_Slide_RGB.shape
                    )
                    Tissue_Mask = np.logical_and(Tissue_Mask, ROI_Mask)

                Tissue_Fraction = Tissue_Mask.mean()
                print(f"Tissue fraction: {Tissue_Fraction:.4f}")

                if Tissue_Fraction < Tissue_Min_Fraction:
                    print("Too little tissue detected, skipping slide.")
                    continue

                print("Skipping stain normalization because staining is expected to be consistent.")

                print("Segmenting nuclei...")
                Nuclei_Label, Hematoxylin = Segment_Nuclei(
                    Whole_Slide_RGB,
                    Tissue_Mask
                )

                N_Nuclei_Before_Edge_Cleanup = Count_Labels(Nuclei_Label)
                print(f"Nuclei segmented: {N_Nuclei_Before_Edge_Cleanup}")

                if N_Nuclei_Before_Edge_Cleanup == 0:
                    print("No nuclei detected, skipping slide.")
                    continue

                if Save_Segmentation_Overlay:
                    print("Saving intermediate segmentation overlay...")
                    Save_Segmentation_Overlay_Image(
                        PatientID=PatientID,
                        Slide_File=Slide_File,
                        Whole_Slide_RGB=Whole_Slide_RGB,
                        Nuclei_Label=Nuclei_Label,
                        Segmentation_Output_Directory=Segmentation_Output_Directory,
                        Suffix="_Intermediate"
                    )

                print(f"Removing edge labels (buffer = {Tissue_Edge_Buffer_Pixels} px)...")
                Nuclei_Label, Edge_Cleanup_Applied, N_Labels_Removed_At_Edge = Remove_Edge_Labels(
                    Nuclei_Label,
                    Tissue_Mask,
                    Tissue_Edge_Buffer_Pixels
                )

                if Edge_Cleanup_Applied:
                    print(
                        f"Edge cleanup: removed {N_Labels_Removed_At_Edge} label(s) "
                        f"of {N_Nuclei_Before_Edge_Cleanup}."
                    )
                else:
                    print("Edge cleanup: no background detected, nothing to remove.")

                N_Nuclei = Count_Labels(Nuclei_Label)
                print(f"Nuclei after edge cleanup: {N_Nuclei}")

                if N_Nuclei == 0:
                    print("All nuclei were on tissue edges and removed; skipping slide.")
                    continue

                if Save_Segmentation_Overlay:
                    print("Saving final segmentation overlay...")
                    Save_Segmentation_Overlay_Image(
                        PatientID=PatientID,
                        Slide_File=Slide_File,
                        Whole_Slide_RGB=Whole_Slide_RGB,
                        Nuclei_Label=Nuclei_Label,
                        Segmentation_Output_Directory=Segmentation_Output_Directory
                    )

                print("Extracting nuclei features...")
                Feature_Table = Extract_Nuclei_Features(
                    Nuclei_Label,
                    Hematoxylin
                )

                if Feature_Table.empty:
                    print("No features extracted, skipping slide.")
                    continue

                print("Aggregating whole-slide features...")
                Slide_Features = Aggregate_Slide_Features(Feature_Table)

                Slide_Runtime_Minutes = (time.time() - Slide_Start_Time) / 60

                Record = {
                    "PatientID": PatientID,
                    "Slide": Slide_File,
                    "Slide_Path": Slide_Path,
                    "Slide_Width": Slide_Width,
                    "Slide_Height": Slide_Height,
                    "Magnification": Magnification,
                    "Tissue_Fraction": Tissue_Fraction,
                    "N_Nuclei": N_Nuclei,
                    "Slide_Runtime_Minutes": Slide_Runtime_Minutes,
                    **Slide_Features
                }

                Global_Results.append(Record)
                Done_Keys.add((str(PatientID), str(Slide_File)))

                print(f"Extracted features: {PatientID} | {Slide_File}")
                print(f"Slide runtime: {Slide_Runtime_Minutes:.2f} minutes")

                Save_Global_Excel(
                    Global_Results,
                    Global_Output_Excel
                )

                ###
                del Whole_Slide_RGB, Nuclei_Label, Hematoxylin, Feature_Table
                gc.collect()
                ###

            except MemoryError:

                print(
                    f"FAILED: {PatientID} | {Slide_File} | "
                    "MemoryError. Region is too large to load into RAM. "
                    "Consider providing a GeoJSON ROI for this slide."
                )

            except Exception as e:

                print(
                    f"FAILED: {PatientID} | {Slide_File} | {e}"
                )


    # =============================================================================
    # FINAL SAVE GLOBAL EXCEL
    # =============================================================================

    Save_Global_Excel(
        Global_Results,
        Global_Output_Excel
    )

    Total_Runtime_Hours = (time.time() - Total_Start_Time) / 3600

    print("\n======================================")
    print("Global histomics Excel saved:")
    print(Global_Output_Excel)
    print(f"Total runtime: {Total_Runtime_Hours:.2f} hours")
    print("======================================")
