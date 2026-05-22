import os
import SimpleITK as sitk

def inspect_patient_images(patient_dir):
    print(f"Inspecting files in: {patient_dir}")
    files = os.listdir(patient_dir)
    nii_files = [f for f in files if f.endswith(".nii.gz")]
    
    for f in sorted(nii_files):
        path = os.path.join(patient_dir, f)
        img = sitk.ReadImage(path)
        print(f"\nFile: {f}")
        print(f"  Size:      {img.GetSize()}")
        print(f"  Spacing:   {img.GetSpacing()}")
        print(f"  Origin:    {img.GetOrigin()}")
        print(f"  Direction: {img.GetDirection()}")
        print(f"  PixelType: {img.GetPixelIDTypeAsString()}")

if __name__ == "__main__":
    sample_dir = "/mnt/d/A1_RainSun_20240916/1-UWMadison/IDiA-Lab/Medical_Images_Public/Brain-Mets-Lung-MRI-Path-Segs_CLEAN/YG_0AXGKD8AFJGS"
    inspect_patient_images(sample_dir)
