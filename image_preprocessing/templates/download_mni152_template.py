from nilearn import datasets
import nibabel as nib

# 1. Load the template in-memory
template_img = datasets.load_mni152_template()

# 2. Save it directly to a NIfTI file on your disk!
nib.save(template_img, "mni152_template.nii.gz")
print("Saved MNI152 template successfully!")
