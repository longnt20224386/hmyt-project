import os,sys,json,hashlib,time,importlib.util
from pathlib import Path
os.environ['QT_QPA_PLATFORM']='offscreen'
import numpy as np,SimpleITK as sitk
from PyQt5.QtWidgets import QApplication,QMessageBox,QFileDialog
src=Path(__file__).resolve().parents[1]/'code/segmentation_14.py'
p=Path(sys.argv[1])
out=Path(__file__).resolve().parent; results={'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'tests':[]}
spec=importlib.util.spec_from_file_location('tested_app',src); m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
app=QApplication.instance() or QApplication([])
messages=[]
for kind in ['warning','critical','information']:
 setattr(QMessageBox,kind,lambda *a,k=kind: messages.append({'kind':k,'text':str(a[2])}))
QMessageBox.question=lambda *a: QMessageBox.No
w=m.MedicalSegmentationApp();results['tests'].append({'test':'GUI initialization offscreen','pass':True})
paths=iter([str(p/'CTres.nii.gz'),str(p/'PET.nii.gz')]);QFileDialog.getOpenFileName=lambda *a:(next(paths),'')
w.load_images();results['tests'].append({'test':'Load supplied CT and PET','shape':list(w.ct_array.shape),'pass':w.ct_array.shape==w.suv_array.shape})
w.refine_all_masks();results['tests'].append({'test':'Refine without ROI','messages':messages.copy()});messages.clear()
pet=w.suv_array.copy();suvim=sitk.ReadImage(str(p/'SUV.nii.gz'));suv=sitk.GetArrayFromImage(suvim)
valid=suv>0.01;ratio=pet[valid]/suv[valid];results['pet_to_suv_ratio']={'median':float(np.median(ratio)),'min':float(ratio.min()),'max':float(ratio.max())}
results['geometry']={'size':list(w.ct_image.GetSize()),'spacing_mm':list(w.ct_image.GetSpacing()),'pet_max':float(pet.max()),'suv_max':float(suv.max())}
# Existing masks are compared without claiming provenance or independent accuracy.
g=sitk.GetArrayFromImage(sitk.ReadImage(str(p/'SEG.nii.gz')))>0
pred=sitk.GetArrayFromImage(sitk.ReadImage(str(p/'segment_Dung.nii')))>0
inter=int(np.count_nonzero(g&pred));ng=int(g.sum());nn=int(pred.sum());vv=float(np.prod(w.ct_image.GetSpacing())/1000)
results['existing_mask_overlap_unverified_provenance']={'reference':'SEG.nii.gz','candidate':'segment_Dung.nii','reference_voxels':ng,'candidate_voxels':nn,'intersection':inter,'dice':2*inter/(ng+nn),'iou':inter/(ng+nn-inter),'voxel_recall':inter/ng,'voxel_precision':inter/nn,'reference_ml':ng*vv,'candidate_ml':nn*vv,'fp_ml':(nn-inter)*vv,'fn_ml':(ng-inter)*vv}
# Actual method on small synthetic inputs, no patient GT passed as ROI.
w.show_current_slice=lambda:None
w.show_refine_results=lambda *a:None
w.ct_image=sitk.Image([8,8,8],sitk.sitkFloat32)
a=np.zeros((8,8,8),np.float32);a[2:5,2:5,2:5]=3;a[3,3,3]=20
w.suv_array=a;w.mask_array=(a>0).astype(np.uint8);w.base_refine_mask=None
w.refine_all_masks();results['tests'].append({'test':'Actual refine 27 voxels at 3 and center at 20','output_voxels':int((w.mask_array>0).sum()),'expected':1,'pass':int((w.mask_array>0).sum())==1})
outputs=[]
for value in [10,60]:
 w.mask_array=(a>0).astype(np.uint8);w.base_refine_mask=None;w.update_refine_threshold(value);w.refine_all_masks();outputs.append(w.mask_array.copy())
results['tests'].append({'test':'Global threshold control 10 versus 60','identical_outputs':bool(np.array_equal(*outputs))})
# More than 255 components, reproduces mask label dtype limit.
a=np.zeros((20,20,20),np.float32);a[::2,::2,::2]=5
w.suv_array=a;w.mask_array=(a>0).astype(np.uint8);w.base_refine_mask=None;messages.clear();w.refine_all_masks()
results['tests'].append({'test':'1000 disconnected components','messages':messages.copy(),'refine_completed_without_error':not any(x['kind']=='critical' for x in messages)})
out.joinpath('test_results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(results,ensure_ascii=False,indent=2))
w.close()
