from monai.transforms import *





train_transforms = Compose([LoadImaged(keys=["image"]),
                            EnsureChannelFirstd(keys=["image"]),
##                                    Orientationd(keys=["image"], axcodes="RAS"),
                            ScaleIntensityRanged(
                                keys=["image"], a_min=0.0, a_max=255.0,
                                b_min=0.0, b_max=1.0, clip=True),
##                                    CropForegroundd(keys=["image"], source_key="image"),
                            #Resized(keys=["image"], mode="trilinear", align_corners=True,
                            #        spatial_size=(192, 192, 96)), 
                            ToTensord(keys=["image"])                                   
                            ])

transformed_image = train_transforms({'image': "/data/fenghetang/report_generation/Chest_New_nitfy_2/Chest_New_nitfy_2/0.nii.gz"})
image = transformed_image['image'].permute(0,3,2,1).contiguous()
print(image.shape)