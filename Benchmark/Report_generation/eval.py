import json

# gt_path = "/data1/rongshengwang/Code/ECAMP_CT/Report_Gen/CT-Rate_split.json"
# with open(gt_path, 'r') as f:
#     gt_data = json.load(f)

# gt_reports = []
# gt_ids = []
# for item in gt_data['test']:
#     gt_reports.append(item['report'].strip())
#     gt_ids.append(item['id'])  # e.g., CT-Rate_0001

# print(gt_reports[:5])
# print(gt_ids[:5])


# 分别储存了 image id 和对应 ground truth report


# CT-Rate
random_path = "/data1/rongshengwang/Code/ECAMP_CT/Report_Gen/results/CT-Rate/random_vit_base_20260121-053257/Enc2Dec-30_0_test_generated.json"
mae_path = "/data1/rongshengwang/Code/ECAMP_CT/Report_Gen/results/CT-Rate/mae_vit_base_20260120-162356/Enc2Dec-2_0_test_generated.json"
m3ae_path = "/data1/rongshengwang/Code/ECAMP_CT/Report_Gen/results/CT-Rate/m3ae_vit_base_20260121-053313/Enc2Dec-20_0_test_generated.json"
mrm_path = "/data1/rongshengwang/Code/ECAMP_CT/Report_Gen/results/CT-Rate/mrm_vit_base_20260120-162536/Enc2Dec-8_0_test_generated.json"
ctclip_path = "/data1/rongshengwang/Code/ECAMP_CT/Report_Gen/results/CT-Rate/ctclip_vit_base_20260120-162544/Enc2Dec-3_0_test_generated.json"
m3d_path = "/data1/rongshengwang/Code/ECAMP_CT/Report_Gen/results/CT-Rate/m3d_vit_base_20260120-162552/Enc2Dec-2_0_test_generated.json"
fvlm_path = "/data1/rongshengwang/Code/ECAMP_CT/Report_Gen/results/CT-Rate/fvlm_vit_base_20260120-162711/Enc2Dec-3_0_test_generated.json"
hlip_path = "/data1/rongshengwang/Code/ECAMP_CT/Report_Gen/results/CT-Rate/hlip_vit_base_20260121-053345/Enc2Dec-26_0_test_generated.json"
simcrop_path = "/data1/rongshengwang/Code/ECAMP_CT/Report_Gen/results/CT-Rate/simcrop_vit_base_20260120-162743/Enc2Dec-2_0_test_generated.json"
seap_path = "/data1/rongshengwang/Code/ECAMP_CT/Report_Gen/results/CT-Rate/simcrop++_vit_base_20260121-154135/Enc2Dec-12_0_test_generated.json"


# CTRG
random_path = "/data1/rongshengwang/Code/ECAMP_CT/Report_Gen/results/ctrg/random_vit_base_20251230-081822/Enc2Dec-24_0_test_generated.json"
mae_path = "/data1/rongshengwang/Code/ECAMP_CT/Report_Gen/results/ctrg/mae_vit_base_20251230-154005/Enc2Dec-5_0_test_generated.json"
m3ae_path = "/data1/rongshengwang/Code/ECAMP_CT/Report_Gen/results/ctrg/m3ae_vit_base_20251230-154025/Enc2Dec-27_0_test_generated.json"
mrm_path = "/data1/rongshengwang/Code/ECAMP_CT/Report_Gen/results/ctrg/mrm_vit_base_20251231-080602/Enc2Dec-27_0_test_generated.json"
ctclip_path = "/data1/rongshengwang/Code/ECAMP_CT/Report_Gen/results/ctrg/ctclip_vit_base_20251230-154121/Enc2Dec-13_0_test_generated.json"
m3d_path = "/data1/rongshengwang/Code/ECAMP_CT/Report_Gen/results/ctrg/m3d_vit_base_20251231-110906/Enc2Dec-10_0_test_generated.json"
fvlm_path = "/data1/rongshengwang/Code/ECAMP_CT/Report_Gen/results/ctrg/fvlm_vit_base_20251230-154131/Enc2Dec-9_0_test_generated.json"
hlip_path = "/data1/rongshengwang/Code/ECAMP_CT/Report_Gen/results/ctrg/hlip_vit_base_20251231-084600/Enc2Dec-26_0_test_generated.json"
simcrop_path = "/data1/rongshengwang/Code/ECAMP_CT/Report_Gen/results/ctrg/simcrop_vit_base_20251231-131123/Enc2Dec-7_0_test_generated.json"
seap_path = "/data1/rongshengwang/Code/ECAMP_CT/Report_Gen/results/ctrg/simcrop++/Enc2Dec-9_0_test_generated.json"


test_files = [random_path, mae_path, m3ae_path, mrm_path, ctclip_path, m3d_path, fvlm_path, hlip_path, simcrop_path]

for file in test_files:
    with open(file, 'r') as f:
        pred_data = json.load(f)
    
    pred_reports = []
    gt_reports = []
    pred_ids = []

    for item in pred_data:
        pred_reports.append(item['prediction'].strip())
        pred_ids.append(item['filename'])
        gt_reports.append(item['ground_truth'].strip())

        ## ---- TODO: Add GREEN metric API here ---- ##