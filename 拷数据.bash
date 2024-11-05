scp BUPT426@2.zhou.narc.letmefly.xyz:C:/Users/BUPT426/Desktop/新建文件夹/评价.docx D:/Data-R1

C:\Users\BUPT426\Desktop\RoseAgg\RoseAgg_Latest\FL_Backdoor_CV


scp -r BUPT426@2.zhou.narc.letmefly.xyz:C:/Users/BUPT426/Desktop/RoseAgg/RoseAgg_Latest/FL_Backdoor_CV/2024-10-02_03-37-50/ D:/Data-R1
scp -r BUPT426@2.zhou.narc.letmefly.xyz:C:/Users/BUPT426/Desktop/RoseAgg/RoseAgg_Latest/FL_Backdoor_CV/2024-10-02_21-40-11/ D:/Data-R1
scp -r BUPT426@2.zhou.narc.letmefly.xyz:C:/Users/BUPT426/Desktop/RoseAgg/RoseAgg_Latest/FL_Backdoor_CV/2024-10-27_16-25-41/ D:/Data-R1
scp -r BUPT426@2.zhou.narc.letmefly.xyz:C:/Users/BUPT426/Desktop/RoseAgg/RoseAgg_Latest/FL_Backdoor_CV/2024-10-29_22-41-06/ D:/Data-R1

scp -r BUPT426@2.zhou.narc.letmefly.xyz:C:/Users/BUPT426/Desktop/RoseAgg/RoseAgg_Latest/FL_Backdoor_CV/2024-10-02_09-11-28/ D:/Data-R1
scp -r BUPT426@2.zhou.narc.letmefly.xyz:C:/Users/BUPT426/Desktop/RoseAgg/RoseAgg_Latest/FL_Backdoor_CV/2024-10-03_03-09-36/ D:/Data-R1
scp -r BUPT426@2.zhou.narc.letmefly.xyz:C:/Users/BUPT426/Desktop/RoseAgg/RoseAgg_Latest/FL_Backdoor_CV/2024-10-28_00-55-46/ D:/Data-R1
scp -r BUPT426@2.zhou.narc.letmefly.xyz:C:/Users/BUPT426/Desktop/RoseAgg/RoseAgg_Latest/FL_Backdoor_CV/2024-10-30_04-48-52/ D:/Data-R1

scp -r BUPT426@2.zhou.narc.letmefly.xyz:C:/Users/BUPT426/Desktop/RoseAgg/RoseAgg_Latest/FL_Backdoor_CV/2024-10-02_12-55-32/ D:/Data-R1
scp -r BUPT426@2.zhou.narc.letmefly.xyz:C:/Users/BUPT426/Desktop/RoseAgg/RoseAgg_Latest/FL_Backdoor_CV/2024-10-03_06-48-00/ D:/Data-R1
scp -r BUPT426@2.zhou.narc.letmefly.xyz:C:/Users/BUPT426/Desktop/RoseAgg/RoseAgg_Latest/FL_Backdoor_CV/2024-10-28_05-59-22/ D:/Data-R1
scp -r BUPT426@2.zhou.narc.letmefly.xyz:C:/Users/BUPT426/Desktop/RoseAgg/RoseAgg_Latest/FL_Backdoor_CV/2024-10-30_08-40-55/ D:/Data-R1






files = {
    "MR": {
        "flame": "10020337",
        "fltrust": "10022140",
        "foolsgold": "10271625",
        "secfft": "10292241",
    },
    "EDGE_CASE": {
        "flame": "10020911",
        "fltrust": "10030309",
        "foolsgold": "10280055",
        "secfft": "10300448",
    },
    "NEUROTOXIN": {
        "flame": "10021255",
        "fltrust": "10030648",
        "foolsgold": "10280559",
        "secfft": "10300840",
    },
}

attack_defense_data = {
    "MR": {
        "FLAME": "FL_Backdoor_CV/2024-10-02_03-37-50/",
        "FLTRUST": "FL_Backdoor_CV/2024-10-02_21-40-11/",
        "FOOLSGOLD": "FL_Backdoor_CV/2024-10-27_16-25-41/",
        "SECFFT": "FL_Backdoor_CV/2024-10-29_22-41-06/",
    },
    "EDGE_CASE": {
        "FLAME": "FL_Backdoor_CV/2024-10-02_09-11-28/",
        "FLTRUST": "FL_Backdoor_CV/2024-10-03_03-09-36/",
        "FOOLSGOLD": "FL_Backdoor_CV/2024-10-28_00-55-46/",
        "SECFFT": "FL_Backdoor_CV/2024-10-30_04-48-52/",
    },
    "NEUROTOXIN": {
        "FLAME": "FL_Backdoor_CV/2024-10-02_12-55-32/",
        "FLTRUST": "FL_Backdoor_CV/2024-10-03_06-48-00/",
        "FOOLSGOLD": "FL_Backdoor_CV/2024-10-28_05-59-22/",
        "SECFFT": "FL_Backdoor_CV/2024-10-30_08-40-55/",
    },
}