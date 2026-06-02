# BP
Stručný popis príloh
/analyza
analyza_vysledkov.py - vstup a výstupy sú definované v hlavičke skriptu
                     - kod na všeobecnú analýzu výsledkov 
/code
run_metacentrum_train_fold_0.pbs
run_singularity_train_fold_0.sh   - oba tieto skripty na trenovanie modelov na Metacentre máme od https://github.com/tomasvicar/nnunet_metacentrum_example/tree/master

/code/trainers
        - nnUNetTrainerMedNeXt.py - implementácia architektúry MedNeXt v1 adaptovaná pre nnU-Net v2
                                  - povodný MedNeXt v1  prevzatý z https://github.com/MIC-DKFZ/MedNeXt
                                  - dokumentácia nnU-Net v2 custom trainers  https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/extending_nnunet.md
        - nnUNetTrainerAttNet.py - implementácia architektúry ResAtt podľa nnU-Net v2, rozšírenie ResEnc U-Net o mechanizmus attention gates na preskokových spojeniach dekodéra
                                  - referenčná implementácia attention gates prevzatá z https://github.com/ozan-oktay/Attention-Gated-Networks
                                  - ResEnc U-Net základ prevzatý z https://github.com/MIC-DKFZ/nnUNet
                                  - dokumentácia nnU-Net v2 custom trainers  https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/extending_nnunet.md
