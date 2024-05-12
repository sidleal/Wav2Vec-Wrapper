#----CORAA v1 test
#eval coraa model in coraa dataset *run for replication candido junior et al (2022)*
python ../../test.py --output_csv d:/results/output_coraa_coraa.csv --config_path ./configs/config_eval_coraa.json --checkpoint_path_or_name d:/models/wav2vec2-large-xlsr-coraa-portuguese --no_kenlm

#eval gris model in coraa dataset *reported*
python ../../test.py --output_csv d:/results/output_gris_coraa.csv --config_path ./configs/config_eval_coraa.json --checkpoint_path_or_name d:/models/bp400-xlsr --no_kenlm

#eval nurc-sp-1 model in coraa dataset
python ../../test.py --output_csv d:/results/output_nurcsp1_coraa.csv --config_path ./configs/config_eval_coraa.json --checkpoint_path_or_name d:/models/wav2vec2-large-xlsr-nurc-sp-1-portuguese --no_kenlm

#eval nurc-sp-2 model in coraa dataset
python ../../test.py --output_csv d:/results/output_nurcsp2_coraa.csv --config_path ./configs/config_eval_coraa.json --checkpoint_path_or_name d:/models/wav2vec2-large-xlsr-nurc-sp-2-portuguese --no_kenlm

#----NURCSP test
#eval coraa model in nurcsp dataset
python ../../test.py --output_csv d:/results/output_coraa_nurcsp.csv --config_path ./configs/config_eval_nurcsp.json --checkpoint_path_or_name d:/models/wav2vec2-large-xlsr-coraa-portuguese --no_kenlm

#eval gris model in nurcsp dataset
python ../../test.py --output_csv d:/results/output_gris_nurcsp.csv --config_path ./configs/config_eval_nurcsp.json --checkpoint_path_or_name d:/models/bp400-xlsr --no_kenlm

#eval nurc-sp-1 model in nurcsp dataset
python ../../test.py --output_csv d:/results/output_nurcsp1_nurcsp.csv --config_path ./configs/config_eval_nurcsp.json --checkpoint_path_or_name d:/models/wav2vec2-large-xlsr-nurc-sp-1-portuguese --no_kenlm

#eval nurc-sp-2 model in nurcsp dataset
python ../../test.py --output_csv d:/results/output_nurcsp2_nurcsp.csv --config_path ./configs/config_eval_nurcsp.json --checkpoint_path_or_name d:/models/wav2vec2-large-xlsr-nurc-sp-2-portuguese --no_kenlm


#----Commonvoice pt test *reported*
#eval coraa model in commonvoice dataset
python ../../test.py --output_csv d:/results/output_coraa_commonvoice.csv --config_path ./configs/config_eval_commonvoice.json --checkpoint_path_or_name d:/models/wav2vec2-large-xlsr-coraa-portuguese --no_kenlm

#eval gris model in commonvoice dataset *reported*
python ../../test.py --output_csv d:/results/output_gris_commonvoice.csv --config_path ./configs/config_eval_commonvoice.json --checkpoint_path_or_name d:/models/bp400-xlsr --no_kenlm

#eval nurc-sp-1 model in commonvoice dataset
python ../../test.py --output_csv d:/results/output_nurcsp1_commonvoice.csv --config_path ./configs/config_eval_commonvoice.json --checkpoint_path_or_name d:/models/wav2vec2-large-xlsr-nurc-sp-1-portuguese --no_kenlm

#eval nurc-sp-2 model in commonvoice dataset
python ../../test.py --output_csv d:/results/output_nurcsp2_commonvoice.csv --config_path ./configs/config_eval_commonvoice.json --checkpoint_path_or_name d:/models/wav2vec2-large-xlsr-nurc-sp-2-portuguese --no_kenlm


